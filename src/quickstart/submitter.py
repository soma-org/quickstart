"""Soma data submitter — finds and submits data against open targets on Modal.

Streams shuffled source files from The Stack v2, scores them against open
targets using a GPU, and uploads to S3 + submits on-chain when the distance
threshold is met.

Deploys as a cron job that runs every 24 hours, with a 23h45m timeout.
The submission loop runs continuously within each invocation.

Deploy and run immediately:
    uv run modal deploy src/quickstart/submitter.py && uv run submit

Deploy only (auto-runs on 24h schedule):
    uv run modal deploy src/quickstart/submitter.py

Required Modal secrets (soma-secrets):
    SOMA_SECRET_KEY, HF_TOKEN,
    S3_BUCKET, S3_ACCESS_KEY_ID, S3_SECRET_ACCESS_KEY, S3_ENDPOINT_URL, S3_PUBLIC_URL
"""

import asyncio
import collections
import http.server
import os
import random
import subprocess
import sys
import threading

import modal

app = modal.App("soma-submitter")

volume = modal.Volume.from_name("soma-scoring-data", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.13")
    .apt_install("curl")
    .env({"PATH": "/root/.local/bin:/usr/local/bin:/usr/bin:/bin", "RUST_LOG": "warn"})
    .run_commands(
        "curl -sSfL https://sup.soma.org | sh",
        "sup install soma",
    )
    .pip_install("datasets>=3.0", "boto3>=1.35", "smart_open")
    .run_commands("pip install --force-reinstall --no-cache-dir soma-sdk")
)

SCORING_PORT = 9124
LOCAL_DATA_PORT = 9125
LOCAL_DATA_DIR = "/tmp/soma-local-data"
REFRESH_EVERY = 50


def stream_stack_v2():
    """Yield shuffled source files from The Stack v2 as UTF-8 bytes."""
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config
    from datasets import load_dataset
    from smart_open import open as smart_open

    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    ds = load_dataset(
        "bigcode/the-stack-v2-dedup",
        split="train",
        streaming=True,
        token=os.environ.get("HF_TOKEN"),
    ).shuffle(buffer_size=10_000)

    for row in ds:
        try:
            s3_url = f"s3://softwareheritage/content/{row['blob_id']}"
            with smart_open(
                s3_url, "rb", compression=".gz", transport_params={"client": s3}
            ) as fin:
                content = fin.read().decode(row["src_encoding"])
        except Exception:
            continue
        if not content.strip():
            continue
        data = content.encode("utf-8")
        if len(data) > 5_000:
            continue
        yield data


def prefetch_stream(stream, buffer_size=64):
    """Wrap a generator with a background-thread prefetch buffer."""
    buf = collections.deque()
    exhausted = threading.Event()
    ready = threading.Semaphore(0)
    lock = threading.Lock()

    def _fill():
        for item in stream:
            with lock:
                buf.append(item)
            ready.release()
            while True:
                with lock:
                    if len(buf) < buffer_size:
                        break
                threading.Event().wait(0.01)
        exhausted.set()
        ready.release()

    t = threading.Thread(target=_fill, daemon=True)
    t.start()

    while True:
        ready.acquire()
        with lock:
            if buf:
                yield buf.popleft()
            elif exhausted.is_set():
                return


def cosine_distance(a, b):
    """Cosine distance between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 1.0
    return 1.0 - dot / (norm_a * norm_b)


def save_local(data: bytes, checksum: str) -> str:
    """Write data to local disk and return a localhost URL."""
    os.makedirs(LOCAL_DATA_DIR, exist_ok=True)
    filename = f"{checksum}.bin"
    path = os.path.join(LOCAL_DATA_DIR, filename)
    with open(path, "wb") as f:
        f.write(data)
    return f"http://localhost:{LOCAL_DATA_PORT}/{filename}", path


def cleanup_local(path: str):
    """Remove a local data file."""
    try:
        os.remove(path)
    except OSError:
        pass


def upload_to_s3(data: bytes, checksum: str, epoch: int) -> str:
    """Upload data to an S3-compatible bucket and return a public URL."""
    import boto3

    bucket = os.environ["S3_BUCKET"]
    endpoint_url = os.environ.get("S3_ENDPOINT_URL")
    if endpoint_url and endpoint_url.rstrip("/").endswith(f"/{bucket}"):
        endpoint_url = endpoint_url.rstrip("/").removesuffix(f"/{bucket}")
    public_url = os.environ.get("S3_PUBLIC_URL")

    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=os.environ["S3_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["S3_SECRET_ACCESS_KEY"],
    )
    key = f"{epoch}/{checksum}.bin"

    s3.put_object(Bucket=bucket, Key=key, Body=data, ACL="public-read")

    if public_url:
        return f"{public_url.rstrip('/')}/{key}"
    region = s3.get_bucket_location(Bucket=bucket)["LocationConstraint"] or "us-east-1"
    return f"https://{bucket}.s3.{region}.amazonaws.com/{key}"


@app.cls(
    image=image,
    gpu="L4",
    timeout=85500,  # 23h45m
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("soma-secrets")],
)
class Submitter:
    @modal.enter()
    def start_soma(self):
        print("Starting scoring service on GPU...")
        print("  This downloads model weights on first run (~5GB per model) and may take several minutes.")
        self.proc = subprocess.Popen(
            ["soma", "start", "scoring", "--device", "cuda", "--data-dir", "/data"],
            stdout=sys.stdout,
            stderr=sys.stderr,
        )

        async def _wait():
            from soma_sdk import SomaClient

            for attempt in range(30):
                try:
                    client = await SomaClient(
                        chain="testnet",
                        scoring_url=f"http://localhost:{SCORING_PORT}",
                    )
                    if await client.scoring_health():
                        print("Scoring service is ready.")
                        return
                except Exception:
                    pass
                if attempt == 5:
                    print("  Still waiting for scoring service... (this is normal on first run while models download)")
                await asyncio.sleep(1)
            raise RuntimeError("Scoring service failed to start within 30s")

        asyncio.run(_wait())

        os.makedirs(LOCAL_DATA_DIR, exist_ok=True)
        handler = lambda *args: http.server.SimpleHTTPRequestHandler(
            *args, directory=LOCAL_DATA_DIR
        )
        httpd = http.server.HTTPServer(("", LOCAL_DATA_PORT), handler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()

        print("Loading data stream from The Stack v2 (shuffling ~10k samples)...")
        self.data_stream = prefetch_stream(stream_stack_v2())
        print("Data stream ready. Starting submission loop.\n")

    @modal.method()
    async def run(self):
        from soma_sdk import Keypair, SomaClient

        secret_key = os.environ.get("SOMA_SECRET_KEY")
        if not secret_key:
            print("No SOMA_SECRET_KEY set — exiting.")
            return

        kp = Keypair.from_secret_key(secret_key)
        print(f"Wallet: {kp.address()}")
        print("Fetching open targets and beginning scoring loop...")
        print("  Each sample is scored via the scoring service (~7s per sample).")
        print("  After scoring, the embedding is checked against all open targets.")
        print("  Hits are submitted on-chain in the background while scoring continues.\n")

        while True:
            try:
                await self._score_and_submit(kp)
            except Exception as e:
                print(f"Error during scoring iteration: {e}")

    async def _score_and_submit(self, kp):
        from soma_sdk import SomaClient

        client = await SomaClient(
            chain="testnet",
            scoring_url=f"http://localhost:{SCORING_PORT}",
        )

        sample_num = 0
        all_targets = []
        removed_ids = set()  # shared with background submit tasks
        needs_refresh = True  # shared flag — set by submit tasks after hits
        manifests_cache = {}
        pending_submits = []

        async def _submit_hits(hit_targets, data_bytes, checksum, winning_model_id, embedding, loss_score):
            nonlocal needs_refresh
            """Upload and submit in background while scoring continues."""
            data_url = upload_to_s3(data_bytes, checksum, hit_targets[0][0].generation_epoch)
            try:
                await client.merge_coins(signer=kp)
            except Exception:
                pass

            for t, dist in hit_targets:
                if t.id in removed_ids:
                    continue
                print(f"\n=== Submitting to {t.id[:16]}... (dist={dist:.6f}, thresh={t.distance_threshold}) ===")
                try:
                    await client.submit_data(
                        signer=kp,
                        target_id=t.id,
                        data=data_bytes,
                        data_url=data_url,
                        model_id=winning_model_id,
                        embedding=embedding,
                        distance_score=dist,
                        loss_score=loss_score,
                    )
                    reward_soma = (t.reward_pool / 2) / 1_000_000_000
                    print(f"")
                    print(f"  *** SUBMISSION SUCCESSFUL ***")
                    print(f"  Target:   {t.id}")
                    print(f"  Model:    {winning_model_id}")
                    print(f"  Distance: {dist:.6f} (threshold: {t.distance_threshold})")
                    print(f"  Reward:   {reward_soma:.4f} SOMA")
                    print(f"  Claimable in 2 epochs after settlement.")
                    print(f"")
                    removed_ids.add(t.id)
                    needs_refresh = True
                except Exception as e:
                    print(f"Submit failed for {t.id[:16]}...: {e}")
                    if "TargetNotOpen" in str(e):
                        removed_ids.add(t.id)
                        needs_refresh = True

        async def _refresh_targets():
            nonlocal all_targets, needs_refresh
            # Wait for pending submissions so we see their removals
            for task in pending_submits:
                await task
            pending_submits.clear()

            all_targets = list(await client.get_targets(status="open"))
            if not all_targets:
                print("No open targets — waiting...")
                await asyncio.sleep(10)
                return False
            random.shuffle(all_targets)
            removed_ids.clear()
            manifests_cache.clear()
            needs_refresh = False
            print(f"\n=== {len(all_targets)} open targets (distance threshold: {all_targets[0].distance_threshold}) ===")
            unique_models = {mid for t in all_targets for mid in t.model_ids}
            print(f"    {len(unique_models)} unique model(s) across targets")
            return True

        while True:
            # Refresh targets on first run, after hits, and periodically
            if needs_refresh or sample_num % REFRESH_EVERY == 0:
                if not await _refresh_targets():
                    return

            # Apply removals from background submissions
            if removed_ids:
                all_targets = [t for t in all_targets if t.id not in removed_ids]
                if not all_targets:
                    for task in pending_submits:
                        await task
                    return

            # Pick scoring target round-robin through shuffled list
            scoring_target = all_targets[sample_num % len(all_targets)]

            # Cache manifests per target
            if scoring_target.id not in manifests_cache:
                manifests_cache[scoring_target.id] = await client.get_model_manifests(scoring_target)
            manifests = manifests_cache[scoring_target.id]

            # Get next data sample
            try:
                data_bytes = next(self.data_stream)
            except StopIteration:
                print("Data stream exhausted — restarting")
                self.data_stream = prefetch_stream(stream_stack_v2())
                for task in pending_submits:
                    await task
                return

            checksum = client.commitment(data_bytes)
            local_url, local_path = save_local(data_bytes, checksum)
            sample_num += 1

            if sample_num == 1:
                print("Scoring first sample... (first score downloads model weights into the scoring service — may take a few minutes)")

            try:
                result = await client.score(
                    data_url=local_url,
                    models=manifests,
                    target_embedding=scoring_target.embedding,
                    data=data_bytes,
                )
            finally:
                cleanup_local(local_path)

            if sample_num == 1:
                print("First score complete. Scoring service is warmed up.\n")

            try:
                winner = result.winner
                distance = result.distance[0]
                embedding = result.embedding
                loss_score = result.loss_score
                winning_model_id = scoring_target.model_ids[winner]
            except (IndexError, TypeError) as e:
                print(f"Result parse error: {e} | winner={result.winner} distance={result.distance} model_ids={scoring_target.model_ids}")
                continue

            # Check verified embedding against ALL targets (free math)
            hit_targets = []
            for t in all_targets:
                if t.id in removed_ids:
                    continue
                if winning_model_id not in t.model_ids:
                    continue
                dist = cosine_distance(embedding, t.embedding)
                if dist <= t.distance_threshold:
                    hit_targets.append((t, dist))

            if not hit_targets:
                if sample_num % 10 == 0:
                    active = len(all_targets) - len(removed_ids)
                    print(f"[sample {sample_num}] no hit | dist={distance:.6f} thresh={scoring_target.distance_threshold} | {active} targets remaining")
                continue

            print(f"[sample {sample_num}] HIT — {len(hit_targets)} target(s) below threshold, submitting in background...")

            # Clean up finished tasks
            pending_submits = [t for t in pending_submits if not t.done()]

            # Fire off submissions in background, immediately continue scoring
            task = asyncio.create_task(
                _submit_hits(hit_targets, data_bytes, checksum, winning_model_id, embedding, loss_score)
            )
            pending_submits.append(task)


@app.function(schedule=modal.Period(hours=24))
async def scheduled_run():
    Submitter().run.remote()


@app.local_entrypoint()
def main():
    Submitter().run.remote()


def trigger():
    """Trigger the deployed submitter. No ephemeral app created."""
    cls = modal.Cls.from_name("soma-submitter", "Submitter")
    cls().run.spawn()
    print("Submitter triggered on deployed app. View at https://modal.com/apps")
