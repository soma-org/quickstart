"""Soma data submitter — finds and submits data against open targets on Modal.

Streams shuffled source files from The Stack v2, scores them against open
targets using a GPU, and uploads to S3 + submits on-chain when the distance
threshold is met.

Deploys as a cron job that runs every 24 hours, with a 23h45m timeout.
The submission loop runs continuously within each invocation.

Deploy with:
    uv run modal deploy src/quickstart/submitter.py

Required Modal secrets (soma-secrets):
    SOMA_SECRET_KEY, HF_TOKEN,
    S3_BUCKET, S3_ACCESS_KEY_ID, S3_SECRET_ACCESS_KEY, S3_ENDPOINT_URL, S3_PUBLIC_URL
"""

import asyncio
import http.server
import os
import random
import subprocess
import sys
import threading

import modal

app = modal.App("soma-submitter-best-model")

volume = modal.Volume.from_name("soma-scoring-data", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.13")
    .apt_install("curl")
    .env({"PATH": "/root/.local/bin:/usr/local/bin:/usr/bin:/bin", "RUST_LOG": "warn"})
    .run_commands(
        "curl -sSfL https://sup.soma.org | sh",
        "sup install soma",
    )
    .pip_install("datasets>=3.0", "boto3>=1.35", "smart_open", "soma-models[torch]>=0.1.7", "torch>=2.0")
    .run_commands("pip install --force-reinstall --no-cache-dir soma-sdk")
)

SCORING_PORT = 9124
LOCAL_DATA_PORT = 9125
LOCAL_DATA_DIR = "/tmp/soma-local-data"

# Dataset has ~600M rows. Skip to a random offset each round.
DATASET_SIZE_ESTIMATE = 600_000_000


def stream_stack_v2():
    """Stream data from a random region of The Stack v2."""
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
    )

    # Skip to a random offset for diversity
    skip = random.randint(0, DATASET_SIZE_ESTIMATE)
    print(f"Skipping to offset {skip}...")
    ds = ds.skip(skip)

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
        if len(data) > 10_000:
            continue
        yield data


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


def load_model(manifest, device):
    """Download, decrypt, and load a single model for local inference."""
    import requests
    from soma_sdk import SomaClient
    from soma_models.v1.torch import Model, ModelConfig

    print(f"Downloading model from {manifest.url[:60]}...")
    resp = requests.get(manifest.url, timeout=300)
    resp.raise_for_status()
    decrypted = SomaClient.decrypt_weights(resp.content, manifest.decryption_key)
    model = Model.load_bytes(decrypted, ModelConfig(dropout_rate=0.0))
    model.eval()
    model.to(device)
    print(f"  Model loaded ({len(resp.content)} bytes)")
    return model


def batch_embed(model, data_list, device):
    """Run batched inference on multiple data samples at once. Returns list of embeddings."""
    import torch
    from soma_models.v1.tokenizer import tokenize
    from soma_models.v1.configs import V1_MAX_SEQ_LEN

    embeddings = []
    # Tokenize all samples — each produces 1+ sequences, we need the mean
    batch_token_ids = []
    batch_positions = []
    sample_indices = []  # maps batch row -> sample index
    seq_counts = []      # how many seqs per sample

    for i, data_bytes in enumerate(data_list):
        seqs = tokenize(data_bytes)
        if not seqs:
            seq_counts.append(0)
            continue
        seq_counts.append(len(seqs))
        for seq in seqs:
            batch_token_ids.append(seq.token_ids)
            batch_positions.append(list(range(V1_MAX_SEQ_LEN)))
            sample_indices.append(i)

    if not batch_token_ids:
        return [None] * len(data_list)

    # Forward pass — all sequences in one batch
    with torch.no_grad():
        token_tensor = torch.tensor(batch_token_ids, device=device)
        pos_tensor = torch.tensor(batch_positions, device=device)
        representations = model.encode(token_tensor, pos_tensor)
        # Mean over sequence dim -> [batch, embed_dim]
        seq_embeds = representations.mean(dim=1)

    # Aggregate sequences back to per-sample embeddings
    result = [None] * len(data_list)
    offset = 0
    for i, count in enumerate(seq_counts):
        if count == 0:
            continue
        sample_embed = seq_embeds[offset:offset + count].mean(dim=0)
        result[i] = sample_embed
        offset += count

    return result


def pick_best_model(targets, active_models):
    """Pick the model with the smallest embedding distance to any target."""
    model_embeds = {m.model_id: m.embedding for m in active_models}
    best_mid = None
    best_dist = float("inf")
    for t in targets:
        for mid in t.model_ids:
            me = model_embeds.get(mid)
            if not me:
                continue
            te = t.embedding
            dot = sum(a * b for a, b in zip(me, te))
            norm_m = sum(a * a for a in me) ** 0.5
            norm_t = sum(a * a for a in te) ** 0.5
            dist = 1.0 - dot / (norm_m * norm_t)
            if dist < best_dist:
                best_dist = dist
                best_mid = mid
    print(f"Best model: {best_mid} (closest dist={best_dist:.6f})")
    return best_mid


def find_favorable_targets(targets, model_id, active_models):
    """Return targets where this model participates, sorted by model embedding distance."""
    model_embeds = {m.model_id: m.embedding for m in active_models}
    me = model_embeds.get(model_id)
    if not me:
        return []
    results = []
    for t in targets:
        if model_id not in t.model_ids:
            continue
        te = t.embedding
        dot = sum(a * b for a, b in zip(me, te))
        norm_m = sum(a * a for a in me) ** 0.5
        norm_t = sum(a * a for a in te) ** 0.5
        model_dist = 1.0 - dot / (norm_m * norm_t)
        results.append((model_dist, t))
    results.sort(key=lambda x: x[0])
    return results


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
        self.proc = subprocess.Popen(
            ["soma", "start", "scoring", "--device", "cuda", "--data-dir", "/data"],
            stdout=sys.stdout,
            stderr=sys.stderr,
        )

        async def _wait():
            from soma_sdk import SomaClient

            for _ in range(30):
                try:
                    client = await SomaClient(
                        chain="testnet",
                        scoring_url=f"http://localhost:{SCORING_PORT}",
                    )
                    if await client.scoring_health():
                        return
                except Exception:
                    pass
                await asyncio.sleep(1)
            raise RuntimeError("Scoring service failed to start within 30s")

        asyncio.run(_wait())

        os.makedirs(LOCAL_DATA_DIR, exist_ok=True)
        handler = lambda *args: http.server.SimpleHTTPRequestHandler(
            *args, directory=LOCAL_DATA_DIR
        )
        httpd = http.server.HTTPServer(("", LOCAL_DATA_PORT), handler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()

        self.data_stream = stream_stack_v2()

    @modal.method()
    async def run(self):
        import torch
        from soma_sdk import Keypair, SomaClient

        secret_key = os.environ.get("SOMA_SECRET_KEY")
        if not secret_key:
            print("No SOMA_SECRET_KEY set — exiting.")
            return

        kp = Keypair.from_secret_key(secret_key)
        print(f"Sender: {kp.address()}")

        client = await SomaClient(
            chain="testnet",
            scoring_url=f"http://localhost:{SCORING_PORT}",
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Local inference device: {device}")

        # Pick the best model and load it once
        targets = await client.get_targets(status="open")
        active_models = await client.get_active_models()
        best_model_id = pick_best_model(targets, active_models)

        manifest = (await client.get_model_manifests([best_model_id]))[0]
        local_model = load_model(manifest, device)

        while True:
            try:
                await self._score_and_submit(kp, client, device, local_model, best_model_id)
            except Exception as e:
                print(f"Error during scoring iteration: {e}")

    async def _score_and_submit(self, kp, client, device, local_model, model_id):
        import torch

        # Refresh targets and find favorable ones for our model
        targets = await client.get_targets(status="open")
        active_models = await client.get_active_models()
        favorable = find_favorable_targets(targets, model_id, active_models)

        if not favorable:
            print("No targets for our model — waiting...")
            await asyncio.sleep(10)
            return

        print(f"\n=== {len(favorable)} targets for model {model_id[:16]}... ===")
        for model_dist, t in favorable:
            marker = " *" if model_dist < t.distance_threshold else ""
            print(f"  {t.id[:20]}...  model_dist={model_dist:.6f}  thresh={t.distance_threshold}{marker}")

        # Stack all target embeddings into a matrix for batch cosine distance
        target_list = [t for _, t in favorable]
        target_matrix = torch.stack([
            torch.tensor(t.embedding, device=device) for t in target_list
        ])
        target_norms = target_matrix.norm(dim=1, keepdim=True)
        target_matrix_normed = target_matrix / target_norms

        thresholds = torch.tensor(
            [t.distance_threshold for t in target_list], device=device
        )

        # Batched scoring loop
        BATCH_SIZE = 16
        COLD_RESHUFFLE = 1   # skip after 1 cold batch (16 cold samples)
        HOT_ZONE_MARGIN = 1.003
        HOT_EXTEND = 50      # 50 more batches in hot zone (800 samples)

        batch_num = 0
        batches_until_reshuffle = COLD_RESHUFFLE

        while True:
            # Collect a batch of data
            batch_data = []
            for _ in range(BATCH_SIZE):
                try:
                    batch_data.append(next(self.data_stream))
                except StopIteration:
                    break
            if not batch_data:
                print("Data stream exhausted — restarting")
                self.data_stream = stream_stack_v2()
                return

            batch_num += 1

            # Batch embed all samples at once
            embeds = batch_embed(local_model, batch_data, device)

            # Check each embedding against all targets
            found_hit = False
            batch_best_dist = float("inf")

            for i, embed in enumerate(embeds):
                if embed is None:
                    continue

                embed_normed = embed / embed.norm()
                cos_sims = target_matrix_normed @ embed_normed
                dists = 1.0 - cos_sims

                best_dist = dists.min().item()
                if best_dist < batch_best_dist:
                    batch_best_dist = best_dist

                hits = (dists <= thresholds).nonzero(as_tuple=True)[0]
                if len(hits) == 0:
                    continue

                # Hit found — verify and submit
                best_hit_idx = hits[dists[hits].argmin()].item()
                local_dist = dists[best_hit_idx].item()
                target = target_list[best_hit_idx]
                data_bytes = batch_data[i]
                print(f"  ** Hit! batch={batch_num} sample={i} target={target.id[:16]}... local_dist={local_dist:.6f}")

                # Verify through scoring service (all 3 models)
                manifests = await client.get_model_manifests(target)
                checksum = client.commitment(data_bytes)
                local_url, local_path = save_local(data_bytes, checksum)

                try:
                    result = await client.score(
                        data_url=local_url,
                        models=manifests,
                        target_embedding=target.embedding,
                        data=data_bytes,
                    )
                finally:
                    cleanup_local(local_path)

                winner = result.winner
                distance = result.distance[0]
                print(f"Verified: local={local_dist:.6f} -> actual={distance:.6f}  thresh={target.distance_threshold}")

                if distance > target.distance_threshold:
                    print("Verification failed — continuing search...")
                    continue

                # Verified hit — merge coins then submit
                try:
                    await client.merge_coins(signer=kp)
                except Exception:
                    pass

                embedding = result.embedding
                loss_score = result.loss_score
                data_url = upload_to_s3(data_bytes, checksum, target.generation_epoch)
                print(f"\n=== Submitting data (url: {data_url}) ===")
                await client.submit_data(
                    signer=kp,
                    target_id=target.id,
                    data=data_bytes,
                    data_url=data_url,
                    model_id=target.model_ids[winner],
                    embedding=embedding,
                    distance_score=distance,
                    loss_score=loss_score,
                )
                print("Submission successful!")
                print(f"Target: {target.id}")
                print(f"Model:  {target.model_ids[winner]}")
                print(f"Distance: {distance:.6f} (threshold: {target.distance_threshold})")
                return  # Refresh targets in outer loop

            # Hot/cold zone logic
            best_thresh = thresholds.min().item()
            if batch_best_dist <= best_thresh * HOT_ZONE_MARGIN:
                batches_until_reshuffle = HOT_EXTEND
                print(f"[batch {batch_num}] HOT best_dist={batch_best_dist:.6f}  thresh={best_thresh}")
            else:
                batches_until_reshuffle -= 1
                print(f"[batch {batch_num}] COLD best_dist={batch_best_dist:.6f}  ({batches_until_reshuffle} batches left)")

            if batches_until_reshuffle <= 0:
                print(f"No hits after {batch_num} batches — skipping to new dataset region")
                self.data_stream = stream_stack_v2()
                return


@app.local_entrypoint()
def main():
    Submitter().run.remote()
