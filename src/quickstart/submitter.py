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
    .pip_install("soma-sdk>=0.1.7", "datasets>=3.0", "boto3>=1.35", "smart_open")
)

SCORING_PORT = 9124
LOCAL_DATA_PORT = 9125
LOCAL_DATA_DIR = "/tmp/soma-local-data"


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
    ).shuffle(buffer_size=1_000_000)

    for row in ds:
        s3_url = f"s3://softwareheritage/content/{row['blob_id']}"
        with smart_open(
            s3_url, "rb", compression=".gz", transport_params={"client": s3}
        ) as fin:
            content = fin.read().decode(row["src_encoding"])
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
        from soma_sdk import Keypair, SomaClient

        secret_key = os.environ.get("SOMA_SECRET_KEY")
        if not secret_key:
            print("No SOMA_SECRET_KEY set — exiting.")
            return

        kp = Keypair.from_secret_key(secret_key)
        print(f"Sender: {kp.address()}")

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

        # Find an open target
        targets = await client.get_targets(status="open")
        target = next(t for t in targets)
        print(f"\nTarget: {target.id}  threshold={target.distance_threshold}")

        manifests = await client.get_model_manifests(target)
        print(f"Fetched {len(manifests)} model manifest(s)")

        # Get next shuffled Stack v2 source file and save locally
        data_bytes = next(self.data_stream)
        checksum = client.commitment(data_bytes)
        local_url, local_path = save_local(data_bytes, checksum)
        print(f"Data size: {len(data_bytes)}, checksum: {checksum}")

        try:
            # Score using local URL
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
        embedding = result.embedding
        loss_score = result.loss_score

        print(f"Winner: index={winner}  distance={distance:.6f}")
        print(f"Threshold: {target.distance_threshold}")
        print(f"Loss score: {loss_score}")

        if distance > target.distance_threshold:
            print("Distance exceeds threshold — no hit, skipping submission.")
            return

        # Score passed — upload to S3 then submit on-chain
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


@app.local_entrypoint()
def main():
    Submitter().run.remote()
