"""Shared utilities for the Soma training loop.

Pure-Python helpers with no Modal dependencies — usable from any Modal app
(training.py, train_torch.py, submitter.py, etc.) or from local scripts.
"""

import glob as globmod
import json
import os
import re


# ---------------------------------------------------------------------------
# Training state — JSON sidecar on disk (caller handles volume sync)
# ---------------------------------------------------------------------------

DEFAULT_STATE = {
    "model_id": None,
    "step": 0,
    "pending_reveal": False,
    "commit_epoch": None,
    "decryption_key": None,
    "weights_url": None,
    "embedding": None,
    "framework": "torch",
}


def load_training_state(state_dir: str) -> dict:
    path = os.path.join(state_dir, "training_state.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return dict(DEFAULT_STATE)


def save_training_state(state: dict, state_dir: str):
    os.makedirs(state_dir, exist_ok=True)
    path = os.path.join(state_dir, "training_state.json")
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def find_latest_checkpoint(
    model_dir: str, prefix: str = "checkpoint"
) -> tuple[str | None, int]:
    """Find the checkpoint with the highest step number, return (path, step)."""
    pattern = os.path.join(model_dir, f"{prefix}-*.safetensors")
    best_path, best_step = None, -1
    for path in globmod.glob(pattern):
        match = re.search(rf"{re.escape(prefix)}-(\d+)\.safetensors$", path)
        if not match:
            continue
        step = int(match.group(1))
        if step > best_step:
            best_step, best_path = step, path
    if best_path is None:
        return None, 0
    return best_path, best_step


# ---------------------------------------------------------------------------
# Training artifact helpers — saved alongside checkpoints so commit
# doesn't need a GPU or model framework
# ---------------------------------------------------------------------------


def save_training_artifacts(
    model_dir: str, step: int, embedding: list[float], weights_bytes: bytes
):
    """Save embedding + raw weights alongside a checkpoint."""
    artifacts_path = os.path.join(model_dir, f"artifacts-{step}.json")
    with open(artifacts_path, "w") as f:
        json.dump({"step": step, "embedding": embedding}, f)

    weights_path = os.path.join(model_dir, f"weights-{step}.bin")
    with open(weights_path, "wb") as f:
        f.write(weights_bytes)


def load_training_artifacts(
    model_dir: str, step: int
) -> tuple[list[float], bytes] | None:
    """Load embedding + raw weights for a given step. Returns None if missing."""
    artifacts_path = os.path.join(model_dir, f"artifacts-{step}.json")
    weights_path = os.path.join(model_dir, f"weights-{step}.bin")

    if not os.path.exists(artifacts_path) or not os.path.exists(weights_path):
        return None

    with open(artifacts_path) as f:
        artifacts = json.load(f)

    with open(weights_path, "rb") as f:
        weights_bytes = f.read()

    return artifacts["embedding"], weights_bytes


# ---------------------------------------------------------------------------
# S3 upload
# ---------------------------------------------------------------------------


def upload_to_s3(data: bytes, key_name: str, epoch: int) -> str:
    """Upload data to S3-compatible bucket, return public URL."""
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
    key = f"models/{epoch}/{key_name}.safetensors.enc"
    s3.put_object(Bucket=bucket, Key=key, Body=data, ACL="public-read")

    if public_url:
        return f"{public_url.rstrip('/')}/{key}"
    region = s3.get_bucket_location(Bucket=bucket)["LocationConstraint"] or "us-east-1"
    return f"https://{bucket}.s3.{region}.amazonaws.com/{key}"


