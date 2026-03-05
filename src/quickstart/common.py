"""Shared utilities for the Soma training loop.

Pure-Python helpers with no Modal dependencies — usable from any Modal app
(training.py, train_torch.py, submitter.py, etc.) or from local scripts.
"""

import glob as globmod
import json
import os
import re
import time


# ---------------------------------------------------------------------------
# Training state — JSON sidecar on disk (caller handles volume sync)
# ---------------------------------------------------------------------------

DEFAULT_STATE = {
    "model_id": None,
    "step": 0,
    "pending_reveal": False,
    "is_update": False,  # True after first model — controls reveal_model vs reveal_model_update
    "commit_epoch": None,
    "decryption_key": None,
    "weights_url": None,
    "embedding": None,
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


def mock_upload_to_s3(
    data: bytes, key_name: str, epoch: int, upload_dir: str
) -> str:
    """Save encrypted weights to disk, return a fake public URL."""
    dest = os.path.join(upload_dir, "mock_s3", "models", str(epoch))
    os.makedirs(dest, exist_ok=True)
    filename = f"{key_name}.safetensors.enc"
    with open(os.path.join(dest, filename), "wb") as f:
        f.write(data)
    url = f"https://mock-s3.example.com/models/{epoch}/{filename}"
    print(f"  [MOCK S3] Saved {len(data)} bytes -> {url}")
    return url


# ---------------------------------------------------------------------------
# Mock chain client — simulates epoch timing for testing
# ---------------------------------------------------------------------------


class MockSomaClient:
    """Simulates on-chain epoch timing without touching a real network.

    Epochs advance based on wall-clock time from a shared start time
    persisted to disk, so all callers see consistent epochs.
    """

    def __init__(self, state_dir: str, epoch_duration_s: float = 30.0):
        self.epoch_duration_s = epoch_duration_s
        self.state_file = os.path.join(state_dir, "mock_chain.json")
        self._load_or_init()

    def _load_or_init(self):
        if os.path.exists(self.state_file):
            with open(self.state_file) as f:
                self.chain = json.load(f)
        else:
            self.chain = {
                "start_time": time.time(),
                "epoch_duration_s": self.epoch_duration_s,
                "models": {},
                "next_model_seq": 1,
            }
            self._save()

    def _save(self):
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump(self.chain, f, indent=2)

    @property
    def current_epoch(self) -> int:
        elapsed = time.time() - self.chain["start_time"]
        return int(elapsed / self.chain["epoch_duration_s"])

    async def get_latest_system_state(self):
        from types import SimpleNamespace

        return SimpleNamespace(epoch=self.current_epoch)

    async def get_embedding_dim(self) -> int:
        return 2048

    async def commit_model(
        self, *, signer, weights_url, encrypted_weights, decryption_key,
        embedding, commission_rate, stake_amount=None,
    ) -> str:
        epoch = self.current_epoch
        model_id = f"mock-model-{self.chain['next_model_seq']:04d}"
        self.chain["models"][model_id] = {
            "commit_epoch": epoch,
            "revealed": False,
            "weights_url": weights_url,
        }
        self.chain["next_model_seq"] += 1
        self._save()
        print(f"  [MOCK] commit_model -> {model_id} at epoch {epoch}")
        print(f"  [MOCK] reveal will be valid at epoch >= {epoch + 1}")
        return model_id

    async def commit_model_update(
        self, *, signer, model_id, weights_url, encrypted_weights,
        decryption_key, embedding,
    ):
        epoch = self.current_epoch
        if model_id not in self.chain["models"]:
            raise RuntimeError(f"[MOCK] Unknown model_id: {model_id}")
        self.chain["models"][model_id].update({
            "commit_epoch": epoch,
            "revealed": False,
            "weights_url": weights_url,
        })
        self._save()
        print(f"  [MOCK] commit_model_update {model_id} at epoch {epoch}")
        print(f"  [MOCK] reveal will be valid at epoch >= {epoch + 1}")

    async def reveal_model(self, *, signer, model_id, decryption_key, embedding):
        epoch = self.current_epoch
        if model_id not in self.chain["models"]:
            raise RuntimeError(f"[MOCK] Unknown model_id: {model_id}")
        model = self.chain["models"][model_id]
        commit_epoch = model["commit_epoch"]
        if epoch <= commit_epoch:
            raise RuntimeError(
                f"[MOCK] Cannot reveal {model_id} in epoch {epoch} — "
                f"committed in epoch {commit_epoch}, "
                f"must wait until epoch >= {commit_epoch + 1}"
            )
        model["revealed"] = True
        self._save()
        print(f"  [MOCK] reveal_model {model_id} at epoch {epoch} ✓")
        print(f"  [MOCK] model is now active for scoring!")

    async def reveal_model_update(self, *, signer, model_id, decryption_key, embedding):
        await self.reveal_model(
            signer=signer, model_id=model_id,
            decryption_key=decryption_key, embedding=embedding,
        )
