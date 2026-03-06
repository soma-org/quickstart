"""Soma training loop — train, commit, reveal, repeat.

Three independently-callable Modal functions:

  - commit:            Encrypt + upload latest checkpoint → commit on-chain (CPU)
  - reveal:            Poll epoch → reveal when ready → optionally spawn next round (CPU cron)
  - train_and_commit:  Train N steps on GPU → then commit (convenience combo)

The commit and reveal steps are independent of training — you can train with
train_torch.py or train_flax.py, then commit/reveal the result.

Usage:
    # Commit the latest checkpoint (works after manual training too):
    uv run modal run src/quickstart/training.py::commit_entrypoint

    # Manually trigger a reveal check:
    uv run modal run src/quickstart/training.py::reveal_entrypoint

    # Train + commit in one shot:
    uv run modal run src/quickstart/training.py::main

    # Deploy the automated training loop (reveal cron + chained training):
    uv run modal deploy src/quickstart/training.py
"""

import json
import os
import time

import modal

app = modal.App("soma-training")

volume = modal.Volume.from_name("soma-training-data", create_if_missing=True)

gpu_image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install(
        "soma-models[torch]>=0.1.7",
        "soma-sdk>=0.1.7",
        "datasets>=3.0",
        "torch>=2.0",
        "boto3",
        "smart_open",
    )
    .env({"PYTHONPATH": "/root/src"})
    .add_local_dir("src", remote_path="/root/src")
)

cpu_image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install(
        "soma-sdk>=0.1.7",
        "boto3",
    )
    .env({"PYTHONPATH": "/root/src"})
    .add_local_dir("src", remote_path="/root/src")
)

MODEL_DIR = "/training"
CHECKPOINT_PREFIX = "checkpoint"
LEARNING_RATE = 1e-4
DROPOUT_RATE = 0.1
MICRO_BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 64
LOG_EVERY = 10
SHUFFLE_BUFFER = 100_000
TRAINING_STATE_FILE = "training_state.json"


# ---------------------------------------------------------------------------
# Data pipeline (reuses pattern from train_torch.py)
# ---------------------------------------------------------------------------


def make_batches(batch_size: int):
    """Stream shuffled, tokenized batches from The Stack v2."""
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config
    from datasets import load_dataset
    from smart_open import open as smart_open

    from soma_models.v1.configs import V1_MAX_SEQ_LEN
    from soma_models.v1.tokenizer import tokenize

    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    ds = load_dataset(
        "bigcode/the-stack-v2-dedup",
        split="train",
        streaming=True,
        token=os.environ.get("HF_TOKEN"),
    )
    ds = ds.shuffle(buffer_size=SHUFFLE_BUFFER)

    def download_contents(row):
        s3_url = f"s3://softwareheritage/content/{row['blob_id']}"
        with smart_open(
            s3_url, "rb", compression=".gz", transport_params={"client": s3}
        ) as fin:
            content = fin.read().decode(row["src_encoding"])
        return {"content": content}

    ds = ds.map(download_contents)

    buffer_ids, buffer_targets = [], []

    for row in ds:
        sequences = tokenize(
            data=row["content"].encode("utf-8"), max_seq_len=V1_MAX_SEQ_LEN
        )
        for seq in sequences:
            buffer_ids.append(seq.token_ids)
            buffer_targets.append(seq.targets)

            if len(buffer_ids) == batch_size:
                yield buffer_ids, buffer_targets
                buffer_ids, buffer_targets = [], []


# ---------------------------------------------------------------------------
# Training — saves checkpoint + artifacts (embedding, weights bytes)
# ---------------------------------------------------------------------------


async def do_training(steps: int) -> int:
    """Train for N steps, save checkpoint + artifacts to volume. Returns final step."""
    import torch

    from soma_models.v1.torch import (
        Model,
        ModelConfig,
        SIGReg,
        SIGRegConfig,
        compute_loss,
    )
    from quickstart.common import (
        find_latest_checkpoint,
        save_training_artifacts,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt_path, ckpt_step = find_latest_checkpoint(MODEL_DIR, CHECKPOINT_PREFIX)
    if ckpt_path:
        print(f"Restoring from {ckpt_path} (step {ckpt_step})")
        model = Model.load(ckpt_path, ModelConfig(dropout_rate=DROPOUT_RATE))
        model = model.to(device)
        start_step = ckpt_step
    else:
        print("Initializing fresh model")
        model = Model(ModelConfig(dropout_rate=DROPOUT_RATE)).to(device)
        start_step = 0

    model.train()
    sig_reg = SIGReg(SIGRegConfig()).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Training steps {start_step} -> {start_step + steps}")
    batches = make_batches(MICRO_BATCH_SIZE)

    t0 = time.time()
    for i in range(steps):
        step = start_step + i
        optimizer.zero_grad()
        accum_loss = 0.0

        for _micro in range(GRAD_ACCUM_STEPS):
            ids, tgts = next(batches)
            token_ids = torch.tensor(ids, device=device)
            targets = torch.tensor(tgts, device=device)
            loss, embedding = compute_loss(model, sig_reg, token_ids, targets)
            (loss / GRAD_ACCUM_STEPS).backward()
            accum_loss += loss.item()

        optimizer.step()
        avg_loss = accum_loss / GRAD_ACCUM_STEPS

        if i % LOG_EVERY == 0:
            elapsed = time.time() - t0
            sps = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"  step {step:>6d} | loss {avg_loss:.4f} | {sps:.1f} steps/s")

    final_step = start_step + steps

    # Save checkpoint
    ckpt = f"{MODEL_DIR}/{CHECKPOINT_PREFIX}-{final_step}.safetensors"
    model.save(ckpt)

    # Compute embedding from last training batch — flatten to 1D
    model.eval()
    with torch.no_grad():
        _loss, embed = compute_loss(model, sig_reg, token_ids, targets)
    if embed.ndim == 2:
        embed = embed.mean(dim=0)
    embedding_list = embed.cpu().tolist()

    # Save artifacts (embedding + serialized weights) so commit can run on CPU
    weights_bytes = model.save_bytes()
    save_training_artifacts(MODEL_DIR, final_step, embedding_list, weights_bytes)

    await volume.commit.aio()
    print(f"Checkpoint + artifacts saved at step {final_step}")

    return final_step


# ---------------------------------------------------------------------------
# Commit — pure CPU, reads artifacts from volume
# ---------------------------------------------------------------------------


async def do_commit(mock: bool = False, epoch_duration_s: float = 30.0) -> dict:
    """Encrypt + upload weights from the latest training artifacts, commit on-chain.

    Reads embedding and serialized weights saved by training — no GPU or
    model framework needed.
    """
    from soma_sdk import Keypair, SomaClient
    from quickstart.common import (
        MockSomaClient,
        find_latest_checkpoint,
        load_training_state,
        load_training_artifacts,
        mock_upload_to_s3,
        save_training_state,
        upload_to_s3,
    )

    await volume.reload.aio()
    state = load_training_state(MODEL_DIR)

    if state["pending_reveal"]:
        print(f"⚠ Skipping commit — model {state['model_id']} has a pending reveal.")
        print(f"  Reveal it first (epoch must advance past {state['commit_epoch']}).")
        return state

    # Find latest checkpoint and its artifacts
    _ckpt_path, ckpt_step = find_latest_checkpoint(MODEL_DIR, CHECKPOINT_PREFIX)
    if ckpt_step == 0:
        raise RuntimeError("No checkpoint found on volume — train first!")

    artifacts = load_training_artifacts(MODEL_DIR, ckpt_step)
    if artifacts is None:
        raise RuntimeError(
            f"No training artifacts for step {ckpt_step}. "
            f"Re-run training with game.py (or save artifacts manually)."
        )

    embedding, weights_bytes = artifacts
    state["step"] = ckpt_step
    state["embedding"] = embedding
    print(f"Loaded artifacts for step {ckpt_step} ({len(weights_bytes)} bytes)")

    # Encrypt with a fresh key
    print(f"Encrypting {len(weights_bytes) / 1e9:.1f} GB of weights...")
    encrypted, decryption_key = SomaClient.encrypt_weights(weights_bytes)
    print(f"Encryption complete ({len(encrypted) / 1e9:.1f} GB encrypted)")

    # Upload
    if mock:
        client = MockSomaClient(MODEL_DIR, epoch_duration_s=epoch_duration_s)
        current_epoch = client.current_epoch
        weights_url = mock_upload_to_s3(
            encrypted, f"model-step-{ckpt_step}", current_epoch, MODEL_DIR
        )
    else:
        client = await SomaClient(chain="testnet")
        sys_state = await client.get_latest_system_state()
        current_epoch = sys_state.epoch
        weights_url = upload_to_s3(
            encrypted, f"model-step-{ckpt_step}", current_epoch
        )

    state["decryption_key"] = decryption_key
    state["weights_url"] = weights_url

    # Commit on-chain
    signer = Keypair.from_secret_key(os.environ["SOMA_SECRET_KEY"])

    if state["model_id"] is None:
        model_id = await client.commit_model(
            signer=signer,
            weights_url=weights_url,
            encrypted_weights=encrypted,
            decryption_key=decryption_key,
            embedding=embedding,
            commission_rate=1000,
        )
        state["model_id"] = model_id
        state["is_update"] = False
        print(f"Committed NEW model: {model_id}")
    else:
        await client.commit_model_update(
            signer=signer,
            model_id=state["model_id"],
            weights_url=weights_url,
            encrypted_weights=encrypted,
            decryption_key=decryption_key,
            embedding=embedding,
        )
        state["is_update"] = True
        print(f"Committed UPDATE to model: {state['model_id']}")

    state["pending_reveal"] = True
    state["commit_epoch"] = current_epoch
    save_training_state(state, MODEL_DIR)
    await volume.commit.aio()

    print(f"\n✓ Commit complete at epoch {current_epoch}")
    print(f"  Model: {state['model_id']}")
    print(f"  Step: {ckpt_step}")
    print(f"  Reveal valid at epoch >= {current_epoch + 1}")

    return state


# ---------------------------------------------------------------------------
# Reveal — uses reveal_model or reveal_model_update based on commit type
# ---------------------------------------------------------------------------


async def do_reveal(mock: bool = False, epoch_duration_s: float = 30.0) -> dict | None:
    """Check epoch and reveal if ready. Returns updated state if revealed, None otherwise."""
    from quickstart.common import (
        MockSomaClient,
        load_training_state,
        save_training_state,
    )

    await volume.reload.aio()
    state = load_training_state(MODEL_DIR)

    if not state["pending_reveal"]:
        print("No pending reveal — nothing to do.")
        return None

    print(f"Model: {state['model_id']}")
    print(f"Committed at epoch: {state['commit_epoch']}")
    print(f"Type: {'update' if state.get('is_update') else 'initial'}")

    if mock:
        client = MockSomaClient(MODEL_DIR, epoch_duration_s=epoch_duration_s)
        current_epoch = client.current_epoch
    else:
        from soma_sdk import SomaClient

        client = await SomaClient(chain="testnet")
        sys_state = await client.get_latest_system_state()
        current_epoch = sys_state.epoch

    print(f"Current epoch: {current_epoch}")

    if current_epoch <= state["commit_epoch"]:
        print(f"Epoch hasn't advanced yet (need > {state['commit_epoch']}). "
              f"Will retry later.")
        return None

    print(f"Epoch {current_epoch} > commit epoch {state['commit_epoch']} — revealing...")
    from soma_sdk import Keypair

    signer = Keypair.from_secret_key(os.environ["SOMA_SECRET_KEY"])

    try:
        if state.get("is_update"):
            await client.reveal_model_update(
                signer=signer,
                model_id=state["model_id"],
                decryption_key=state["decryption_key"],
                embedding=state["embedding"],
            )
        else:
            await client.reveal_model(
                signer=signer,
                model_id=state["model_id"],
                decryption_key=state["decryption_key"],
                embedding=state["embedding"],
            )
    except RuntimeError as e:
        print(f"Reveal failed: {e}")
        return None

    state["pending_reveal"] = False
    save_training_state(state, MODEL_DIR)
    await volume.commit.aio()

    print(f"✓ Model {state['model_id']} revealed at epoch {current_epoch}!")
    return state


# ---------------------------------------------------------------------------
# Local helper — write game state to local disk so CLI tools can read it
# ---------------------------------------------------------------------------


def write_local_state(state: dict):
    """Write training state to local training_state.json for stake/status CLI tools."""
    with open(TRAINING_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)
    print(f"Local {TRAINING_STATE_FILE} updated (model_id={state.get('model_id')})")


# ---------------------------------------------------------------------------
# Modal functions — thin wrappers
# ---------------------------------------------------------------------------


@app.function(
    image=cpu_image,
    timeout=600,
    volumes={MODEL_DIR: volume},
    secrets=[modal.Secret.from_name("soma-secrets")],
)
async def commit(mock: bool = False, epoch_duration_s: float = 30.0):
    """Commit the latest checkpoint on the volume (CPU only)."""
    return await do_commit(mock=mock, epoch_duration_s=epoch_duration_s)


@app.function(
    image=cpu_image,
    schedule=modal.Cron("0 */6 * * *"),  # every 6h — epochs are 24h
    timeout=600,
    volumes={MODEL_DIR: volume},
    secrets=[modal.Secret.from_name("soma-secrets")],
)
async def reveal(
    mock: bool = False,
    epoch_duration_s: float = 30.0,
    auto_continue: bool = True,
    steps_per_round: int = 500,
):
    """Reveal the model if the epoch has advanced. Optionally spawn next round."""
    result = await do_reveal(mock=mock, epoch_duration_s=epoch_duration_s)

    if result is not None and auto_continue:
        print(f"\nSpawning next training round ({steps_per_round} steps)...")
        train_and_commit.spawn(
            steps=steps_per_round,
            mock=mock,
            epoch_duration_s=epoch_duration_s,
        )


@app.function(
    image=gpu_image,
    gpu="H100",
    timeout=86400,
    volumes={MODEL_DIR: volume},
    secrets=[modal.Secret.from_name("soma-secrets")],
)
async def train_and_commit(
    steps: int = 500,
    mock: bool = False,
    epoch_duration_s: float = 30.0,
):
    """Train for N steps then commit. GPU is released after commit."""
    print(f"\n{'='*60}")
    print(f"TRAIN AND COMMIT (steps={steps}, mock={mock})")
    print(f"{'='*60}\n")

    await do_training(steps)

    return await do_commit(mock=mock, epoch_duration_s=epoch_duration_s)


# ---------------------------------------------------------------------------
# Entrypoints — default to mock=True for local dev, production uses defaults
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main(
    steps_per_round: int = 500,
    mock: bool = True,
    epoch_duration: float = 30.0,
):
    """Train + commit. Reveal happens via cron or manually."""
    state = train_and_commit.remote(
        steps=steps_per_round,
        mock=mock,
        epoch_duration_s=epoch_duration,
    )
    write_local_state(state)
    print("Committed. Reveal will happen via cron (or run game.py::reveal_entrypoint).")


@app.local_entrypoint()
def commit_entrypoint(
    mock: bool = True,
    epoch_duration: float = 30.0,
):
    """Commit the latest checkpoint (no training)."""
    state = commit.remote(mock=mock, epoch_duration_s=epoch_duration)
    write_local_state(state)


@app.local_entrypoint()
def reveal_entrypoint(
    mock: bool = True,
    epoch_duration: float = 30.0,
):
    """Manually trigger a reveal check."""
    state = reveal.remote(mock=mock, epoch_duration_s=epoch_duration, auto_continue=False)
    if state is not None:
        write_local_state(state)
