"""Soma V1 training (PyTorch) on The Stack v2 — runs on Modal with a GPU.

Run with:
    uv run modal run --detach src/quickstart/train_torch.py
"""

import time

import modal

app = modal.App("soma-training-torch")

volume = modal.Volume.from_name("soma-training-data", create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.13").pip_install(
    "soma-models[torch]>=0.1.7",
    "datasets>=3.0",
    "torch>=2.0",
    "boto3",
    "smart_open",
)

MODEL_DIR = "/training"
CHECKPOINT_EVERY = 500
LOG_EVERY = 10
LEARNING_RATE = 1e-4
DROPOUT_RATE = 0.1
MICRO_BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 8  # effective batch size = 2 * 8 = 16
SHUFFLE_BUFFER = 100_000


def make_batches(batch_size: int):
    """Stream shuffled, tokenized batches from The Stack v2."""
    import os

    import boto3
    from datasets import load_dataset
    from smart_open import open as smart_open

    from soma_models.v1.configs import V1_MAX_SEQ_LEN
    from soma_models.v1.tokenizer import tokenize

    session = boto3.Session(
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )
    s3 = session.client("s3")

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


@app.function(
    image=image,
    gpu="H100",
    timeout=86400,
    volumes={MODEL_DIR: volume},
    secrets=[modal.Secret.from_name("soma-secrets")],
)
def train(num_steps: int = 10_000):
    import torch

    from soma_models.v1.torch import (
        Model,
        ModelConfig,
        SIGReg,
        SIGRegConfig,
        compute_loss,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Initializing model...")
    model = Model(ModelConfig(dropout_rate=DROPOUT_RATE)).to(device)
    model.train()
    sig_reg = SIGReg(SIGRegConfig()).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print("Model initialized.")

    print("Loading dataset (streaming from The Stack v2)...")
    batches = make_batches(MICRO_BATCH_SIZE)
    first_ids, first_tgts = next(batches)
    print("First batch ready — starting training.\n")

    t0 = time.time()
    for step in range(num_steps):
        optimizer.zero_grad()
        accum_loss = 0.0

        for micro in range(GRAD_ACCUM_STEPS):
            if step == 0 and micro == 0:
                ids, tgts = first_ids, first_tgts
            else:
                ids, tgts = next(batches)
            token_ids = torch.tensor(ids, device=device)
            targets = torch.tensor(tgts, device=device)

            loss, embedding = compute_loss(model, sig_reg, token_ids, targets)
            (loss / GRAD_ACCUM_STEPS).backward()
            accum_loss += loss.item()

        optimizer.step()
        avg_loss = accum_loss / GRAD_ACCUM_STEPS

        if step % LOG_EVERY == 0:
            elapsed = time.time() - t0
            steps_per_sec = (step + 1) / elapsed if elapsed > 0 else 0
            print(
                f"step {step:>6d} | loss {avg_loss:.4f} | {steps_per_sec:.1f} steps/s"
            )

        if step > 0 and step % CHECKPOINT_EVERY == 0:
            path = f"{MODEL_DIR}/checkpoint-{step}.safetensors"
            model.save(path)
            volume.commit()
            print(f"  → saved {path}")

    final_path = f"{MODEL_DIR}/model-final.safetensors"
    model.save(final_path)
    volume.commit()
    print(f"Training complete — saved {final_path}")


@app.local_entrypoint()
def main(num_steps: int = 10_000):
    train.remote(num_steps=num_steps)
