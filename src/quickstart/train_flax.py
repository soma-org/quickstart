"""Soma V1 training (Flax/JAX) on FineWeb — runs on Modal with a GPU.

Run with:
    uv run modal run --detach src/quickstart/train_flax.py
"""

import time

import modal

app = modal.App("soma-training-flax")

volume = modal.Volume.from_name("soma-training-data", create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.13").pip_install(
    "soma-models[flax]>=0.1.7",
    "datasets>=3.0",
    "optax>=0.2",
    "jax[cuda12]>=0.5",
)

MODEL_DIR = "/training"
CHECKPOINT_EVERY = 500
LOG_EVERY = 10
LEARNING_RATE = 1e-4
DROPOUT_RATE = 0.1
MICRO_BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 8  # effective batch size = 2 * 8 = 16


def make_batches(batch_size: int):
    """Stream tokenized batches from FineWeb."""
    from datasets import load_dataset

    from soma_models.v1.configs import V1_MAX_SEQ_LEN
    from soma_models.v1.tokenizer import tokenize

    ds = load_dataset(
        "HuggingFaceFW/fineweb", "sample-10BT", split="train", streaming=True
    )

    buffer_ids, buffer_targets = [], []

    for example in ds:
        text = example.get("text", "")
        if not text.strip():
            continue

        sequences = tokenize(data=text.encode("utf-8"), max_seq_len=V1_MAX_SEQ_LEN)

        for seq in sequences:
            buffer_ids.append(seq.token_ids)
            buffer_targets.append(seq.targets)

            if len(buffer_ids) == batch_size:
                yield buffer_ids, buffer_targets
                buffer_ids, buffer_targets = [], []


@app.function(image=image, gpu="A100", timeout=86400, volumes={MODEL_DIR: volume})
def train(num_steps: int = 10_000):
    import jax.numpy as jnp
    import optax
    from flax import nnx

    from soma_models.v1.flax import (
        Model,
        ModelConfig,
        SIGReg,
        SIGRegConfig,
        compute_loss,
    )

    import jax

    print(f"JAX backend: {jax.default_backend()}")

    print("Initializing model...")
    rngs = nnx.Rngs(0)
    model = Model(ModelConfig(dropout_rate=DROPOUT_RATE), rngs)
    model.train()
    sig_reg = SIGReg(SIGRegConfig(), rngs)

    optimizer = nnx.Optimizer(
        model, optax.adam(learning_rate=LEARNING_RATE), wrt=nnx.Param
    )
    print("Model initialized.")

    @nnx.jit
    def micro_step(model, sig_reg, token_ids, targets):
        def loss_fn(model, sig_reg):
            return compute_loss(model, sig_reg, token_ids, targets)

        (loss, embedding), grads = nnx.value_and_grad(loss_fn, has_aux=True)(
            model, sig_reg
        )
        return loss, embedding, grads

    print("Loading dataset (streaming from HuggingFace)...")
    batches = make_batches(MICRO_BATCH_SIZE)
    first_ids, first_tgts = next(batches)
    print("First batch ready — starting training.\n")

    t0 = time.time()
    for step in range(num_steps):
        accum_loss = jnp.zeros(())
        accum_grads = None

        for micro in range(GRAD_ACCUM_STEPS):
            if step == 0 and micro == 0:
                ids, tgts = first_ids, first_tgts
            else:
                ids, tgts = next(batches)
            token_ids = jnp.array(ids)
            targets = jnp.array(tgts)

            loss, embedding, grads = micro_step(model, sig_reg, token_ids, targets)

            if step == 0 and micro == 0:
                print("JAX compiled the training step (first step is slow).")

            accum_loss = accum_loss + loss

            if accum_grads is None:
                accum_grads = grads
            else:
                accum_grads = jax.tree.map(jnp.add, accum_grads, grads)

        accum_grads = jax.tree.map(lambda g: g / GRAD_ACCUM_STEPS, accum_grads)
        optimizer.update(model, accum_grads)
        avg_loss = accum_loss / GRAD_ACCUM_STEPS

        if step % LOG_EVERY == 0:
            elapsed = time.time() - t0
            steps_per_sec = (step + 1) / elapsed if elapsed > 0 else 0
            print(
                f"step {step:>6d} | loss {float(avg_loss):.4f} | {steps_per_sec:.1f} steps/s"
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
