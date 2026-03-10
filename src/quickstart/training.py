"""Soma training loop — train, commit, reveal, repeat.

Two modes:

  Localnet (dev/test):
    Full cycle (train → commit → advance_epoch → reveal) in one invocation.

      uv run modal run src/quickstart/training.py::localnet
      uv run modal run src/quickstart/training.py::localnet --framework flax

  Testnet (production):
    Train + commit, then deploy cron for automated reveal + re-training.

      uv run modal run src/quickstart/training.py --steps-per-round 500
      uv run modal deploy src/quickstart/training.py

Supports both PyTorch and Flax/JAX via --framework torch|flax (default: torch).
See train_torch.py and train_flax.py for standalone training-only reference scripts.
"""

import os
import time

import modal

app = modal.App("soma-training")

volume = modal.Volume.from_name("soma-training-data", create_if_missing=True)

gpu_image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install(
        "soma-models[torch,flax]>=0.1.7",
        "datasets>=3.0",
        "torch>=2.0",
        "optax>=0.2",
        "jax[cuda12]>=0.5",
        "boto3",
        "smart_open",
    )
    .run_commands("pip install --force-reinstall --no-cache-dir soma-sdk")
    .env({"PYTHONPATH": "/root/src"})
    .add_local_dir("src", remote_path="/root/src")
)

cpu_image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install(
        "boto3",
    )
    .run_commands("pip install --force-reinstall --no-cache-dir soma-sdk")
    .env({"PYTHONPATH": "/root/src"})
    .add_local_dir("src", remote_path="/root/src")
)

localnet_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.13"
    )
    .apt_install("curl")
    .run_commands("rm -rf /opt/nvidia/entrypoint.d")
    .env({
        "PATH": "/root/.local/bin:/usr/local/bin:/usr/bin:/bin",
        "RUST_LOG": "error",
        "PYTHONPATH": "/root/src",
    })
    .run_commands(
        "curl -sSfL https://sup.soma.org | sh",
        "sup install soma",
    )
    .pip_install(
        "soma-models[torch,flax]>=0.1.7",
        "datasets>=3.0",
        "torch>=2.0",
        "optax>=0.2",
        "jax[cuda12]>=0.5",
        "boto3",
        "smart_open",
    )
    .run_commands("pip install --force-reinstall --no-cache-dir soma-sdk")
    .add_local_dir("src", remote_path="/root/src")
)

MODEL_DIR = "/training"
LOCALNET_MODEL_DIR = "/localnet-training"
CHECKPOINT_PREFIX = "checkpoint"
LEARNING_RATE = 1e-4
DROPOUT_RATE = 0.1
MICRO_BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 64
LOG_EVERY = 10
SHUFFLE_BUFFER = 100_000


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


async def do_training(
    steps: int,
    framework: str = "torch",
    model_dir: str = MODEL_DIR,
    vol: modal.Volume | None = None,
    grad_accum_steps: int = GRAD_ACCUM_STEPS,
    log_every: int = LOG_EVERY,
) -> int:
    """Train for N steps, save checkpoint + artifacts to volume. Returns final step."""
    if framework == "flax":
        return await _do_training_flax(steps, model_dir, vol, grad_accum_steps, log_every)
    return await _do_training_torch(steps, model_dir, vol, grad_accum_steps, log_every)


async def _do_training_torch(
    steps: int,
    model_dir: str = MODEL_DIR,
    vol: modal.Volume | None = None,
    grad_accum_steps: int = GRAD_ACCUM_STEPS,
    log_every: int = LOG_EVERY,
) -> int:
    """PyTorch training path."""
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
    print(f"[torch] Device: {device}")

    os.makedirs(model_dir, exist_ok=True)

    ckpt_path, ckpt_step = find_latest_checkpoint(model_dir, CHECKPOINT_PREFIX)
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

        for _micro in range(grad_accum_steps):
            ids, tgts = next(batches)
            token_ids = torch.tensor(ids, device=device)
            targets = torch.tensor(tgts, device=device)
            loss, embedding = compute_loss(model, sig_reg, token_ids, targets)
            (loss / grad_accum_steps).backward()
            accum_loss += loss.item()

        optimizer.step()
        avg_loss = accum_loss / grad_accum_steps

        if i % log_every == 0:
            elapsed = time.time() - t0
            sps = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"  step {step:>6d} | loss {avg_loss:.4f} | {sps:.1f} steps/s")

    final_step = start_step + steps

    # Save checkpoint
    ckpt = f"{model_dir}/{CHECKPOINT_PREFIX}-{final_step}.safetensors"
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
    save_training_artifacts(model_dir, final_step, embedding_list, weights_bytes)

    await (vol or volume).commit.aio()
    print(f"Checkpoint + artifacts saved at step {final_step}")

    return final_step


async def _do_training_flax(
    steps: int,
    model_dir: str = MODEL_DIR,
    vol: modal.Volume | None = None,
    grad_accum_steps: int = GRAD_ACCUM_STEPS,
    log_every: int = LOG_EVERY,
) -> int:
    """Flax/JAX training path."""
    import jax
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
    from quickstart.common import (
        find_latest_checkpoint,
        save_training_artifacts,
    )

    print(f"[flax] JAX backend: {jax.default_backend()}")

    os.makedirs(model_dir, exist_ok=True)

    # TODO: checkpoint restore for flax (init fresh for now)
    ckpt_path, ckpt_step = find_latest_checkpoint(model_dir, CHECKPOINT_PREFIX)
    start_step = ckpt_step if ckpt_path else 0

    print("Initializing Flax model...")
    rngs = nnx.Rngs(0)
    model = Model(ModelConfig(dropout_rate=DROPOUT_RATE), rngs)
    model.train()
    sig_reg = SIGReg(SIGRegConfig(), rngs)

    optimizer = nnx.Optimizer(
        model, optax.adam(learning_rate=LEARNING_RATE), wrt=nnx.Param
    )

    @nnx.jit
    def micro_step(model, sig_reg, token_ids, targets):
        def loss_fn(model, sig_reg):
            return compute_loss(model, sig_reg, token_ids, targets)
        (loss, embedding), grads = nnx.value_and_grad(loss_fn, has_aux=True)(
            model, sig_reg
        )
        return loss, embedding, grads

    print(f"Training steps {start_step} -> {start_step + steps}")
    batches = make_batches(MICRO_BATCH_SIZE)
    first_ids, first_tgts = next(batches)

    t0 = time.time()
    embedding = None
    for i in range(steps):
        step = start_step + i
        accum_loss = jnp.zeros(())
        accum_grads = None

        for micro in range(grad_accum_steps):
            if i == 0 and micro == 0:
                ids, tgts = first_ids, first_tgts
            else:
                ids, tgts = next(batches)
            token_ids = jnp.array(ids)
            targets = jnp.array(tgts)

            loss, embedding, grads = micro_step(model, sig_reg, token_ids, targets)

            if i == 0 and micro == 0:
                print("JAX compiled the training step (first step is slow).")

            accum_loss = accum_loss + loss
            if accum_grads is None:
                accum_grads = grads
            else:
                accum_grads = jax.tree.map(jnp.add, accum_grads, grads)

        accum_grads = jax.tree.map(lambda g: g / grad_accum_steps, accum_grads)
        optimizer.update(model, accum_grads)
        avg_loss = accum_loss / grad_accum_steps

        if i % log_every == 0:
            elapsed = time.time() - t0
            sps = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"  step {step:>6d} | loss {float(avg_loss):.4f} | {sps:.1f} steps/s")

    final_step = start_step + steps

    # Save checkpoint
    ckpt = f"{model_dir}/{CHECKPOINT_PREFIX}-{final_step}.safetensors"
    model.save(ckpt)

    # Flatten embedding to 1D
    if embedding is not None and hasattr(embedding, 'ndim') and embedding.ndim == 2:
        embedding = jnp.mean(embedding, axis=0)
    embedding_list = embedding.tolist() if embedding is not None else []

    # Save artifacts
    weights_bytes = model.save_bytes()
    save_training_artifacts(model_dir, final_step, embedding_list, weights_bytes)

    await (vol or volume).commit.aio()
    print(f"Checkpoint + artifacts saved at step {final_step}")

    return final_step


# ---------------------------------------------------------------------------
# Commit — pure CPU, reads artifacts from volume
# ---------------------------------------------------------------------------


async def do_commit(localnet: bool = True, model_dir: str = MODEL_DIR, vol: modal.Volume | None = None) -> dict:
    """Encrypt + upload weights from the latest training artifacts, commit on-chain.

    Reads embedding and serialized weights saved by training — no GPU or
    model framework needed.
    """
    from soma_sdk import Keypair, SomaClient
    from quickstart.common import (
        DEFAULT_STATE,
        find_latest_checkpoint,
        load_training_state,
        load_training_artifacts,
        save_training_state,
        upload_to_s3,
    )

    _vol = vol or volume
    await _vol.reload.aio()

    if localnet:
        state = dict(DEFAULT_STATE)
    else:
        state = load_training_state(model_dir)
        if state["pending_reveal"]:
            print(f"Skipping commit — model {state['model_id']} has a pending reveal.")
            print(f"  Reveal it first (epoch must advance past {state['commit_epoch']}).")
            return state

    # Find latest checkpoint and its artifacts
    _ckpt_path, ckpt_step = find_latest_checkpoint(model_dir, CHECKPOINT_PREFIX)
    if ckpt_step == 0:
        raise RuntimeError("No checkpoint found on volume — train first!")

    artifacts = load_training_artifacts(model_dir, ckpt_step)
    if artifacts is None:
        raise RuntimeError(
            f"No training artifacts for step {ckpt_step}. "
            f"Re-run training (or save artifacts manually)."
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
    if localnet:
        from quickstart.localnet import upload_weights

        client = await SomaClient(chain="localnet")
        sys_state = await client.get_latest_system_state()
        current_epoch = sys_state.epoch
        weights_url = upload_weights(
            encrypted, f"model-step-{ckpt_step}", current_epoch
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

    # Create model on first commit, then commit weights
    signer = Keypair.from_secret_key(os.environ["SOMA_SECRET_KEY"])

    if state["model_id"] is None:
        model_id = await client.create_model(
            signer=signer,
            commission_rate=1000,
        )
        state["model_id"] = model_id
        print(f"Created NEW model: {model_id}")

    await client.commit_model(
        signer=signer,
        model_id=state["model_id"],
        weights_url=weights_url,
        encrypted_weights=encrypted,
        decryption_key=decryption_key,
        embedding=embedding,
    )
    print(f"Committed model: {state['model_id']}")

    state["pending_reveal"] = True
    state["commit_epoch"] = current_epoch

    if not localnet:
        save_training_state(state, model_dir)
        await _vol.commit.aio()

    print(f"\nCommit complete at epoch {current_epoch}")
    print(f"  Model: {state['model_id']}")
    print(f"  Step: {ckpt_step}")
    print(f"  Reveal valid at epoch >= {current_epoch + 1}")

    return state


# ---------------------------------------------------------------------------
# Reveal — reveal_model after epoch advances past commit epoch
# ---------------------------------------------------------------------------


async def do_reveal(localnet: bool = True, model_dir: str = MODEL_DIR, vol: modal.Volume | None = None, state: dict | None = None) -> dict | None:
    """Check epoch and reveal if ready. Returns updated state if revealed, None otherwise.

    If *state* is provided (localnet), uses it directly instead of loading from disk.
    """
    from quickstart.common import (
        load_training_state,
        save_training_state,
    )

    if state is None:
        _vol = vol or volume
        await _vol.reload.aio()
        state = load_training_state(model_dir)

    if not state["pending_reveal"]:
        print("No pending reveal — nothing to do.")
        return None

    print(f"Model: {state['model_id']}")
    print(f"Committed at epoch: {state['commit_epoch']}")

    from soma_sdk import Keypair, SomaClient

    if localnet:
        client = await SomaClient(chain="localnet")
    else:
        client = await SomaClient(chain="testnet")

    sys_state = await client.get_latest_system_state()
    current_epoch = sys_state.epoch
    print(f"Current epoch: {current_epoch}")

    if current_epoch <= state["commit_epoch"]:
        print(f"Epoch hasn't advanced yet (need > {state['commit_epoch']}). "
              f"Will retry later.")
        return None

    print(f"Epoch {current_epoch} > commit epoch {state['commit_epoch']} — revealing...")
    signer = Keypair.from_secret_key(os.environ["SOMA_SECRET_KEY"])

    try:
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

    if not localnet:
        save_training_state(state, model_dir)
        await _vol.commit.aio()

    print(f"Model {state['model_id']} revealed at epoch {current_epoch}!")
    return state



# ---------------------------------------------------------------------------
# Modal functions — thin wrappers
# ---------------------------------------------------------------------------


@app.function(
    image=cpu_image,
    timeout=600,
    volumes={MODEL_DIR: volume},
    secrets=[modal.Secret.from_name("soma-secrets")],
)
async def commit(localnet: bool = False):
    """Commit the latest checkpoint on the volume (CPU only)."""
    return await do_commit(localnet=localnet)


@app.function(
    image=cpu_image,
    schedule=modal.Cron("0 */6 * * *"),  # every 6h — epochs are 24h
    timeout=600,
    volumes={MODEL_DIR: volume},
    secrets=[modal.Secret.from_name("soma-secrets")],
)
async def reveal(
    localnet: bool = False,
    auto_continue: bool = True,
    steps_per_round: int = 500,
):
    """Reveal the model if the epoch has advanced. Optionally spawn next round."""
    result = await do_reveal(localnet=localnet)

    if result is not None and auto_continue:
        fw = result.get("framework", "torch")
        print(f"\nSpawning next training round ({steps_per_round} steps, {fw})...")
        train_and_commit.spawn(
            steps=steps_per_round,
            localnet=localnet,
            framework=fw,
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
    localnet: bool = False,
    framework: str = "torch",
):
    """Train for N steps then commit. GPU is released after commit."""
    print(f"\n{'='*60}")
    print(f"TRAIN AND COMMIT (steps={steps}, framework={framework})")
    print(f"{'='*60}\n")

    await do_training(steps, framework=framework)

    state = await do_commit(localnet=localnet)
    state["framework"] = framework
    from quickstart.common import save_training_state
    save_training_state(state, MODEL_DIR)
    await volume.commit.aio()
    return state


# ---------------------------------------------------------------------------
# Localnet — full cycle in one container (no cron)
# ---------------------------------------------------------------------------


localnet_volume = modal.Volume.from_name("soma-localnet-data", create_if_missing=True)


@app.cls(
    image=localnet_image,
    gpu="H100",
    timeout=86400,
    volumes={LOCALNET_MODEL_DIR: localnet_volume},
    secrets=[modal.Secret.from_name("soma-secrets")],
)
class LocalnetTrainer:
    """Runs a real Soma localnet inside Modal for the full train/commit/reveal cycle."""

    @modal.enter()
    def setup(self):
        import asyncio
        from quickstart.localnet import start_localnet, start_weights_server

        self.localnet_proc = asyncio.run(start_localnet())
        self.weights_server = start_weights_server()

        async def _fund():
            from soma_sdk import Keypair, SomaClient

            kp = Keypair.from_secret_key(os.environ["SOMA_SECRET_KEY"])
            client = await SomaClient(chain="localnet")
            await client.request_faucet(kp.address())
            balance = await client.get_balance(kp.address())
            print(f"Funded {kp.address()} with {balance:.2f} SOMA")

        asyncio.run(_fund())

    @modal.method()
    async def run(self, steps: int = 500, framework: str = "torch"):
        """Train → create → commit → advance epoch → reveal."""
        from soma_sdk import SomaClient

        print(f"\n{'='*60}")
        print(f"LOCALNET: TRAIN + COMMIT + REVEAL (steps={steps}, framework={framework})")
        print(f"{'='*60}\n")

        await do_training(
            steps, framework=framework,
            model_dir=LOCALNET_MODEL_DIR, vol=localnet_volume,
            grad_accum_steps=4, log_every=1,
        )
        state = await do_commit(localnet=True, model_dir=LOCALNET_MODEL_DIR, vol=localnet_volume)

        # Advance epoch so reveal is possible
        client = await SomaClient(chain="localnet")
        new_epoch = await client.advance_epoch()
        print(f"\nAdvanced epoch to {new_epoch}")

        state = await do_reveal(localnet=True, model_dir=LOCALNET_MODEL_DIR, vol=localnet_volume, state=state)
        return state

    @modal.exit()
    def teardown(self):
        if hasattr(self, "localnet_proc"):
            self.localnet_proc.terminate()
            self.localnet_proc.wait()


# ---------------------------------------------------------------------------
# Entrypoints
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def localnet(steps_per_round: int = 10, framework: str = "torch"):
    """Full train + commit + reveal cycle on localnet (dev/test)."""
    LocalnetTrainer().run.remote(steps=steps_per_round, framework=framework)


@app.local_entrypoint()
def main(steps_per_round: int = 500, framework: str = "torch"):
    """Train + commit on testnet. Deploy this app for automated reveal + re-training."""
    train_and_commit.remote(
        steps=steps_per_round, localnet=False, framework=framework,
    )
    print("Committed. Deploy with `uv run modal deploy src/quickstart/training.py` for automated reveal.")
