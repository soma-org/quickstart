# Soma Quickstart

Train a model, publish it on-chain, submit data against targets, and earn rewards — all on [Modal](https://modal.com) GPUs.

## Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- A [Modal](https://modal.com) account
- Latest soma binary [installed](https://docs.soma.org/getting-started/install/)

## Setup

```bash
uv sync
uv run modal setup
```

### Secrets

Copy `.env.example` to `.env` and fill in:

- **`SOMA_SECRET_KEY`** — Base58-encoded Ed25519 secret key. Get yours by running `soma wallet export`.
- **`HF_TOKEN`** — HuggingFace access token. Create one at <https://huggingface.co/settings/tokens>, then approve access to the [gated dataset](https://huggingface.co/datasets/bigcode/the-stack-v2-dedup).

### Upload Bucket (Cloudflare R2)

You need an S3-compatible bucket for uploading model weights and submission data. **Cloudflare R2** is the simplest option (no IAM, free egress):

1. Create a [Cloudflare account](https://dash.cloudflare.com/sign-up) and go to Storage & databases → **R2 object storage** → Overview.
    - Activate R2 Subscription on your account ($0/mo with 10 GB/month free)
2. Create a bucket (e.g. `soma-data`).
3. Enable public access: select your bucket → **Settings** → **Public Development URL** → enable the `r2.dev` subdomain. Copy the public URL (your S3_PUBLIC_URL).
4. Go back to R2 object storage Overview. Under Account Details, copy the S3 API (your S3_ENDPOINT_URL). Then next to API Tokens, click **Manage**. Create a token with **Object Read & Write** permissions for your bucket. Copy the Access Key ID (your S3_ACCESS_KEY_ID) and Secret Access Key (your S3_SECRET_ACCESS_KEY).
5. Fill in your `.env`:
   - **`S3_BUCKET`** — your bucket name
   - **`S3_ACCESS_KEY_ID`** / **`S3_SECRET_ACCESS_KEY`** — from the API token
   - **`S3_ENDPOINT_URL`** — `https://<account-id>.r2.cloudflarestorage.com`
   - **`S3_PUBLIC_URL`** — your bucket's public URL (e.g. `https://pub-xxx.r2.dev`)

AWS S3 and GCS are also supported if preferred.

### Push secrets to Modal

```bash
uv run create-secrets
```

---

## 1. Submit Data

The submitter streams data from [The Stack v2](https://huggingface.co/datasets/bigcode/the-stack-v2-dedup), scores it against open targets, and submits on-chain when the distance threshold is met. Run it interactively to see how it works:

```bash
uv run modal run src/quickstart/submitter.py
```

## 2. Localnet Training (Dev/Test)

Run a full **train → commit → reveal** cycle using an embedded Soma localnet inside Modal. No testnet access needed — one command, instant feedback:

```bash
# PyTorch (default):
uv run modal run src/quickstart/training.py::localnet

# Flax/JAX:
uv run modal run src/quickstart/training.py::localnet --framework flax

# More training steps:
uv run modal run src/quickstart/training.py::localnet --steps-per-round 20
```

This trains a small model, commits it on-chain, advances the epoch, and reveals it — the entire lifecycle in a single run. Use it to verify your setup and understand the flow before going to testnet.

## 3. Production (Testnet)

Graduate to testnet. The goal is to deploy two automations that run continuously: one trains your model each epoch, the other submits data to earn rewards.

**Kick off the first training round** — this trains on an H100 and commits on-chain:

```bash
uv run modal run src/quickstart/training.py --steps-per-round 500

# Or use Flax/JAX instead of PyTorch:
uv run modal run src/quickstart/training.py --steps-per-round 500 --framework flax
```

**Deploy** — a cron reveals your model when the epoch advances (every 24h), spawns the next training round, and the submitter scores data against open targets:

```bash
uv run modal deploy src/quickstart/training.py
uv run modal deploy src/quickstart/submitter.py
```

After this, everything runs without intervention. Each epoch your model is trained, committed, revealed, and the submitter earns against open targets.

Fork `training.py` and `submitter.py` to build your own strategies.

## 4. Claim Rewards

After targets settle, claim your rewards:

```bash
uv run claim
```

## Reference: Standalone Training Scripts

For studying the training loop without the commit/reveal machinery:

- `src/quickstart/train_torch.py` — PyTorch
- `src/quickstart/train_flax.py` — Flax/JAX

```bash
uv run modal run --detach src/quickstart/train_torch.py --num-steps 500
uv run modal run --detach src/quickstart/train_flax.py --num-steps 500
```

## Project Structure

```
src/quickstart/
├── common.py              # Shared utilities — training state, S3, checkpoints
├── training.py            # Modal app — train, commit, reveal (torch/flax)
├── submitter.py           # Modal app — data submission (score + upload + submit)
├── train_torch.py         # Standalone PyTorch training reference
├── train_flax.py          # Standalone Flax/JAX training reference
├── settle_targets.py      # CLI — claim rewards from settled targets
├── localnet.py            # Localnet helpers for dev/test
└── create_modal_secret.py # CLI — push .env secrets to Modal
```
