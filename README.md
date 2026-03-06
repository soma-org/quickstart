# Soma Quickstart

Train a model, publish it on-chain, submit data against targets, and earn rewards — all on [Modal](https://modal.com) GPUs.

## Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- A [Modal](https://modal.com) account

## Setup

```bash
uv sync
uv run modal setup
```

### Secrets

Copy `.env.example` to `.env` and fill in the values:

- **`SOMA_SECRET_KEY`** — Base58-encoded Ed25519 secret key.
- **`HF_TOKEN`** — HuggingFace access token. Create one at <https://huggingface.co/settings/tokens>, then approve access to the [gated dataset](https://huggingface.co/datasets/bigcode/the-stack-v2-dedup).

#### Upload bucket (Cloudflare R2 recommended)

You need an S3-compatible bucket for uploading model weights and submission data. **Cloudflare R2** is the simplest option (no IAM, free egress):

1. Create a [Cloudflare account](https://dash.cloudflare.com/sign-up) and go to Storage & databases → **R2 object storage** → Overview.
    - Activate R2 Subscription on your account ($0/mo with 10 GB/month free)
2. Create a bucket (e.g. `soma-data`).
3. Enable public access: select your bucket → **Settings** → **Public Development URL** → enable the `r2.dev` subdomain. Copy the public URL (your S3_PUBLIC_URL).
4. Go back to R2 object storage Overiview. Under Account Details, copy the S3 API (your S3_ENDPOINT_URL). Then next to API Tokens, click **Manage**. Create a token with **Object Read & Write** permissions for your bucket. Copy the Access Key ID (your S3_ACCESS_KEY_ID) and Secret Access Key (your S3_SECRET_ACCESS_KEY).
5. Fill in your `.env`:
   - **`S3_BUCKET`** — your bucket name
   - **`S3_ACCESS_KEY_ID`** / **`S3_SECRET_ACCESS_KEY`** — from the API token
   - **`S3_ENDPOINT_URL`** — `https://<account-id>.r2.cloudflarestorage.com`
   - **`S3_PUBLIC_URL`** — your bucket's public URL (e.g. `https://pub-xxx.r2.dev`)

AWS S3 and GCS (via HMAC keys) are also supported — set `S3_ENDPOINT_URL` accordingly or leave it blank for AWS.

Then push them to Modal:

```bash
uv run create-secrets
```

## Training a Model

Each epoch (24h) you train your model, publish it on-chain, and then submit data against targets to earn rewards:

```
Train → Commit → (epoch boundary) → Reveal → Submit Data → Claim Rewards
```

### 1. Train

Train a Soma V1 model on [The Stack v2](https://huggingface.co/datasets/bigcode/the-stack-v2-dedup) using an H100 GPU. Two backends are available:

```bash
# PyTorch
uv run modal run --detach src/quickstart/train_torch.py --num-steps 500

# Flax/JAX
uv run modal run --detach src/quickstart/train_flax.py --num-steps 500
```

Both scripts stream data from The Stack v2, checkpoint every 500 steps to a Modal volume (`soma-training-data`), and save training artifacts (weights + embedding) alongside each checkpoint so they can be committed without a GPU.

### 2. Commit your model on-chain

Commit encrypts your latest checkpoint, uploads it to S3, and registers it on-chain. No GPU needed — runs on CPU:

```bash
uv run modal run src/quickstart/training.py::commit_entrypoint --no-mock
```

This writes a local `training_state.json` with your `model_id` for use by other CLI tools.

### 3. Reveal after the epoch advances

Models must be revealed one epoch after they are committed. Deploy the reveal cron to handle this automatically (checks every 6 hours):

```bash
uv run modal deploy src/quickstart/training.py
```

Or trigger a reveal manually:

```bash
uv run modal run src/quickstart/training.py::reveal_entrypoint --no-mock
```

### 4. Stake to your model

Models need stake to be competitive. Stake SOMA to your model after it's revealed:

```bash
uv run stake-model --model-id 0x...
```

### 5. Submit data

Once your model is active (revealed + staked), the submitter streams data from The Stack v2, finds entries that beat the distance threshold on open targets, uploads them to S3, and submits on-chain:

```bash
# Run interactively:
uv run modal run src/quickstart/submitter.py

# Deploy as a persistent service:
uv run modal deploy src/quickstart/submitter.py
```

### 6. Claim rewards

After targets settle, claim your rewards:

```bash
uv run settle-targets
```

## Automated Training Loop

Instead of running each step manually, you can automate the full training cycle. Train + commit in one shot, and the deployed cron reveals then spawns the next training round:

```bash
# Kick off the first round (train + commit):
uv run modal run src/quickstart/training.py::main --no-mock --steps-per-round 500

# Deploy the reveal cron (reveals when epoch advances, spawns next training round):
uv run modal deploy src/quickstart/training.py
```

### Testing with mock chain

All training commands default to `--mock` which simulates epoch timing without real on-chain transactions. Use this to verify your setup before going live:

```bash
# Mock mode (default) — 30s epochs, no real chain calls:
uv run modal run src/quickstart/training.py::main --steps-per-round 10 --epoch-duration 30
```

## Monitoring

Check your model status, open targets, claimable rewards, and network stats:

```bash
uv run soma-status
uv run soma-status --model-id 0x...
```

## Staking

```bash
# Stake to a model:
uv run stake-model --model-id 0x... --amount 10.0

# List your stakes:
uv run stake-model --list

# Withdraw a stake:
uv run stake-model --withdraw 0x...
```

All local CLI commands require `SOMA_SECRET_KEY` in your `.env` file.

## Project Structure

```
src/quickstart/
├── common.py              # Shared utilities — training state, S3, mock client, checkpoints
├── training.py            # Modal app — commit, reveal (cron), train+commit loop
├── submitter.py           # Modal app — data submission (score + upload + submit on-chain)
├── train_torch.py         # Modal app — PyTorch training on The Stack v2
├── train_flax.py          # Modal app — Flax/JAX training on The Stack v2
├── stake.py               # CLI — stake SOMA to models
├── status.py              # CLI — model, target, and network status dashboard
├── create_modal_secret.py # CLI — push .env secrets to Modal
└── settle_targets.py      # CLI — claim rewards from settled targets
```
