# Soma Scoring Quickstart

Run GPU-accelerated scoring, training, and reward settlement on [Modal](https://modal.com).

## Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- A [Modal](https://modal.com) account

## Setup

Install dependencies and authenticate with Modal:

```bash
uv sync
uv run modal setup
```

### Secrets

All Modal apps use a single secret called `soma-secrets`, populated from your `.env` file. Copy `.env.example` to `.env` and fill in the values:

- **`SOMA_SECRET_KEY`** — Base58-encoded Ed25519 secret key.
- **`HF_TOKEN`** — HuggingFace access token. Create one at <https://huggingface.co/settings/tokens>.
- **`AWS_ACCESS_KEY_ID`** / **`AWS_SECRET_ACCESS_KEY`** — AWS credentials with read access to the `softwareheritage` S3 bucket.
- **`S3_BUCKET`** — S3-compatible bucket for uploading scored data (e.g. Cloudflare R2).
- **`S3_ACCESS_KEY_ID`** / **`S3_SECRET_ACCESS_KEY`** — credentials for the upload bucket.
- **`S3_ENDPOINT_URL`** *(optional)* — endpoint for non-AWS providers (e.g. `https://<account>.r2.cloudflarestorage.com`).
- **`S3_PUBLIC_URL`** *(optional)* — public base URL for the bucket (e.g. `https://pub-xxx.r2.dev`).

Then push the secrets to Modal:

```bash
uv run create-secrets
```

## Scoring

The scorer streams shuffled source files from The Stack v2, scores them against open targets on a GPU, and automatically uploads + submits on-chain when the distance threshold is met.

Run interactively (stays attached to your terminal):

```bash
uv run modal run src/quickstart/scorer.py
```

Deploy as a persistent service:

```bash
uv run modal deploy src/quickstart/scorer.py
```

## Training

Train a Soma V1 model on [The Stack v2](https://huggingface.co/datasets/bigcode/the-stack-v2-dedup) using an H100 GPU. Two backends are available:

### PyTorch

```bash
uv run modal run --detach src/quickstart/train_torch.py
```

### Flax/JAX

```bash
uv run modal run --detach src/quickstart/train_flax.py
```

Both scripts stream data from The Stack v2, checkpoint every 500 steps to a Modal volume (`soma-training-data`), and save a final model at completion. Pass `--num-steps` to control training length (default: 10,000).

## Settling Rewards

Claim rewards from all settled targets:

```bash
uv run settle-targets
```

This requires `SOMA_SECRET_KEY` in your `.env` file.

## Project Structure

```
src/quickstart/
├── scorer.py              # Modal app — GPU scoring + S3 upload + on-chain submission
├── train_torch.py         # Modal app — PyTorch training
├── train_flax.py          # Modal app — Flax/JAX training
├── create_modal_secret.py # CLI — push .env secrets to Modal
└── settle_targets.py      # CLI — claim rewards from settled targets
```
