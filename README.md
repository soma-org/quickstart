# Soma Scoring Quickstart

Run GPU-accelerated scoring on [Modal](https://modal.com) and submit results from your local machine.

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

## Usage

### 1. Start the scoring server

This launches the scoring service on Modal with an L4 GPU:

```bash
uv run modal serve src/quickstart/scoring_server.py
```

Modal will print a URL for the scoring endpoint — copy it for the next step.

### 2. Send a scoring request

In a separate terminal, run the client with the endpoint URL from step 1:

```bash
uv run score <scoring-url>
```

The client fetches open targets and model manifests from the Soma testnet, then sends them to your Modal server for scoring.

## Training

Train a Soma V1 model on [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) using Modal with an A100 GPU. Two backends are available:

### PyTorch

```bash
uv run modal run src/quickstart/train_torch.py
```

### Flax/JAX

```bash
uv run modal run src/quickstart/train_flax.py
```

Both scripts stream data from HuggingFace, checkpoint every 500 steps to a Modal volume (`soma-training-data`), and save a final model at completion. Pass `--num-steps` to control training length (default: 10,000).

## Project Structure

```
src/quickstart/
├── scoring_server.py   # Modal app — GPU scoring server
├── scoring_client.py   # CLI client — sends requests to the server
├── train_torch.py      # Modal app — PyTorch training
└── train_flax.py       # Modal app — Flax/JAX training
```

## Deploying

To keep the server running without a local `modal serve` process:

```bash
uv run modal deploy src/quickstart/scoring_server.py
```
