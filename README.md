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

## Project Structure

```
src/quickstart/
├── scoring_server.py   # Modal app — GPU scoring server
└── scoring_client.py   # CLI client — sends requests to the server
```

## Deploying

To keep the server running without a local `modal serve` process:

```bash
uv run modal deploy src/quickstart/scoring_server.py
```
