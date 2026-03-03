"""Soma scoring server — runs on Modal with a GPU.

Start locally with:
    uv run modal serve src/quickstart/scoring_server.py

Deploy to Modal with:
    uv run modal deploy src/quickstart/scoring_server.py
"""

import subprocess
import sys

import modal

app = modal.App("soma-scoring")

volume = modal.Volume.from_name("soma-scoring-data", create_if_missing=True)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.13"
    )
    .apt_install("curl")
    .env({"PATH": "/root/.local/bin:/usr/local/bin:/usr/bin:/bin", "RUST_LOG": "warn"})
    .run_commands(
        "curl -sSfL https://sup.soma.org | sh",
        "sup install soma",
    )
    .pip_install("soma-sdk>=0.1.7", "fastapi[standard]")
)

SCORING_PORT = 9124


@app.cls(image=image, gpu="L4", scaledown_window=300, timeout=900, volumes={"/data": volume})
class Scorer:
    @modal.enter()
    def start_soma(self):
        self.proc = subprocess.Popen(
            ["soma", "start", "scoring", "--device", "cuda", "--data-dir", "/data"],
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        import asyncio

        from soma_sdk import SomaClient

        async def _wait():
            for _ in range(30):
                try:
                    client = await SomaClient(
                        chain="testnet",
                        scoring_url=f"http://localhost:{SCORING_PORT}",
                    )
                    if await client.scoring_health():
                        return
                except Exception:
                    pass
                await asyncio.sleep(1)
            raise RuntimeError("Scoring service failed to start within 30s")

        asyncio.run(_wait())

    @modal.fastapi_endpoint(method="POST")
    async def score(self, request: dict):
        from types import SimpleNamespace

        from soma_sdk import SomaClient

        client = await SomaClient(
            chain="testnet",
            scoring_url=f"http://localhost:{SCORING_PORT}",
        )

        models = [SimpleNamespace(**m) for m in request["models"]]

        result = await client.score(
            data_url=request["data_url"],
            models=models,
            target_embedding=request["target_embedding"],
            data_checksum=request["data_checksum"],
            data_size=request["data_size"],
        )

        return {
            "winner": result.winner,
            "loss_score": result.loss_score,
            "embedding": result.embedding,
            "distance": result.distance,
        }
