"""Helpers for running a real Soma localnet inside Modal.

Starts ``soma start localnet --force-regenesis`` which provides:
  - Full node (port 9000)
  - Faucet (port 9123)
  - Scoring service (port 9124, GPU)
  - Admin service (port 9125, for advance_epoch)

All services run on localhost within the same Modal container.
"""

import asyncio
import http.server
import os
import socket
import subprocess
import sys
import threading

WEIGHTS_DIR = "/tmp/soma-localnet-weights"
WEIGHTS_PORT = 9200
RPC_PORT = 9000


def _port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


async def start_localnet() -> subprocess.Popen | None:
    """Start soma localnet and wait for all services to be healthy.

    Returns None if a localnet is already running (port already bound).
    """
    if _port_in_use(RPC_PORT):
        print("Localnet already running, skipping start")
        return None

    proc = subprocess.Popen(
        [
            "soma", "start", "localnet",
            "--force-regenesis",
        ],
        stdout=sys.stdout,
        stderr=sys.stderr,
    )

    from soma_sdk import SomaClient

    print("Starting localnet...", end="", flush=True)
    for _ in range(60):
        try:
            client = await SomaClient(chain="localnet")
            state = await client.get_latest_system_state()
            if state.epoch >= 0 and await client.scoring_health():
                print(f" ready (epoch {state.epoch})")
                return proc
        except Exception:
            pass
        await asyncio.sleep(1)

    proc.terminate()
    raise RuntimeError("Localnet failed to start within 60s")


def start_weights_server() -> http.server.HTTPServer:
    """Start an HTTP file server for weight files (scoring service fetches from here)."""
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    handler = lambda *args: http.server.SimpleHTTPRequestHandler(
        *args, directory=WEIGHTS_DIR
    )
    httpd = http.server.HTTPServer(("", WEIGHTS_PORT), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def upload_weights(data: bytes, key_name: str, epoch: int) -> str:
    """Save encrypted weights to disk and return a localhost URL."""
    dest = os.path.join(WEIGHTS_DIR, "models", str(epoch))
    os.makedirs(dest, exist_ok=True)
    filename = f"{key_name}.safetensors.enc"
    with open(os.path.join(dest, filename), "wb") as f:
        f.write(data)
    url = f"http://localhost:{WEIGHTS_PORT}/models/{epoch}/{filename}"
    print(f"  Saved {len(data)} bytes -> {url}")
    return url
