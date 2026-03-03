"""Client that scores data against a target and submits if it hits.

Usage:
    uv run score <scoring-url>

The scoring URL is printed by `uv run modal serve src/quickstart/server.py`.
"""

import argparse
import asyncio
import os

import aiohttp
from dotenv import load_dotenv
from soma_sdk import Keypair, SomaClient

load_dotenv()


async def run(scoring_url: str):
    secret_key = os.environ.get("SOMA_SECRET_KEY")
    if secret_key:
        kp = Keypair.from_secret_key(secret_key)
    else:
        print(f"No SOMA_SECRET_KEY")
        return

    sender = kp.address()
    print(f"Sender: {sender}")
    client = await SomaClient(chain="testnet")


    # Find an open target
    targets = await client.get_targets(status="open")
    target = next(t for t in targets)
    print(f"\nTarget: {target.id}  threshold={target.distance_threshold}")

    manifests = await client.get_model_manifests(target)
    print(f"Fetched {len(manifests)} model manifest(s)")

    data_url = "https://github.com/soma-org/soma/releases/download/testnet-v0.1.6/SHA256SUMS"

    async with aiohttp.ClientSession() as session:
        async with session.get(data_url) as response:
            data_bytes = await response.read()
            data_size = len(data_bytes)
            data_checksum = client.commitment(data_bytes)

    print(f"Data size: {data_size}, checksum: {data_checksum}")

    payload = {
        "data_url": data_url,
        "models": [
            {"url": m.url, "checksum": m.checksum, "size": m.size, "decryption_key": m.decryption_key}
            for m in manifests
        ],
        "target_embedding": target.embedding,
        "data_checksum": data_checksum,
        "data_size": data_size,
    }

    print("\nScoring...")
    async with aiohttp.ClientSession() as session:
        async with session.post(scoring_url, json=payload) as response:
            result = await response.json()

    winner = result["winner"]
    distance = result["distance"][0]
    embedding = result["embedding"]
    loss_score = result["loss_score"]

    print(f"Winner: index={winner}  distance={distance:.6f}")
    print(f"Threshold: {target.distance_threshold}")

    # if distance > target.distance_threshold:
    #     print("\nDistance exceeds threshold — no hit, skipping submission.")
    #     return

    # # Submit the data on-chain
    # print("\n=== Submitting data ===")
    # await client.submit_data(
    #     signer=kp,
    #     target_id=target.id,
    #     data=data_bytes,
    #     data_url=data_url,
    #     model_id=target.model_ids[winner],
    #     embedding=embedding,
    #     distance_score=distance,
    #     loss_score=loss_score,
    # )
    # print("Submission successful!")
    # print(f"Target: {target.id}")
    # print(f"Model:  {target.model_ids[winner]}")
    # print(f"Distance: {distance:.6f} (threshold: {target.distance_threshold})")


def main():
    parser = argparse.ArgumentParser(description="Score data and submit to hit a target.")
    parser.add_argument("scoring_url", help="URL of the scoring endpoint (from `modal serve`)")
    args = parser.parse_args()
    asyncio.run(run(args.scoring_url))


if __name__ == "__main__":
    main()
