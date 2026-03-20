"""Claim rewards from settled targets on the SOMA network.

Usage:
    uv run claim
"""

import asyncio
import os

from dotenv import load_dotenv
from soma_sdk import Keypair, SomaClient

load_dotenv()


async def run():
    secret_key = os.environ.get("SOMA_SECRET_KEY")
    if not secret_key:
        print("Error: SOMA_SECRET_KEY not set in environment or .env file")
        return

    kp = Keypair.from_secret_key(secret_key)
    sender = kp.address()
    print(f"Sender: {sender}")

    client = await SomaClient(chain="testnet")

    balance = await client.get_balance(sender)
    print(f"Balance: {balance:.2f} SOMA")

    targets = await client.get_targets(status="claimable", submitter=sender)
    print(f"Found {len(targets)} claimable target(s)")

    if not targets:
        print("Nothing to settle.")
        return

    settled = 0
    for target in targets:
        print(f"\nSettling target {target.id} (reward_pool={target.reward_pool})")
        try:
            await client.claim_rewards(signer=kp, target_id=target.id)
            print(f"  ✓ Settled {target.id}")
            settled += 1
        except Exception as e:
            print(f"  ✗ Failed to settle {target.id}: {e}")

    balance = await client.get_balance(sender)
    print(f"\nSettled {settled}/{len(targets)} target(s)")
    print(f"Balance after settling: {balance:.2f} SOMA")


def main():
    asyncio.run(run())


if __name__ == "__main__":
    main()
