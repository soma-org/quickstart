"""Stake SOMA tokens to a model.

Models need stake to be competitive — staking to your own model (or a model
you believe in) earns you a share of that model's target rewards.

Usage:
    # Stake to your own model (reads model_id from game state):
    uv run stake-model

    # Stake a specific amount to a specific model:
    uv run stake-model --model-id 0x... --amount 10.0

    # List your current staked objects:
    uv run stake-model --list

    # Withdraw a specific stake:
    uv run stake-model --withdraw 0x...
"""

import argparse
import asyncio
import json
import os

from dotenv import load_dotenv
from soma_sdk import Keypair, SomaClient

load_dotenv()

TRAINING_STATE_FILE = "training_state.json"


def load_model_id_from_training_state() -> str | None:
    """Try to read model_id from a local training_state.json."""
    if os.path.exists(TRAINING_STATE_FILE):
        with open(TRAINING_STATE_FILE) as f:
            state = json.load(f)
        return state.get("model_id")
    return None


async def stake_to_model(model_id: str, amount: float | None):
    secret_key = os.environ.get("SOMA_SECRET_KEY")
    if not secret_key:
        print("Error: SOMA_SECRET_KEY not set in environment or .env file")
        return

    kp = Keypair.from_secret_key(secret_key)
    sender = kp.address()
    client = await SomaClient(chain="testnet")

    balance = await client.get_balance(sender)
    print(f"Address: {sender}")
    print(f"Balance: {balance:.2f} SOMA")

    if amount:
        print(f"\nStaking {amount:.2f} SOMA to model {model_id}...")
    else:
        print(f"\nStaking all available SOMA to model {model_id}...")

    balance_before = await client.get_balance(sender)

    await client.add_stake_to_model(
        signer=kp,
        model_id=model_id,
        amount=amount,
    )

    balance_after = await client.get_balance(sender)
    staked = balance_before - balance_after
    print(f"✓ Staked {staked:.2f} SOMA to model {model_id}")
    print(f"  Balance: {balance_before:.2f} -> {balance_after:.2f} SOMA")


async def list_stakes():
    secret_key = os.environ.get("SOMA_SECRET_KEY")
    if not secret_key:
        print("Error: SOMA_SECRET_KEY not set in environment or .env file")
        return

    kp = Keypair.from_secret_key(secret_key)
    sender = kp.address()
    client = await SomaClient(chain="testnet")

    balance = await client.get_balance(sender)
    print(f"Address: {sender}")
    print(f"Balance: {balance:.2f} SOMA\n")

    staked_objects = await client.list_owned_objects(
        sender, object_type="staked_soma"
    )
    print(f"Staked objects: {len(staked_objects)}")
    for s in staked_objects:
        print(f"  id={s.id}  version={s.version}")

    if not staked_objects:
        print("  (none)")


async def withdraw(staked_soma_id: str):
    secret_key = os.environ.get("SOMA_SECRET_KEY")
    if not secret_key:
        print("Error: SOMA_SECRET_KEY not set in environment or .env file")
        return

    kp = Keypair.from_secret_key(secret_key)
    sender = kp.address()
    client = await SomaClient(chain="testnet")

    balance_before = await client.get_balance(sender)
    print(f"Address: {sender}")
    print(f"Balance: {balance_before:.2f} SOMA")

    print(f"\nWithdrawing stake {staked_soma_id}...")
    await client.withdraw_stake(signer=kp, staked_soma_id=staked_soma_id)

    balance_after = await client.get_balance(sender)
    print(f"✓ Withdrew stake")
    print(f"  Balance: {balance_before:.2f} -> {balance_after:.2f} SOMA")


def main():
    parser = argparse.ArgumentParser(description="Stake SOMA tokens to a model")
    parser.add_argument("--model-id", help="Model ID to stake to (reads from training_state.json if omitted)")
    parser.add_argument("--amount", type=float, help="Amount of SOMA to stake (stakes all if omitted)")
    parser.add_argument("--list", action="store_true", help="List current staked objects")
    parser.add_argument("--withdraw", metavar="STAKED_ID", help="Withdraw a staked object by ID")
    args = parser.parse_args()

    if args.list:
        asyncio.run(list_stakes())
    elif args.withdraw:
        asyncio.run(withdraw(args.withdraw))
    else:
        model_id = args.model_id or load_model_id_from_training_state()
        if not model_id:
            print("Error: No --model-id provided and no training_state.json found.")
            print("Run a commit first, or pass --model-id explicitly.")
            return
        asyncio.run(stake_to_model(model_id, args.amount))


if __name__ == "__main__":
    main()
