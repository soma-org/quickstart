"""Show the status of your Soma model, targets, and account.

Queries on-chain state to show:
  - Current epoch + timing
  - Account balance
  - Your model's status (pending/active/inactive)
  - Open targets your model is competing in
  - Claimable rewards
  - Network stats

Usage:
    uv run soma-status
    uv run soma-status --model-id 0x...
"""

import argparse
import asyncio
import json
import os
import time

from dotenv import load_dotenv
from soma_sdk import Keypair, SomaClient

load_dotenv()

TRAINING_STATE_FILE = "training_state.json"


def load_model_id_from_training_state() -> str | None:
    if os.path.exists(TRAINING_STATE_FILE):
        with open(TRAINING_STATE_FILE) as f:
            state = json.load(f)
        return state.get("model_id")
    return None


def format_duration_ms(ms: int) -> str:
    seconds = ms // 1000
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    if hours > 0:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


async def show_status(model_id: str | None):
    secret_key = os.environ.get("SOMA_SECRET_KEY")
    if not secret_key:
        print("Error: SOMA_SECRET_KEY not set in environment or .env file")
        return

    kp = Keypair.from_secret_key(secret_key)
    sender = kp.address()
    client = await SomaClient(chain="testnet")

    # -- Account --
    balance = await client.get_balance(sender)
    print(f"{'='*60}")
    print(f"SOMA STATUS")
    print(f"{'='*60}")
    print(f"\nAccount: {sender}")
    print(f"Balance: {balance:.4f} SOMA")

    # -- Epoch info --
    state = await client.get_latest_system_state()
    epoch = state.epoch
    epoch_duration_ms = state.parameters.epoch_duration_ms
    epoch_start_ms = state.epoch_start_timestamp_ms
    now_ms = int(time.time() * 1000)
    elapsed_ms = now_ms - epoch_start_ms
    remaining_ms = max(0, epoch_duration_ms - elapsed_ms)

    print(f"\n--- Epoch ---")
    print(f"Current epoch:    {epoch}")
    print(f"Epoch duration:   {format_duration_ms(epoch_duration_ms)}")
    print(f"Time remaining:   ~{format_duration_ms(remaining_ms)}")

    # -- Network stats --
    target_st = state.target_state
    emission = state.emission_pool
    print(f"\n--- Network ---")
    print(f"Emission/epoch:   {SomaClient.to_soma(emission.emission_per_epoch):.2f} SOMA")
    print(f"Reward/target:    {SomaClient.to_soma(target_st.reward_per_target):.4f} SOMA")
    print(f"Targets this epoch: {target_st.targets_generated_this_epoch}")
    print(f"Hits this epoch:  {target_st.hits_this_epoch}")
    print(f"Validators:       {len(state.validators.validators)}")

    # -- Model status --
    if model_id:
        print(f"\n--- Model: {model_id} ---")

        registry = state.model_registry

        # Check pending
        pending = vars(registry.pending_models) if hasattr(registry, 'pending_models') else {}
        if model_id in pending:
            m = pending[model_id]
            print(f"Status:           PENDING (committed, awaiting reveal)")
            print(f"Owner:            {m.owner}")
            print(f"Commit epoch:     {getattr(m, 'commit_epoch', '?')}")

        # Check active
        active = vars(registry.active_models) if hasattr(registry, 'active_models') else {}
        if model_id in active:
            m = active[model_id]
            print(f"Status:           ACTIVE ✓")
            print(f"Owner:            {m.owner}")
            if hasattr(m, 'staking_pool'):
                pool_soma = SomaClient.to_soma(m.staking_pool.soma_balance)
                print(f"Staked:           {pool_soma:.2f} SOMA")
            if hasattr(m, 'commission_rate'):
                print(f"Commission:       {m.commission_rate / 100:.1f}%")

        # Check inactive
        inactive = vars(registry.inactive_models) if hasattr(registry, 'inactive_models') else {}
        if model_id in inactive:
            print(f"Status:           INACTIVE")

        if model_id not in pending and model_id not in active and model_id not in inactive:
            print(f"Status:           NOT FOUND in model registry")

        # -- Targets with this model --
        print(f"\n--- Targets ---")
        open_targets = await client.get_targets(status="open")
        my_targets = [t for t in open_targets if model_id in t.model_ids]
        print(f"Open targets with your model: {len(my_targets)}")
        for t in my_targets[:10]:
            reward = SomaClient.to_soma(t.reward_pool)
            print(f"  {t.id}  threshold={t.distance_threshold:.6f}  "
                  f"reward={reward:.4f} SOMA  models={len(t.model_ids)}")
        if len(my_targets) > 10:
            print(f"  ... and {len(my_targets) - 10} more")

    # -- Claimable rewards --
    print(f"\n--- Claimable Rewards ---")
    claimable = await client.get_targets(status="claimable")
    if claimable:
        total_reward = sum(t.reward_pool for t in claimable)
        print(f"Claimable targets: {len(claimable)}")
        print(f"Total pool:        {SomaClient.to_soma(total_reward):.4f} SOMA")
        for t in claimable[:5]:
            reward = SomaClient.to_soma(t.reward_pool)
            winner = t.winning_model_id or "?"
            print(f"  {t.id}  reward={reward:.4f} SOMA  winner={winner[:20]}...")
        if len(claimable) > 5:
            print(f"  ... and {len(claimable) - 5} more")
        print(f"\nRun `uv run settle-targets` to claim.")
    else:
        print("No claimable targets.")

    # -- Stakes --
    print(f"\n--- Your Stakes ---")
    staked_objects = await client.list_owned_objects(
        sender, object_type="staked_soma"
    )
    if staked_objects:
        print(f"Staked objects: {len(staked_objects)}")
        for s in staked_objects:
            print(f"  id={s.id}")
    else:
        print("No active stakes.")
        if model_id:
            print(f"Run `uv run stake-model --model-id {model_id}` to stake.")

    print(f"\n{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Show Soma model and network status")
    parser.add_argument(
        "--model-id",
        help="Model ID to check (reads from training_state.json if omitted)",
    )
    args = parser.parse_args()

    model_id = args.model_id or load_model_id_from_training_state()
    asyncio.run(show_status(model_id))


if __name__ == "__main__":
    main()
