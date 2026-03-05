"""Create the 'soma-secrets' Modal secret from your .env file.

Run with:
    uv run create-secrets
"""

import modal
from dotenv import dotenv_values

SECRET_NAME = "soma-secrets"


def main():
    env = {k: v for k, v in dotenv_values(".env").items() if v}
    if not env:
        raise SystemExit("No keys found in .env")

    modal.Secret.objects.delete(SECRET_NAME, allow_missing=True)
    modal.Secret.objects.create(SECRET_NAME, env)
    print(f"✓ Modal secret '{SECRET_NAME}' created with keys: {', '.join(env)}")
