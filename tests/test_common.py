"""Tests for quickstart.common — game state, checkpoints, artifacts, mock chain."""

import asyncio
import json
import os
import time

import pytest

from quickstart.common import (
    DEFAULT_STATE,
    MockSomaClient,
    find_latest_checkpoint,
    load_training_state,
    load_training_artifacts,
    mock_upload_to_s3,
    save_training_state,
    save_training_artifacts,
)


# ---------------------------------------------------------------------------
# Game state
# ---------------------------------------------------------------------------


class TestTrainingState:
    def test_load_returns_defaults_when_missing(self, tmp_path):
        state = load_training_state(str(tmp_path))
        assert state == DEFAULT_STATE
        assert state["model_id"] is None
        assert state["pending_reveal"] is False
        assert state["is_update"] is False

    def test_save_and_load_round_trip(self, tmp_path):
        state = dict(DEFAULT_STATE)
        state["model_id"] = "0xabc123"
        state["step"] = 1500
        state["pending_reveal"] = True
        state["is_update"] = True
        state["commit_epoch"] = 42
        state["decryption_key"] = "key123"
        state["embedding"] = [0.1, 0.2, 0.3]

        save_training_state(state, str(tmp_path))
        loaded = load_training_state(str(tmp_path))
        assert loaded == state

    def test_save_creates_directory(self, tmp_path):
        nested = tmp_path / "a" / "b" / "c"
        save_training_state({"model_id": "test"}, str(nested))
        assert nested.exists()
        assert (nested / "training_state.json").exists()

    def test_load_returns_fresh_copy(self, tmp_path):
        """Mutating returned state doesn't affect subsequent loads."""
        s1 = load_training_state(str(tmp_path))
        s1["model_id"] = "mutated"
        s2 = load_training_state(str(tmp_path))
        assert s2["model_id"] is None

    def test_overwrite_existing_state(self, tmp_path):
        save_training_state({"model_id": "first"}, str(tmp_path))
        save_training_state({"model_id": "second"}, str(tmp_path))
        loaded = load_training_state(str(tmp_path))
        assert loaded["model_id"] == "second"


# ---------------------------------------------------------------------------
# Checkpoint finding
# ---------------------------------------------------------------------------


class TestFindLatestCheckpoint:
    def test_empty_directory(self, tmp_path):
        path, step = find_latest_checkpoint(str(tmp_path))
        assert path is None
        assert step == 0

    def test_single_checkpoint(self, tmp_path):
        (tmp_path / "checkpoint-500.safetensors").touch()
        path, step = find_latest_checkpoint(str(tmp_path))
        assert step == 500
        assert path.endswith("checkpoint-500.safetensors")

    def test_picks_highest_step_not_alphabetical(self, tmp_path):
        """Regression: alphabetical sort puts 9500 after 10000."""
        for s in [500, 1000, 9500, 10000]:
            (tmp_path / f"checkpoint-{s}.safetensors").touch()
        path, step = find_latest_checkpoint(str(tmp_path))
        assert step == 10000
        assert path.endswith("checkpoint-10000.safetensors")

    def test_non_sequential_steps(self, tmp_path):
        for s in [100, 50000, 3, 999]:
            (tmp_path / f"checkpoint-{s}.safetensors").touch()
        path, step = find_latest_checkpoint(str(tmp_path))
        assert step == 50000

    def test_custom_prefix(self, tmp_path):
        (tmp_path / "mymodel-100.safetensors").touch()
        (tmp_path / "mymodel-200.safetensors").touch()
        (tmp_path / "checkpoint-9999.safetensors").touch()
        path, step = find_latest_checkpoint(str(tmp_path), prefix="mymodel")
        assert step == 200

    def test_ignores_non_matching_files(self, tmp_path):
        (tmp_path / "checkpoint-500.safetensors").touch()
        (tmp_path / "checkpoint-invalid.safetensors").touch()
        (tmp_path / "other-file.safetensors").touch()
        (tmp_path / "checkpoint-300.txt").touch()
        path, step = find_latest_checkpoint(str(tmp_path))
        assert step == 500


# ---------------------------------------------------------------------------
# Training artifacts
# ---------------------------------------------------------------------------


class TestTrainingArtifacts:
    def test_save_and_load_round_trip(self, tmp_path):
        embedding = [0.1, 0.2, 0.3, 0.4]
        weights = b"\x00\x01\x02\x03" * 100

        save_training_artifacts(str(tmp_path), 500, embedding, weights)
        result = load_training_artifacts(str(tmp_path), 500)

        assert result is not None
        loaded_emb, loaded_weights = result
        assert loaded_emb == embedding
        assert loaded_weights == weights

    def test_load_returns_none_for_missing_step(self, tmp_path):
        save_training_artifacts(str(tmp_path), 500, [0.1], b"data")
        assert load_training_artifacts(str(tmp_path), 999) is None

    def test_load_returns_none_when_artifacts_json_missing(self, tmp_path):
        # Only weights file exists, no JSON
        (tmp_path / "weights-500.bin").write_bytes(b"data")
        assert load_training_artifacts(str(tmp_path), 500) is None

    def test_load_returns_none_when_weights_missing(self, tmp_path):
        # Only JSON exists, no weights
        (tmp_path / "artifacts-500.json").write_text('{"step": 500, "embedding": [0.1]}')
        assert load_training_artifacts(str(tmp_path), 500) is None

    def test_large_embedding(self, tmp_path):
        """Verify 2048-dim embedding (production size) round-trips."""
        embedding = [float(i) / 2048 for i in range(2048)]
        weights = os.urandom(1024 * 1024)  # 1MB

        save_training_artifacts(str(tmp_path), 1, embedding, weights)
        loaded_emb, loaded_weights = load_training_artifacts(str(tmp_path), 1)

        assert len(loaded_emb) == 2048
        assert loaded_weights == weights

    def test_multiple_steps_independent(self, tmp_path):
        save_training_artifacts(str(tmp_path), 100, [0.1], b"weights-100")
        save_training_artifacts(str(tmp_path), 200, [0.2], b"weights-200")

        emb1, w1 = load_training_artifacts(str(tmp_path), 100)
        emb2, w2 = load_training_artifacts(str(tmp_path), 200)

        assert emb1 == [0.1]
        assert w1 == b"weights-100"
        assert emb2 == [0.2]
        assert w2 == b"weights-200"


# ---------------------------------------------------------------------------
# Mock S3
# ---------------------------------------------------------------------------


class TestMockS3:
    def test_saves_file_and_returns_url(self, tmp_path):
        data = b"encrypted-weights-data"
        url = mock_upload_to_s3(data, "model-step-500", epoch=3, upload_dir=str(tmp_path))

        assert "mock-s3.example.com" in url
        assert "models/3/" in url
        assert "model-step-500.safetensors.enc" in url

        # Verify file was actually written
        saved = tmp_path / "mock_s3" / "models" / "3" / "model-step-500.safetensors.enc"
        assert saved.exists()
        assert saved.read_bytes() == data


# ---------------------------------------------------------------------------
# MockSomaClient — epoch timing
# ---------------------------------------------------------------------------


class TestMockSomaClientEpochs:
    def test_epoch_starts_at_zero(self, tmp_path):
        client = MockSomaClient(str(tmp_path), epoch_duration_s=1000.0)
        assert client.current_epoch == 0

    def test_epoch_advances_with_time(self, tmp_path):
        # Set start_time in the past so we're in a later epoch
        client = MockSomaClient(str(tmp_path), epoch_duration_s=10.0)
        client.chain["start_time"] = time.time() - 25  # 25s ago, 10s epochs = epoch 2
        assert client.current_epoch == 2

    def test_epoch_persists_across_instances(self, tmp_path):
        c1 = MockSomaClient(str(tmp_path), epoch_duration_s=100.0)
        start = c1.chain["start_time"]

        c2 = MockSomaClient(str(tmp_path), epoch_duration_s=100.0)
        assert c2.chain["start_time"] == start

    @pytest.mark.asyncio
    async def test_get_latest_system_state(self, tmp_path):
        client = MockSomaClient(str(tmp_path), epoch_duration_s=1000.0)
        state = await client.get_latest_system_state()
        assert state.epoch == 0


# ---------------------------------------------------------------------------
# MockSomaClient — commit / reveal lifecycle
# ---------------------------------------------------------------------------


class TestMockCommitReveal:
    @pytest.mark.asyncio
    async def test_commit_returns_model_id(self, tmp_path):
        client = MockSomaClient(str(tmp_path), epoch_duration_s=1000.0)
        model_id = await client.commit_model(
            signer=None,
            weights_url="https://example.com/w.enc",
            encrypted_weights=b"enc",
            decryption_key="key",
            embedding=[0.1],
            commission_rate=1000,
        )
        assert model_id.startswith("mock-model-")
        assert model_id in client.chain["models"]

    @pytest.mark.asyncio
    async def test_commit_increments_model_seq(self, tmp_path):
        client = MockSomaClient(str(tmp_path), epoch_duration_s=1000.0)
        id1 = await client.commit_model(
            signer=None, weights_url="u", encrypted_weights=b"e",
            decryption_key="k", embedding=[0.1], commission_rate=1000,
        )
        id2 = await client.commit_model(
            signer=None, weights_url="u", encrypted_weights=b"e",
            decryption_key="k", embedding=[0.1], commission_rate=1000,
        )
        assert id1 != id2

    @pytest.mark.asyncio
    async def test_reveal_fails_in_same_epoch(self, tmp_path):
        client = MockSomaClient(str(tmp_path), epoch_duration_s=1000.0)
        model_id = await client.commit_model(
            signer=None, weights_url="u", encrypted_weights=b"e",
            decryption_key="k", embedding=[0.1], commission_rate=1000,
        )

        with pytest.raises(RuntimeError, match="Cannot reveal"):
            await client.reveal_model(
                signer=None, model_id=model_id,
                decryption_key="k", embedding=[0.1],
            )

    @pytest.mark.asyncio
    async def test_reveal_succeeds_after_epoch_advance(self, tmp_path):
        client = MockSomaClient(str(tmp_path), epoch_duration_s=10.0)
        model_id = await client.commit_model(
            signer=None, weights_url="u", encrypted_weights=b"e",
            decryption_key="k", embedding=[0.1], commission_rate=1000,
        )

        # Advance epoch by shifting start_time back
        client.chain["start_time"] -= 15
        assert client.current_epoch >= 1

        await client.reveal_model(
            signer=None, model_id=model_id,
            decryption_key="k", embedding=[0.1],
        )
        assert client.chain["models"][model_id]["revealed"] is True

    @pytest.mark.asyncio
    async def test_reveal_unknown_model_fails(self, tmp_path):
        client = MockSomaClient(str(tmp_path), epoch_duration_s=1000.0)

        with pytest.raises(RuntimeError, match="Unknown model_id"):
            await client.reveal_model(
                signer=None, model_id="nonexistent",
                decryption_key="k", embedding=[0.1],
            )

    @pytest.mark.asyncio
    async def test_commit_model_update(self, tmp_path):
        client = MockSomaClient(str(tmp_path), epoch_duration_s=10.0)
        model_id = await client.commit_model(
            signer=None, weights_url="u1", encrypted_weights=b"e",
            decryption_key="k1", embedding=[0.1], commission_rate=1000,
        )

        # Advance epoch, reveal
        client.chain["start_time"] -= 15
        await client.reveal_model(
            signer=None, model_id=model_id,
            decryption_key="k1", embedding=[0.1],
        )

        # Now update
        await client.commit_model_update(
            signer=None, model_id=model_id, weights_url="u2",
            encrypted_weights=b"e2", decryption_key="k2", embedding=[0.2],
        )
        assert client.chain["models"][model_id]["revealed"] is False
        assert client.chain["models"][model_id]["weights_url"] == "u2"

    @pytest.mark.asyncio
    async def test_commit_update_unknown_model_fails(self, tmp_path):
        client = MockSomaClient(str(tmp_path), epoch_duration_s=1000.0)

        with pytest.raises(RuntimeError, match="Unknown model_id"):
            await client.commit_model_update(
                signer=None, model_id="nonexistent", weights_url="u",
                encrypted_weights=b"e", decryption_key="k", embedding=[0.1],
            )

    @pytest.mark.asyncio
    async def test_reveal_model_update_same_epoch_rules(self, tmp_path):
        """reveal_model_update enforces the same epoch check as reveal_model."""
        client = MockSomaClient(str(tmp_path), epoch_duration_s=10.0)
        model_id = await client.commit_model(
            signer=None, weights_url="u", encrypted_weights=b"e",
            decryption_key="k", embedding=[0.1], commission_rate=1000,
        )
        # Advance, reveal first time
        client.chain["start_time"] -= 15
        await client.reveal_model(
            signer=None, model_id=model_id,
            decryption_key="k", embedding=[0.1],
        )

        # Commit update in current epoch
        await client.commit_model_update(
            signer=None, model_id=model_id, weights_url="u2",
            encrypted_weights=b"e2", decryption_key="k2", embedding=[0.2],
        )

        # reveal_model_update should fail in same epoch
        with pytest.raises(RuntimeError, match="Cannot reveal"):
            await client.reveal_model_update(
                signer=None, model_id=model_id,
                decryption_key="k2", embedding=[0.2],
            )

        # Advance epoch again
        client.chain["start_time"] -= 15
        await client.reveal_model_update(
            signer=None, model_id=model_id,
            decryption_key="k2", embedding=[0.2],
        )
        assert client.chain["models"][model_id]["revealed"] is True


# ---------------------------------------------------------------------------
# MockSomaClient — full game loop simulation
# ---------------------------------------------------------------------------


class TestMockFullLoop:
    @pytest.mark.asyncio
    async def test_two_round_game_loop(self, tmp_path):
        """Simulate: train → commit → advance epoch → reveal → train → commit update → advance → reveal update."""
        client = MockSomaClient(str(tmp_path), epoch_duration_s=10.0)
        state = dict(DEFAULT_STATE)

        # Round 1: commit new model
        model_id = await client.commit_model(
            signer=None, weights_url="u1", encrypted_weights=b"e1",
            decryption_key="k1", embedding=[0.1], commission_rate=1000,
        )
        state["model_id"] = model_id
        state["is_update"] = False
        state["pending_reveal"] = True
        state["commit_epoch"] = client.current_epoch

        # Same epoch → can't reveal
        assert client.current_epoch <= state["commit_epoch"]

        # Advance epoch
        client.chain["start_time"] -= 15
        assert client.current_epoch > state["commit_epoch"]

        # Reveal (initial)
        await client.reveal_model(
            signer=None, model_id=model_id,
            decryption_key="k1", embedding=[0.1],
        )
        state["pending_reveal"] = False

        # Round 2: commit update
        await client.commit_model_update(
            signer=None, model_id=model_id, weights_url="u2",
            encrypted_weights=b"e2", decryption_key="k2", embedding=[0.2],
        )
        state["is_update"] = True
        state["pending_reveal"] = True
        state["commit_epoch"] = client.current_epoch

        # Same epoch → can't reveal
        with pytest.raises(RuntimeError):
            await client.reveal_model_update(
                signer=None, model_id=model_id,
                decryption_key="k2", embedding=[0.2],
            )

        # Advance epoch
        client.chain["start_time"] -= 15

        # Reveal update
        await client.reveal_model_update(
            signer=None, model_id=model_id,
            decryption_key="k2", embedding=[0.2],
        )
        state["pending_reveal"] = False

        # Verify final state
        assert state["model_id"] == model_id
        assert state["is_update"] is True
        assert state["pending_reveal"] is False
        assert client.chain["models"][model_id]["revealed"] is True
        assert client.chain["models"][model_id]["weights_url"] == "u2"

    @pytest.mark.asyncio
    async def test_state_persists_across_client_instances(self, tmp_path):
        """Simulate commit and reveal happening in separate processes."""
        # Process 1: commit
        c1 = MockSomaClient(str(tmp_path), epoch_duration_s=10.0)
        model_id = await c1.commit_model(
            signer=None, weights_url="u", encrypted_weights=b"e",
            decryption_key="k", embedding=[0.1], commission_rate=1000,
        )

        # Advance epoch
        c1.chain["start_time"] -= 15
        c1._save()

        # Process 2: reveal (new client instance, reads from disk)
        c2 = MockSomaClient(str(tmp_path), epoch_duration_s=10.0)
        assert model_id in c2.chain["models"]
        assert c2.current_epoch > c2.chain["models"][model_id]["commit_epoch"]

        await c2.reveal_model(
            signer=None, model_id=model_id,
            decryption_key="k", embedding=[0.1],
        )
        assert c2.chain["models"][model_id]["revealed"] is True


# ---------------------------------------------------------------------------
# Integration: game state + artifacts + checkpoints together
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_full_checkpoint_and_artifact_workflow(self, tmp_path):
        """Simulate what do_training + do_commit would do."""
        model_dir = str(tmp_path)
        step = 500

        # Simulate training output
        (tmp_path / f"checkpoint-{step}.safetensors").write_bytes(b"model-data")
        embedding = [0.1] * 16
        weights = b"serialized-weights"
        save_training_artifacts(model_dir, step, embedding, weights)

        # Simulate commit reading
        ckpt_path, ckpt_step = find_latest_checkpoint(model_dir)
        assert ckpt_step == step

        artifacts = load_training_artifacts(model_dir, ckpt_step)
        assert artifacts is not None
        loaded_emb, loaded_weights = artifacts
        assert loaded_emb == embedding
        assert loaded_weights == weights

        # Save training state
        state = load_training_state(model_dir)
        state["model_id"] = "0xabc"
        state["step"] = ckpt_step
        state["embedding"] = loaded_emb
        state["pending_reveal"] = True
        state["commit_epoch"] = 5
        save_training_state(state, model_dir)

        # Verify state persisted
        reloaded = load_training_state(model_dir)
        assert reloaded["model_id"] == "0xabc"
        assert reloaded["step"] == 500
        assert reloaded["pending_reveal"] is True

    def test_multiple_training_rounds(self, tmp_path):
        """Simulate multiple training rounds with increasing step counts."""
        model_dir = str(tmp_path)

        for step in [500, 1000, 1500]:
            (tmp_path / f"checkpoint-{step}.safetensors").write_bytes(b"data")
            save_training_artifacts(model_dir, step, [float(step)], b"w")

        # Latest checkpoint should be 1500
        _, latest = find_latest_checkpoint(model_dir)
        assert latest == 1500

        # Artifacts for each step are independent
        for step in [500, 1000, 1500]:
            emb, _ = load_training_artifacts(model_dir, step)
            assert emb == [float(step)]
