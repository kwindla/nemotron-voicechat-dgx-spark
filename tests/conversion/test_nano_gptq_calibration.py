import hashlib
import json
from pathlib import Path

import pytest

torch = pytest.importorskip(
    "torch",
    reason="conversion tensor tests run in the public runtime container",
)
from nano_gptq_calibration import (
    GIB,
    POLICY_W8_ALL_G128,
    PUBLIC_REPOSITORY,
    PUBLIC_REVISION,
    assert_byte_identical,
    assert_disjoint_corpora,
    bits_for_target,
    conversion_disk_preflight,
    effective_rows,
    gptq_quantization_config,
    gptq_quantize_matrix,
    group_size_for_target,
    inspect_replay_corpus,
    is_gptq_target,
    load_fused_sequences,
    pack_gptq_codes,
    select_replay_corpus,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _replay(root: Path, marker: float, *, revision: str = PUBLIC_REVISION) -> Path:
    root.mkdir()
    (root / "metadata.json").write_text(
        json.dumps(
            {
                "kind": "public_nano_deterministic_replay",
                "checkpoint_provenance": {
                    "repository": PUBLIC_REPOSITORY,
                    "revision": revision,
                    "model_sha256": "model",
                    "nano_sha256": "nano",
                    "derived_from_ea": False,
                },
            }
        )
    )
    calls = [
        {
            "inputs": {
                "input_embeds": torch.full((1, 3, 8), marker, dtype=torch.bfloat16),
                "request_id": "stream",
            }
        },
        {
            "inputs": {
                "input_embeds": torch.full((1, 1, 8), marker + 1, dtype=torch.bfloat16),
                "request_id": "stream",
            }
        },
    ]
    manifest = []
    for index, call in enumerate(calls):
        path = root / f"nano-call-{index:06d}.pt"
        torch.save(call, path)
        manifest.append(
            {
                "call_index": index,
                "file": path.name,
                "bytes": path.stat().st_size,
                "sha256": _sha(path),
                "decode_steps": 0 if index == 0 else 1,
                "request_id": "stream",
                "sequence_epoch": 0,
            }
        )
    (root / "manifest.jsonl").write_text("".join(json.dumps(record) + "\n" for record in manifest))
    return root


def test_selection_is_exactly_mlp_mamba_and_text_head():
    selected = [
        "stt_model.llm.layers.2.mixer.up_proj.weight",
        "stt_model.llm.layers.2.mixer.down_proj.weight",
        "stt_model.llm.layers.3.mixer.in_proj.weight",
        "stt_model.llm.layers.3.mixer.out_proj.weight",
        "stt_model.lm_head.weight",
    ]
    assert all(is_gptq_target(name) for name in selected)
    assert all(bits_for_target(name) == 8 for name in selected)
    assert group_size_for_target(selected[0]) == 128
    assert group_size_for_target(selected[1]) == 128
    assert group_size_for_target(selected[-1]) == 128
    for excluded in (
        "stt_model.function_head.weight",
        "stt_model.embed_tokens.weight",
        "stt_model.llm.layers.4.mixer.q_proj.weight",
        "stt_model.llm.layers.4.mixer.o_proj.weight",
    ):
        assert not is_gptq_target(excluded)
    config = gptq_quantization_config()
    assert config["bits"] == 8
    assert config["group_size"] == 128
    assert not any(key.startswith("+:") for key in config["dynamic"])


def test_authorized_w8_policy_uses_group128_for_all_105_targets():
    selected = [
        "stt_model.llm.layers.2.mixer.up_proj.weight",
        "stt_model.llm.layers.2.mixer.down_proj.weight",
        "stt_model.llm.layers.3.mixer.in_proj.weight",
        "stt_model.llm.layers.3.mixer.out_proj.weight",
        "stt_model.lm_head.weight",
    ]
    assert all(bits_for_target(name, POLICY_W8_ALL_G128) == 8 for name in selected)
    assert all(group_size_for_target(name, POLICY_W8_ALL_G128) == 128 for name in selected)
    config = gptq_quantization_config(POLICY_W8_ALL_G128)
    assert config["bits"] == 8
    assert config["group_size"] == 128
    assert not any(key.startswith("+:") for key in config["dynamic"])
    assert r"-:.*function_head$" in config["dynamic"]


def test_w8_runtime_padding_patch_is_exact_shape_and_precision_gated():
    source = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "nemotron_voicechat_runtime"
        / "patch_voicechat_marlin_g128_padding.py"
    ).read_text()
    assert "input_size_per_partition == 15680" in source
    assert "sum(output_partition_sizes) == 4480" in source
    assert "self.quant_config.group_size == 128" in source
    assert "self.quant_config.quant_type.size_bits == 8" in source
    assert "voicechat_input_padding = 64" in source


def test_preflight_enforces_projected_peak_and_hard_floor(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.safetensors").write_bytes(b"x" * 100)
    projected_output = 200
    projected_peak = 100 * 2 + projected_output + projected_output * 2
    result = conversion_disk_preflight(
        source,
        tmp_path / "output",
        tmp_path / "work",
        estimated_output_bytes=projected_output,
        free_bytes=projected_peak + 100 * GIB,
    )
    assert result["projected_remaining_bytes"] == 100 * GIB
    with pytest.raises(RuntimeError, match="below the 100.00 GiB floor"):
        conversion_disk_preflight(
            source,
            tmp_path / "output",
            tmp_path / "work",
            estimated_output_bytes=projected_output,
            free_bytes=projected_peak + 100 * GIB - 1,
        )
    with pytest.raises(ValueError, match="hard 100 GiB"):
        conversion_disk_preflight(
            source,
            tmp_path / "output",
            tmp_path / "work",
            estimated_output_bytes=projected_output,
            min_remaining_gib=99,
            free_bytes=1000 * GIB,
        )


def test_corpus_reconstructs_fused_prefill_and_decode_rows(tmp_path):
    root = _replay(tmp_path / "calibration", 2.0)
    corpus = inspect_replay_corpus([root])
    sequences = load_fused_sequences(corpus, torch)
    assert len(sequences) == 1
    assert tuple(sequences[0].shape) == (1, 4, 8)
    torch.testing.assert_close(
        sequences[0][:, :3], torch.full((1, 3, 8), 2.0, dtype=torch.bfloat16)
    )
    torch.testing.assert_close(
        sequences[0][:, 3:], torch.full((1, 1, 8), 3.0, dtype=torch.bfloat16)
    )
    assert effective_rows({"decode_steps": 0}, 7) == 7
    assert effective_rows({"decode_steps": 1}, 7) == 1


def test_conversations_are_inferred_from_first_seen_positive_request_ids(tmp_path):
    root = _replay(tmp_path / "calibration", 2.0)
    records = [json.loads(line) for line in (root / "manifest.jsonl").read_text().splitlines()]
    records[0]["request_id"] = "-1"
    records[1]["request_id"] = "7"
    # Add a new positive request while retaining the older recorder's stale
    # sequence_epoch=0. This is the schema emitted by already-running capture
    # servers when a new WebSocket session starts.
    extra = dict(records[1])
    extra["call_index"] = 2
    extra["request_id"] = "8"
    records.append(extra)
    (root / "manifest.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records))
    corpus = inspect_replay_corpus([root])
    assert corpus.conversations == 2
    assert [record["normalized_conversation_index"] for record in corpus.records] == [0, 0, 1]


def test_selection_manifest_whitelists_and_exhaustively_rejects_requests(tmp_path):
    root = _replay(tmp_path / "calibration", 2.0)
    records = [json.loads(line) for line in (root / "manifest.jsonl").read_text().splitlines()]
    records[0]["request_id"] = "accepted"
    records[1]["request_id"] = "rejected"
    (root / "manifest.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records))
    corpus = inspect_replay_corpus([root])
    manifest = {
        "schema": 1,
        "kind": "nano_gptq_corpus_selection",
        "calibration": [
            {
                "root_index": 0,
                "label": "first capture server",
                "include_request_ids": ["accepted"],
                "exclude_request_ids": {"rejected": "turn timeout"},
            }
        ],
    }
    selected = select_replay_corpus(corpus, manifest, split="calibration")
    assert selected.calls == 1
    assert selected.conversations == 1
    assert selected.records[0]["request_id"] == "accepted"
    incomplete = json.loads(json.dumps(manifest))
    incomplete["calibration"][0]["exclude_request_ids"] = {}
    with pytest.raises(ValueError, match="not exhaustive"):
        select_replay_corpus(corpus, incomplete, split="calibration")
    overlap = json.loads(json.dumps(manifest))
    overlap["calibration"][0]["include_request_ids"].append("rejected")
    with pytest.raises(ValueError, match="both accepts and rejects"):
        select_replay_corpus(corpus, overlap, split="calibration")


def test_corpora_must_be_disjoint_and_exact_public(tmp_path):
    calibration = inspect_replay_corpus([_replay(tmp_path / "calibration", 2.0)])
    evaluation = inspect_replay_corpus([_replay(tmp_path / "evaluation", 3.0)])
    assert_disjoint_corpora(calibration, evaluation)
    with pytest.raises(ValueError, match="roots overlap"):
        assert_disjoint_corpora(calibration, calibration)
    bad = _replay(tmp_path / "bad", 4.0, revision="old")
    with pytest.raises(ValueError, match="wrong replay revision"):
        inspect_replay_corpus([bad])


def test_current_nested_vllm_provenance_schema_is_normalized(tmp_path):
    root = _replay(tmp_path / "nested", 5.0)
    metadata = json.loads((root / "metadata.json").read_text())
    provenance = metadata["checkpoint_provenance"]
    provenance.pop("derived_from_ea")
    provenance.pop("nano_sha256")
    provenance["vllm_artifacts"] = {
        "derived_from_ea": False,
        "components": {"nano": {"sha256": "nano"}},
    }
    (root / "metadata.json").write_text(json.dumps(metadata))
    corpus = inspect_replay_corpus([root])
    assert corpus.provenance["derived_from_ea"] is False
    assert corpus.provenance["nano_sha256"] == "nano"


def test_small_gptq_is_deterministic_and_packable():
    torch.manual_seed(7)
    weight = torch.randn(8, 8)
    activations = torch.randn(32, 8)
    hessian = 2 * activations.t().matmul(activations) / activations.shape[0]
    left, left_metrics = gptq_quantize_matrix(
        weight, hessian, torch, bits=8, group_size=2, block_size=4
    )
    right, right_metrics = gptq_quantize_matrix(
        weight, hessian, torch, bits=8, group_size=2, block_size=4
    )
    assert torch.equal(left["codes"], right["codes"])
    assert torch.equal(left["scales"], right["scales"])
    assert left_metrics == right_metrics
    packed = pack_gptq_codes(left["codes"], left["scales"], torch, bits=8, group_size=2)
    assert tuple(packed["qweight"].shape) == (2, 8)
    assert tuple(packed["scales"].shape) == (4, 8)
    assert tuple(packed["qzeros"].shape) == (4, 2)
    assert tuple(packed["g_idx"].shape) == (8,)


def test_gptq_preserves_calibration_dead_columns_for_held_out_inputs():
    weight = torch.tensor(
        [
            [0.25, 0.50, -0.75, 1.00],
            [-0.40, 0.80, 0.60, -0.20],
            [0.15, -0.45, 0.90, 0.30],
            [-0.70, 0.35, -0.10, 0.55],
        ],
        dtype=torch.float32,
    )
    activations = torch.randn(32, 4)
    activations[:, 1] = 0
    hessian = 2 * activations.t().matmul(activations) / activations.shape[0]
    result, metrics = gptq_quantize_matrix(
        weight, hessian, torch, bits=8, group_size=2, block_size=2
    )
    assert metrics["dead_columns"] == 1
    assert metrics["dead_column_policy"] == "isolated_quantize_preserve_weight"
    assert torch.count_nonzero(result["codes"][:, 1]) == weight.shape[0]
    assert metrics["relative_rmse"] < 0.02


def test_byte_reproducibility_compares_every_file(tmp_path):
    left, right = tmp_path / "left", tmp_path / "right"
    left.mkdir()
    right.mkdir()
    (left / "model").write_bytes(b"same")
    (right / "model").write_bytes(b"same")
    assert assert_byte_identical(left, right)["byte_identical"] is True
    (right / "model").write_bytes(b"different")
    with pytest.raises(ValueError, match="not byte-identical"):
        assert_byte_identical(left, right)
