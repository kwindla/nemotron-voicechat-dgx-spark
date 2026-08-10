from __future__ import annotations

import importlib.util
from copy import deepcopy
from pathlib import Path

from nemotron_voicechat_runtime.provenance import RUNTIME_PROVENANCE_REQUIRED_SOURCES
from nemotron_voicechat_runtime.semantic_corpus import load_semantic_corpus

CORPUS = Path("tools/qualification/direct_text_semantic_corpus.json")


def load_module():
    path = Path("tools/qualification/evaluate_direct_text_semantics.py")
    spec = importlib.util.spec_from_file_location("evaluate_direct_text_semantics", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def provenance() -> dict:
    return {
        "runtime_image_id": f"sha256:{'a' * 64}",
        "source_sha256": {name: "b" * 64 for name in RUNTIME_PROVENANCE_REQUIRED_SOURCES},
    }


def satisfying_output(case: dict) -> tuple[str, list[dict]]:
    predicate = case["predicate"]
    response = " ".join(
        [group[0] for group in predicate.get("required_any", [])]
        + list(predicate.get("required_literal", []))
    )
    expected_tool = predicate.get("tool")
    calls = [expected_tool] if expected_tool else []
    return response or "Done.", calls


def satisfying_carrier(case: dict) -> str:
    carrier = case["carrier"]
    return " ".join(
        [group[0] for group in carrier.get("required_any", [])]
        + list(carrier.get("required_literal", []))
    )


def reports() -> tuple[dict, dict, dict]:
    corpus = load_semantic_corpus(CORPUS)
    digest = __import__("hashlib").sha256(CORPUS.read_bytes()).hexdigest()
    raw_cases = []
    for case in corpus["cases"]:
        text, calls = satisfying_output(case)
        raw_cases.append(
            {
                "case_id": case["id"],
                "assistant_text": text,
                "input_transcript": satisfying_carrier(case),
                "tool_calls": calls,
                "structural_passed": True,
            }
        )
    pocket = {
        "corpus_sha256": digest,
        "runtime_provenance": provenance(),
        "evidence_complete": True,
        "cases": raw_cases,
    }

    def direct(precision: str) -> dict:
        positions = [
            {
                "model": {
                    "source_dtype": "torch.bfloat16",
                    "source_shape": [1, 1, 4480],
                    "effective_output_forced_pad": True,
                },
                "effective_token_id": 12,
                "effective_subword_mask": True,
                "decoded_audio_samples_discarded": 1764,
                "nano_position_before": 4,
                "nano_position_after": 5,
                "eartts_position_before": 4,
                "eartts_position_after": 5,
                "epoch_offset_before": 0,
                "epoch_offset_after": 0,
                "perception_cache_unchanged": True,
                "rnnt_state_unchanged": True,
                "function_state_unchanged": True,
                "previous_feedback": {"agent": 12, "asr": 12, "function": 12},
            }
        ]
        direct_cases = deepcopy(raw_cases)
        for case in direct_cases:
            case["direct_positions"] = deepcopy(positions)
        return {
            "precision": precision,
            "corpus_sha256": digest,
            "runtime_provenance": provenance(),
            "pad_token_id": 12,
            "lanes": [
                {"token_form": token_form, "cases": deepcopy(direct_cases)}
                for token_form in ("text", "user_bos_text_user_eos")
            ],
        }

    return pocket, direct("fp32"), direct("w8")


def test_comparison_selects_plain_text_when_all_gates_pass() -> None:
    module = load_module()
    pocket, fp32, w8 = reports()

    report = module.evaluate(corpus_path=CORPUS, pocket=pocket, fp32=fp32, w8=w8)

    assert report["passed"]
    assert report["qualified_token_forms"] == ["text", "user_bos_text_user_eos"]
    assert report["selected_token_form"] == "text"
    assert all(
        lane["direct_intent_predicate_rate"] == 1.0
        for lane in report["lanes"].values()
    )


def test_comparison_regrades_raw_tool_calls_and_rejects_wrong_argument() -> None:
    module = load_module()
    pocket, fp32, w8 = reports()
    for report in (fp32, w8):
        report["lanes"][0]["cases"][12]["tool_calls"][0]["arguments"]["weight"] = 82

    result = module.evaluate(corpus_path=CORPUS, pocket=pocket, fp32=fp32, w8=w8)

    assert not result["lanes"]["fp32/text"]["tool_name_and_arguments_passed"]
    assert not result["token_forms"]["text"]["qualified"]
    assert result["token_forms"]["user_bos_text_user_eos"]["qualified"]


def test_comparison_rejects_two_noncritical_w8_regressions() -> None:
    module = load_module()
    pocket, fp32, w8 = reports()
    w8["lanes"][0]["cases"][0]["assistant_text"] = "wrong"
    w8["lanes"][0]["cases"][1]["assistant_text"] = "wrong"

    result = module.evaluate(corpus_path=CORPUS, pocket=pocket, fp32=fp32, w8=w8)

    token_form = result["token_forms"]["text"]
    assert token_form["noncritical_w8_regressions"] == ["c001", "c002"]
    assert not token_form["w8_regression_gate_passed"]
    assert not token_form["qualified"]


def test_comparison_rejects_malformed_or_mixed_provenance() -> None:
    module = load_module()
    pocket, fp32, w8 = reports()
    w8["runtime_provenance"]["runtime_image_id"] = "mutable:latest"

    malformed = module.evaluate(corpus_path=CORPUS, pocket=pocket, fp32=fp32, w8=w8)
    assert not malformed["runtime_provenance_passed"]
    assert not malformed["passed"]

    pocket, fp32, w8 = reports()
    w8["runtime_provenance"]["source_sha256"]["src/nemotron_voicechat_runtime/server.py"] = "c" * 64
    mixed = module.evaluate(corpus_path=CORPUS, pocket=pocket, fp32=fp32, w8=w8)
    assert not mixed["source_provenance_consistent"]
    assert not mixed["passed"]


def test_comparison_rederives_hidden_transaction_checks() -> None:
    module = load_module()
    pocket, fp32, w8 = reports()
    fp32["lanes"][0]["cases"][0]["direct_positions"][0]["nano_position_after"] = 7

    result = module.evaluate(corpus_path=CORPUS, pocket=pocket, fp32=fp32, w8=w8)

    assert not result["lanes"]["fp32/text"]["hidden_transaction_rederived_passed"]
    assert not result["lanes"]["fp32/text"]["structural_passed"]
    assert not result["token_forms"]["text"]["qualified"]


def test_pocket_failure_remains_visible_but_does_not_veto_clean_direct_lanes() -> None:
    module = load_module()
    pocket, fp32, w8 = reports()
    c009 = pocket["cases"][8]
    assert "emergency" in c009["assistant_text"]
    c009["response_status"] = "failed"
    c009["response_reason"] = "function_call_loop_limit"

    result = module.evaluate(corpus_path=CORPUS, pocket=pocket, fp32=fp32, w8=w8)

    grade = result["pocket"]["cases"]["c009"]
    assert grade["text_passed"]
    assert not grade["response_terminal_passed"]
    assert not grade["passed"]
    assert not result["pocket"]["safety_critical_passed"]
    assert result["pocket"]["genuine_semantic_failures"] == ["c009"]
    assert not result["pocket"]["reference_passed"]
    assert result["passed"]


def test_carrier_loss_is_inconclusive_and_never_regraded_as_pocket_success() -> None:
    module = load_module()
    pocket, fp32, w8 = reports()
    pocket["cases"][2]["input_transcript"] = "AX seventeen B"

    result = module.evaluate(corpus_path=CORPUS, pocket=pocket, fp32=fp32, w8=w8)

    assert result["pocket"]["carrier"]["c003"]["classification"] == (
        "carrier_inconclusive"
    )
    assert result["pocket"]["carrier_inconclusive_cases"] == ["c003"]
    assert result["pocket"]["carrier_preserved_count"] == 19
    assert result["passed"]


def test_incomplete_pocket_evidence_still_blocks_direct_qualification() -> None:
    module = load_module()
    pocket, fp32, w8 = reports()
    pocket["evidence_complete"] = False

    result = module.evaluate(corpus_path=CORPUS, pocket=pocket, fp32=fp32, w8=w8)

    assert not result["pocket"]["evidence_complete"]
    assert not result["passed"]


def test_pocket_reference_cannot_pass_vacuously_without_tool_carriers() -> None:
    module = load_module()
    pocket, fp32, w8 = reports()
    for raw_case in pocket["cases"][12:]:
        raw_case["input_transcript"] = "carrier lost this request"

    result = module.evaluate(corpus_path=CORPUS, pocket=pocket, fp32=fp32, w8=w8)

    assert not result["pocket"]["tool_name_and_arguments_passed"]
    assert not result["pocket"]["reference_passed"]
    assert result["passed"]
