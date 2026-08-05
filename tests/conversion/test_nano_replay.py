from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nano_component_gate import (  # noqa: E402
    DIAGNOSTIC_CUSTOM_OUTPUTS,
    control_margin,
    prepare_diagnostic_candidate,
)
from nano_replay import (  # noqa: E402
    generation_result_payload,
    install_public_nano_capture,
    outputs_contain_full_text_logits,
    read_manifest,
)


class NanoReplayTest(unittest.TestCase):
    def test_diagnostic_candidate_restores_logits_without_copying_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            weights = source / "model.safetensors"
            weights.write_bytes(b"weights")
            (source / "config.json").write_text(
                json.dumps(
                    {
                        "custom_outputs": ["function_tokens", "function_logits"],
                        "voicechat_skip_custom_text_logits": {
                            "contract": "temperature=0, top_p=1, repetition_penalty=1",
                            "enabled": True,
                            "source_config_sha256": "a" * 64,
                        },
                    }
                )
            )

            diagnostic, provenance = prepare_diagnostic_candidate(source, root / "gate")

            config = json.loads((diagnostic / "config.json").read_text())
            self.assertEqual(config["custom_outputs"], DIAGNOSTIC_CUSTOM_OUTPUTS)
            self.assertNotIn("voicechat_skip_custom_text_logits", config)
            self.assertTrue((diagnostic / "model.safetensors").is_symlink())
            self.assertEqual((diagnostic / "model.safetensors").read_bytes(), b"weights")
            self.assertEqual(provenance["mode"], "temporary-text-logit-overlay")
            self.assertFalse(provenance["weights_changed"])

    def test_diagnostic_candidate_rejects_uncontracted_missing_logits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            (source / "config.json").write_text(
                json.dumps({"custom_outputs": ["function_tokens", "function_logits"]})
            )
            with self.assertRaisesRegex(ValueError, "without the qualified"):
                prepare_diagnostic_candidate(source, root / "gate")

    def test_bos_pad_control_margin_is_explicit(self):
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("conversion tensor tests run in the public runtime container")

        logits = torch.tensor([[0.0, 4.0, 1.0, 2.5]])
        self.assertEqual(control_margin(logits, bos_token_id=1, pad_token_id=3), 1.5)

    def test_generation_result_payload_retains_scalar_controls_without_torch(self):
        result = SimpleNamespace(
            token_id=17,
            is_finished=False,
            finish_reason=None,
            total_tokens=4,
            custom_outputs={"label": "teacher"},
        )
        self.assertEqual(
            generation_result_payload(result),
            {
                "token_id": 17,
                "is_finished": False,
                "finish_reason": None,
                "total_tokens": 4,
                "custom_outputs": {"label": "teacher"},
            },
        )

    def test_capture_metadata_detects_omitted_text_logits(self):
        self.assertTrue(
            outputs_contain_full_text_logits(
                {
                    "engine_results": [
                        {"custom_outputs": {"text_logits": [1], "function_tokens": [12]}}
                    ]
                }
            )
        )
        self.assertFalse(
            outputs_contain_full_text_logits(
                {"engine_results": [{"custom_outputs": {"function_tokens": [12]}}]}
            )
        )

    def test_manifest_requires_contiguous_causal_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.jsonl").write_text(
                json.dumps({"call_index": 0}) + "\n" + json.dumps({"call_index": 2}) + "\n"
            )
            with self.assertRaisesRegex(ValueError, "not contiguous"):
                read_manifest(root)

    def test_capture_is_noop_without_explicit_environment(self):
        import os

        previous = os.environ.pop("VOICECHAT_NANO_REPLAY_CAPTURE_DIR", None)
        try:
            self.assertIsNone(install_public_nano_capture(SimpleNamespace()))
        finally:
            if previous is not None:
                os.environ["VOICECHAT_NANO_REPLAY_CAPTURE_DIR"] = previous


if __name__ == "__main__":
    unittest.main()
