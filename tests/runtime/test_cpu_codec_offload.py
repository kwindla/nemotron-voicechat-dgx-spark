import unittest

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="runtime image supplies PyTorch")

from nemotron_voicechat_runtime.cpu_codec_offload import (
    LinearPointwiseConv1d,
    OrderedAsyncCodecDecode,
    OrderedProcessCodecDecode,
    SlicedDepthwiseConv1d,
)


class CpuCodecOffloadTests(unittest.TestCase):
    def test_linear_pointwise_matches_conv1d(self):
        torch.manual_seed(23)
        source = torch.nn.Conv1d(17, 29, kernel_size=1)
        candidate = LinearPointwiseConv1d.create(torch, source)
        value = torch.randn(2, 17, 11)
        expected = source(value)
        actual = candidate(value)
        self.assertTrue(torch.allclose(actual, expected, rtol=1e-5, atol=1e-6))

    def test_sliced_depthwise_matches_grouped_conv(self):
        torch.manual_seed(17)
        source = torch.nn.Conv1d(32, 32, kernel_size=7, groups=32)
        candidate = SlicedDepthwiseConv1d.create(torch, source)
        value = torch.randn(1, 32, 69)
        expected = source(value)
        actual = candidate(value)
        self.assertTrue(torch.allclose(actual, expected, rtol=1e-5, atol=1e-6))

    def test_ordered_async_codec_restores_exact_frame_order(self):
        class FakeCodec:
            pass

        codec = FakeCodec()

        def decode(codes, lengths, *args, **kwargs):
            value = codes[:, :, :1].to(dtype=torch.float32).transpose(1, 2)
            return value.repeat_interleave(4, dim=-1), lengths * 4

        codec._ea_cpu_original_decode = decode
        pipeline = OrderedAsyncCodecDecode(codec, samples_per_frame=4, torch_module=torch)
        try:
            returned = []
            for value in (1, 2, 3):
                codes = torch.tensor([[[value]]])
                audio, _ = pipeline.decode(codes, torch.tensor([1]))
                returned.append(audio)
            buffered = torch.cat(returned, dim=-1)
            actual = pipeline.finalize_audio(buffered, output_device=torch.device("cpu"))
        finally:
            pipeline.close()

        expected = torch.tensor([[[1.0] * 4 + [2.0] * 4 + [3.0] * 4]])
        self.assertTrue(torch.equal(actual, expected))

    def test_ordered_async_codec_finalizes_pipeline_2d_buffer(self):
        class FakeCodec:
            pass

        codec = FakeCodec()
        codec._ea_cpu_original_decode = lambda codes, lengths: (
            torch.full((1, 1, 4), float(codes.item())),
            lengths * 4,
        )
        pipeline = OrderedAsyncCodecDecode(codec, samples_per_frame=4, torch_module=torch)
        try:
            first, _ = pipeline.decode(torch.tensor([[[7]]]), torch.tensor([1]))
            actual = pipeline.finalize_audio(
                first.reshape(1, -1), output_device=torch.device("cpu")
            )
        finally:
            pipeline.close()
        self.assertEqual(actual.shape, (1, 4))
        self.assertTrue(torch.equal(actual, torch.full((1, 4), 7.0)))

    def test_persistent_process_codec_resets_cache_between_sessions(self):
        class FakeConnection:
            def __init__(self):
                self.requests = []
                self.responses = []

            def send(self, request):
                self.requests.append(request)
                if request["op"] == "decode":
                    value = float(request["codes"].item())
                    self.responses.append(
                        {
                            "status": "ok",
                            "sequence": request["sequence"],
                            "audio": np.full((1, 1, 4), value, dtype=np.float32),
                            "lengths": np.array([4], dtype=np.int64),
                        }
                    )

            def recv(self):
                return self.responses.pop(0)

        connection = FakeConnection()
        pipeline = OrderedProcessCodecDecode.__new__(OrderedProcessCodecDecode)
        pipeline.torch = torch
        pipeline.samples_per_frame = 4
        pipeline._connection = connection
        pipeline._pending_sequence = None
        pipeline._pending_device = None
        pipeline._next_sequence = 0
        pipeline._cache_identity = None
        pipeline._force_reset_cache = True
        pipeline._first_placeholder_samples = 0
        pipeline._session_submitted = 0
        pipeline.submitted = 0
        pipeline._closed = False

        cache = object()
        first, _ = pipeline.decode(torch.tensor([[[7]]]), torch.tensor([1]), cache=cache)
        first = pipeline.finalize_audio(first, output_device=torch.device("cpu"))
        self.assertTrue(torch.equal(first, torch.full((1, 1, 4), 7.0)))
        self.assertEqual(pipeline.finish_session(), 1)

        second, _ = pipeline.decode(torch.tensor([[[8]]]), torch.tensor([1]), cache=cache)
        second = pipeline.finalize_audio(second, output_device=torch.device("cpu"))
        self.assertTrue(torch.equal(second, torch.full((1, 1, 4), 8.0)))
        self.assertEqual(pipeline.finish_session(), 1)
        decode_requests = [request for request in connection.requests if request["op"] == "decode"]
        self.assertEqual([request["reset_cache"] for request in decode_requests], [True, True])
        self.assertEqual(pipeline.submitted, 2)


if __name__ == "__main__":
    unittest.main()
