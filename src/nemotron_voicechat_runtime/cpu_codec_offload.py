"""Experimental CPU codec offload for DGX Spark.

The ARM64 PyTorch build falls back to thousands of scalar grouped-convolution
calls for the codec's depthwise Conv1d layers.  The sliced expression below is
mathematically equivalent but vectorizes over channels and time.
"""

from __future__ import annotations

import atexit
import os
import subprocess
import sys
import tempfile
from concurrent.futures import Future, ThreadPoolExecutor
from multiprocessing.connection import Listener
from pathlib import Path

X925_CPUS = (5, 6, 7, 8, 9, 15, 16, 17, 18, 19)


def selected_x925_cpus() -> tuple[int, ...]:
    raw_cpus = os.environ.get("EA_CPU_CODEC_CORES", "5,6")
    try:
        configured = tuple(int(value) for value in raw_cpus.split(","))
    except ValueError as error:
        raise ValueError("EA_CPU_CODEC_CORES must be a comma-separated CPU list") from error
    raw = os.environ.get("EA_CPU_CODEC_THREADS", str(len(configured)))
    try:
        count = int(raw)
    except ValueError as error:
        raise ValueError("EA_CPU_CODEC_THREADS must be an integer") from error
    if not 1 <= count <= len(configured):
        raise ValueError(
            f"EA_CPU_CODEC_THREADS must be between 1 and {len(configured)} configured cores"
        )
    cpus = configured[:count]
    if len(set(cpus)) != len(cpus) or not set(cpus) <= set(X925_CPUS):
        raise ValueError("EA_CPU_CODEC_CORES must name unique measured X925 cores")
    return cpus


def configure_cpu_runtime() -> None:
    """Set defaults before the first substantial CPU Torch operation."""
    cpus = selected_x925_cpus()
    affinity = " ".join(str(cpu) for cpu in cpus)
    os.environ.setdefault("OMP_NUM_THREADS", str(len(cpus)))
    os.environ.setdefault("NVPL_NUM_THREADS", str(len(cpus)))
    os.environ.setdefault("OMP_DYNAMIC", "FALSE")
    os.environ.setdefault("OMP_PROC_BIND", "TRUE")
    os.environ.setdefault("OMP_WAIT_POLICY", "ACTIVE")
    os.environ.setdefault("GOMP_CPU_AFFINITY", affinity)


class SlicedDepthwiseConv1d:
    """Factory namespace kept importable without importing Torch eagerly."""

    @staticmethod
    def create(torch, source):
        class _SlicedDepthwiseConv1d(torch.nn.Module):
            def __init__(self):
                super().__init__()
                if source.stride != (1,) or source.padding != (0,) or source.dilation != (1,):
                    raise ValueError(
                        "CPU codec depthwise path requires stride=1/padding=0/dilation=1"
                    )
                if source.groups != source.in_channels or source.out_channels != source.in_channels:
                    raise ValueError("CPU codec depthwise path requires one filter per channel")
                self.weight = source.weight
                self.bias = source.bias

            def forward(self, value):
                kernel = self.weight.shape[-1]
                output_length = value.shape[-1] - kernel + 1
                result = value[..., :output_length] * self.weight[:, 0, 0].view(1, -1, 1)
                for offset in range(1, kernel):
                    result = result + value[..., offset : offset + output_length] * self.weight[
                        :, 0, offset
                    ].view(1, -1, 1)
                if self.bias is not None:
                    result = result + self.bias.view(1, -1, 1)
                return result

        return _SlicedDepthwiseConv1d()


class LinearPointwiseConv1d:
    """Exact FP32 1x1 Conv1d expressed through the optimized linear path."""

    @staticmethod
    def create(torch, source):
        class _LinearPointwiseConv1d(torch.nn.Module):
            def __init__(self):
                super().__init__()
                if (
                    source.kernel_size != (1,)
                    or source.stride != (1,)
                    or source.padding != (0,)
                    or source.dilation != (1,)
                    or source.groups != 1
                ):
                    raise ValueError("CPU codec pointwise path requires an ordinary 1x1 Conv1d")
                self.weight = source.weight
                self.bias = source.bias

            def forward(self, value):
                result = torch.nn.functional.linear(
                    value.transpose(1, 2), self.weight[:, :, 0], self.bias
                )
                return result.transpose(1, 2)

        return _LinearPointwiseConv1d()


def replace_pointwise_convs(torch, model) -> int:
    replacements = 0
    for module in model.modules():
        for child_name, candidate in tuple(module.named_children()):
            if (
                isinstance(candidate, torch.nn.Conv1d)
                and candidate.kernel_size == (1,)
                and candidate.stride == (1,)
                and candidate.padding == (0,)
                and candidate.dilation == (1,)
                and candidate.groups == 1
            ):
                setattr(module, child_name, LinearPointwiseConv1d.create(torch, candidate))
                replacements += 1
    return replacements


def install_cpu_codec(model, *, checkpoint_root=None, torch_module=None) -> dict:
    """Move the decoder to CPU and bridge its tiny streaming I/O tensors."""
    configure_cpu_runtime()
    if torch_module is None:
        import torch as torch_module

    torch = torch_module
    cpus = selected_x925_cpus()
    torch.set_num_threads(len(cpus))
    codec = model.tts_model.audio_codec
    codec.to(device="cpu", dtype=torch.float32)
    codec.eval()
    replacements = 0
    for module in codec.modules():
        candidate = getattr(module, "dwconv", None)
        if (
            isinstance(candidate, torch.nn.Conv1d)
            and candidate.groups == candidate.in_channels
            and candidate.out_channels == candidate.in_channels
        ):
            module.dwconv = SlicedDepthwiseConv1d.create(torch, candidate)
            replacements += 1
    if replacements != 18:
        raise RuntimeError(f"expected 18 codec depthwise convolutions, replaced {replacements}")
    pointwise_replacements = 0
    if os.environ.get("EA_CPU_CODEC_POINTWISE_LINEAR", "0") == "1":
        pointwise_replacements = replace_pointwise_convs(torch, codec)
        if pointwise_replacements != 38:
            raise RuntimeError(
                f"expected 38 codec pointwise convolutions, replaced {pointwise_replacements}"
            )

    original_decode = codec.decode
    # Keep the CPU-native entry point available to the ordered async bridge.
    # The ordinary bridge below preserves the upstream synchronous contract.
    object.__setattr__(codec, "_ea_cpu_original_decode", original_decode)
    if checkpoint_root is not None:
        object.__setattr__(codec, "_ea_checkpoint_root", str(checkpoint_root))

    def bridged_decode(codes, lengths, *args, **kwargs):
        output_device = codes.device
        cpu_codes = codes.to(device="cpu", non_blocking=False)
        cpu_lengths = lengths.to(device="cpu", non_blocking=False)
        audio, output_lengths = original_decode(cpu_codes, cpu_lengths, *args, **kwargs)
        # The upstream accumulator also creates CUDA silence chunks. Keep its
        # device contract; this transfer is only 1,764 FP32 samples per slice.
        return (
            audio.to(device=output_device, non_blocking=False),
            output_lengths.to(device=output_device, non_blocking=False),
        )

    object.__setattr__(codec, "decode", bridged_decode)
    object.__setattr__(codec, "_ea_cpu_bridged_decode", bridged_decode)
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in codec.parameters()
    )
    return {
        "depthwise_replacements": replacements,
        "pointwise_replacements": pointwise_replacements,
        "parameter_bytes": parameter_bytes,
        "threads": len(cpus),
        "cpus": list(cpus),
    }


def prepare_process_codec(model, *, checkpoint_root) -> dict:
    """Mark the resident codec for replacement without copying its weights.

    The isolated worker loads its own CPU copy. Keeping the parent codec on its
    existing device avoids the expensive in-process GPU-to-CPU migration and
    preserves a synchronous fallback for later streams.
    """
    codec = model.tts_model.audio_codec
    object.__setattr__(codec, "_ea_checkpoint_root", str(checkpoint_root))
    object.__setattr__(codec, "_ea_cpu_bridged_decode", codec.decode)
    return {
        "mode": "process",
        "checkpoint_root": str(checkpoint_root),
        "threads": len(selected_x925_cpus()),
        "cpus": list(selected_x925_cpus()),
    }


class OrderedAsyncCodecDecode:
    """Pipeline one causal CPU decode behind the GPU inference loop.

    Codec cache mutation is stateful, so jobs execute on one FIFO worker.  The
    main thread copies the tiny token input to CPU before submission and copies
    only the previous frame's waveform back to the caller.  This keeps CUDA out
    of the worker and bounds the queue at one frame.
    """

    def __init__(self, codec, *, samples_per_frame: int, torch_module=None):
        if torch_module is None:
            import torch as torch_module

        self.torch = torch_module
        self.codec = codec
        self.samples_per_frame = samples_per_frame
        self._cpu_decode = getattr(codec, "_ea_cpu_original_decode", None)
        if self._cpu_decode is None:
            raise RuntimeError("async codec requires install_cpu_codec first")
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ea-codec")
        self._pending: Future | None = None
        self._pending_device = None
        self._first_placeholder_samples = 0
        self.submitted = 0

    def _run(self, codes, lengths, args, kwargs):
        # sched_setaffinity with pid=0 applies to this worker thread on Linux.
        try:
            os.sched_setaffinity(0, selected_x925_cpus())
        except (AttributeError, OSError):
            pass
        return self._cpu_decode(codes, lengths, *args, **kwargs)

    def decode(self, codes, lengths, *args, **kwargs):
        torch = self.torch
        output_device = codes.device
        # These tensors are tiny (one 80 ms code frame and one length scalar).
        # Materialize them before the worker starts so it never touches CUDA.
        cpu_codes = codes.detach().to(device="cpu", non_blocking=False).clone()
        cpu_lengths = lengths.detach().to(device="cpu", non_blocking=False).clone()

        previous = self._pending
        previous_device = self._pending_device
        self._pending = self._executor.submit(self._run, cpu_codes, cpu_lengths, args, dict(kwargs))
        self._pending_device = output_device
        self.submitted += 1

        if previous is None:
            samples = self.samples_per_frame * int(cpu_codes.shape[1])
            self._first_placeholder_samples = samples
            audio = torch.zeros(
                (int(cpu_codes.shape[0]), 1, samples),
                dtype=torch.float32,
                device=output_device,
            )
            output_lengths = torch.full(
                (int(cpu_codes.shape[0]),),
                samples,
                dtype=torch.long,
                device=output_device,
            )
            return audio, output_lengths

        audio, output_lengths = previous.result()
        return (
            audio.to(device=previous_device, non_blocking=False),
            output_lengths.to(device=previous_device, non_blocking=False),
        )

    def finalize_audio(self, buffered_audio, *, output_device):
        """Replace the initial placeholder with the final pending decode."""
        if self._pending is None:
            return buffered_audio
        audio, _ = self._pending.result()
        tail = audio.to(device=output_device, non_blocking=False)
        self._pending = None
        if buffered_audio is None:
            return tail
        if self._first_placeholder_samples <= 0:
            raise RuntimeError("async codec produced no initial placeholder")
        if buffered_audio.dim() == 2 and tail.dim() == 3:
            tail = tail.reshape(tail.shape[0], -1)
        elif buffered_audio.dim() == 3 and tail.dim() == 2:
            tail = tail.unsqueeze(1)
        if buffered_audio.dim() != tail.dim():
            raise RuntimeError(
                "async codec buffered/tail rank mismatch: "
                f"{buffered_audio.shape} versus {tail.shape}"
            )
        return self.torch.cat(
            [buffered_audio[..., self._first_placeholder_samples :], tail], dim=-1
        )

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=False)


class OrderedProcessCodecDecode:
    """One-frame codec pipeline backed by an isolated CPU process."""

    def __init__(self, codec, *, samples_per_frame: int, torch_module=None):
        if torch_module is None:
            import torch as torch_module

        self.torch = torch_module
        self.samples_per_frame = samples_per_frame
        checkpoint_root = getattr(codec, "_ea_checkpoint_root", None)
        if checkpoint_root is None:
            raise RuntimeError("process codec requires a checkpoint root")
        self._tempdir = tempfile.TemporaryDirectory(prefix="ea-codec-")
        socket_path = str(Path(self._tempdir.name) / "worker.sock")
        self._listener = Listener(socket_path, family="AF_UNIX")
        worker = Path(__file__).with_name("cpu_codec_worker.py")
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join(sys.path)
        worker_command = [
            sys.executable,
            str(worker),
            "--socket",
            socket_path,
            "--checkpoint-root",
            checkpoint_root,
            "--threads",
            str(len(selected_x925_cpus())),
            "--cpus",
            ",".join(str(cpu) for cpu in selected_x925_cpus()),
        ]
        if os.environ.get("EA_CPU_CODEC_POINTWISE_LINEAR", "0") == "1":
            worker_command.append("--pointwise-linear")
        self._process = subprocess.Popen(
            worker_command,
            env=environment,
        )
        self._connection = self._listener.accept()
        ready = self._connection.recv()
        if ready.get("status") != "ready":
            raise RuntimeError(f"CPU codec worker failed: {ready}")
        self._pending_sequence = None
        self._pending_device = None
        self._next_sequence = 0
        self._cache_identity = None
        self._force_reset_cache = True
        self._first_placeholder_samples = 0
        self._session_submitted = 0
        self.submitted = 0
        self._closed = False
        self.worker_details = ready
        atexit.register(self.close)

    def warmup(self, codes, lengths) -> dict:
        """Pay the worker's first-decode cost before the service becomes ready."""
        if self._pending_sequence is not None:
            raise RuntimeError("cannot warm the CPU codec with pending decode work")
        self._connection.send(
            {
                "op": "warmup",
                "codes": codes.detach().to(device="cpu", non_blocking=False).numpy(),
                "lengths": lengths.detach().to(device="cpu", non_blocking=False).numpy(),
            }
        )
        response = self._connection.recv()
        if response.get("status") != "ok":
            raise RuntimeError(f"CPU codec worker warmup failed: {response}")
        self._force_reset_cache = True
        self._cache_identity = None
        return response

    def decode(self, codes, lengths, *args, **kwargs):
        torch = self.torch
        output_device = codes.device
        cache = kwargs.get("cache")
        cache_identity = id(cache) if cache is not None else None
        reset_cache = self._force_reset_cache or (
            self._cache_identity is not None and cache_identity != self._cache_identity
        )
        self._force_reset_cache = False
        self._cache_identity = cache_identity
        sequence = self._next_sequence
        self._next_sequence += 1
        self._connection.send(
            {
                "op": "decode",
                "sequence": sequence,
                "codes": codes.detach().to(device="cpu", non_blocking=False).numpy(),
                "lengths": lengths.detach().to(device="cpu", non_blocking=False).numpy(),
                "reset_cache": reset_cache,
            }
        )
        previous_sequence = self._pending_sequence
        previous_device = self._pending_device
        self._pending_sequence = sequence
        self._pending_device = output_device
        self._session_submitted += 1
        self.submitted += 1

        if previous_sequence is None:
            samples = self.samples_per_frame * int(codes.shape[1])
            self._first_placeholder_samples = samples
            return (
                torch.zeros(
                    (int(codes.shape[0]), 1, samples),
                    dtype=torch.float32,
                    device=output_device,
                ),
                torch.full(
                    (int(codes.shape[0]),),
                    samples,
                    dtype=torch.long,
                    device=output_device,
                ),
            )
        return self._receive(previous_sequence, previous_device)

    def _receive(self, expected_sequence, output_device):
        response = self._connection.recv()
        if response.get("status") != "ok":
            raise RuntimeError(f"CPU codec worker decode failed: {response}")
        if response["sequence"] != expected_sequence:
            raise RuntimeError(
                f"CPU codec worker sequence mismatch: {response['sequence']} != {expected_sequence}"
            )
        return (
            self.torch.from_numpy(response["audio"]).to(output_device),
            self.torch.from_numpy(response["lengths"]).to(output_device),
        )

    def finalize_audio(self, buffered_audio, *, output_device):
        if self._pending_sequence is None:
            return buffered_audio
        tail, _ = self._receive(self._pending_sequence, output_device)
        self._pending_sequence = None
        if buffered_audio.dim() == 2 and tail.dim() == 3:
            tail = tail.reshape(tail.shape[0], -1)
        elif buffered_audio.dim() == 3 and tail.dim() == 2:
            tail = tail.unsqueeze(1)
        return self.torch.cat(
            [buffered_audio[..., self._first_placeholder_samples :], tail], dim=-1
        )

    def finish_session(self) -> int:
        """Reset stream-local bookkeeping while retaining the warm worker."""
        if self._pending_sequence is not None:
            raise RuntimeError("cannot finish CPU codec session with pending work")
        submitted = self._session_submitted
        self._session_submitted = 0
        self._pending_device = None
        self._cache_identity = None
        self._force_reset_cache = True
        self._first_placeholder_samples = 0
        return submitted

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._connection.send({"op": "shutdown"})
            self._connection.close()
        except (BrokenPipeError, EOFError, OSError):
            pass
        self._listener.close()
        try:
            self._process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._process.terminate()
            self._process.wait(timeout=10)
        self._tempdir.cleanup()
