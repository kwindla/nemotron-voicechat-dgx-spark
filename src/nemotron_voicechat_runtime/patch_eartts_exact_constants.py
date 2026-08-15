#!/usr/bin/env python3
"""Hoist bitwise-exact EarTTS constants in two independently gated stages."""

# The long embedded lines below are byte-sensitive source-patch anchors.
# ruff: noqa: E501

from __future__ import annotations

import argparse
import hashlib
import py_compile
import sysconfig
from pathlib import Path

QUALIFIED_EARTTS_SHA256 = (
    "e86ff2197e5b086ae1c6b3e91c1df778ae6ade8a8386a3e38f84495cfd3e9f3a"
)
CHANGE_A_MARKER = "voicechat_eartts_exact_padded_rvq_v1"
CHANGE_B_MARKER = "voicechat_eartts_exact_frozen_constants_v1"


def replace_once(source: str, anchor: str, replacement: str, label: str) -> str:
    count = source.count(anchor)
    if count != 1:
        raise RuntimeError(f"{label}: expected one anchor, found {count}")
    return source.replace(anchor, replacement, 1)


def apply_change_a(source: str) -> str:
    if CHANGE_A_MARKER in source:
        return source

    source = replace_once(
        source,
        """  self.rvq_embs = nn.Parameter(torch.empty(
   self.config.num_quantizers,
   self.config.codebook_size,
   self.config.latent_size
  ))
""",
        f"""  self.rvq_embs = nn.Parameter(torch.empty(
   self.config.num_quantizers,
   self.config.codebook_size,
   self.config.latent_size
  ))
  # {CHANGE_A_MARKER}. This non-persistent buffer is valid only for the
  # exact rvq_embs bytes, device, and dtype from the most recent completed
  # weight load (or _apply transition). Any future in-place weight mutation
  # must call _refresh_exact_constants before inference.
  self.register_buffer(\"_rvq_embs_padded\", None, persistent=False)
  self._exact_constants_ready = False
""",
        "Change A sampler buffer registration",
    )
    source = replace_once(
        source,
        """ def _depthsum_embedding(self, code: torch.Tensor) -> torch.Tensor:
""",
        f""" def _invalidate_exact_constants(self):
  self._rvq_embs_padded = None
  self._exact_constants_ready = False

 def _refresh_exact_constants(self):
  # Construct the serving buffer only after final weight load/device/dtype.
  # Keep the original dynamic pad solely as a startup byte oracle, then drop it.
  dynamic = nn.functional.pad(self.rvq_embs.detach(), [0, 0, 0, 1])
  padded = torch.empty(
   (self.num_quantizers, self.codebook_size + 1, self.config.latent_size),
   device=self.rvq_embs.device,
   dtype=self.rvq_embs.dtype,
  )
  padded[:, :self.codebook_size].copy_(self.rvq_embs.detach())
  padded[:, self.codebook_size].zero_()
  if not torch.equal(padded.view(torch.uint8), dynamic.view(torch.uint8)):
   raise RuntimeError(\"{CHANGE_A_MARKER}: pre-padded RVQ byte check failed\")
  self._rvq_embs_padded = padded
  self._exact_constants_ready = True

 def _apply(self, fn, recurse=True):
  # nn.Module device/dtype transitions invalidate derived bytes. Rebuild only
  # if a completed weight load had made the cache live; constructor-time moves
  # must not derive constants from uninitialized parameters.
  was_ready = self._exact_constants_ready
  self._invalidate_exact_constants()
  result = super()._apply(fn, recurse=recurse)
  if was_ready:
   self._refresh_exact_constants()
  return result

 def _depthsum_embedding(self, code: torch.Tensor) -> torch.Tensor:
""",
        "Change A sampler cache lifecycle",
    )
    source = replace_once(
        source,
        """  embs = nn.functional.pad(self.rvq_embs, [0, 0, 0, 1]) # num_quantizers x (codebook_size + 1) x latent_size
  res = nn.functional.embedding(code[0], embs[0])
""",
        """  if not self._exact_constants_ready or self._rvq_embs_padded is None:
   raise RuntimeError("EarTTS exact constants were not refreshed after weight load")
  embs = self._rvq_embs_padded # num_quantizers x (codebook_size + 1) x latent_size
  res = nn.functional.embedding(code[0], embs[0])
""",
        "Change A padded RVQ use",
    )
    source = replace_once(
        source,
        """  return loader.load_weights(weights)
""",
        """  loaded = loader.load_weights(weights)
  # This is the final runtime weight-load boundary: parameters already have
  # their serving device/dtype, so derived constants may now be constructed.
  self.model.sampler._refresh_exact_constants()
  return loaded
""",
        "Change A post-load refresh",
    )
    return source


def apply_change_b(source: str) -> str:
    if CHANGE_B_MARKER in source:
        return source
    if CHANGE_A_MARKER not in source:
        raise RuntimeError("Change B requires Change A")

    source = replace_once(
        source,
        """  self.residual_scale = nn.Parameter(torch.tensor(init_residual_scale, dtype=torch.float32), requires_grad=False)

  self.final_norm = RMSNorm(hidden_dim) if final_norm else nn.Identity()
""",
        f"""  self.residual_scale = nn.Parameter(torch.tensor(init_residual_scale, dtype=torch.float32), requires_grad=False)
  # {CHANGE_B_MARKER}. These buffers are derived from frozen parameters and
  # are valid only for the bytes/device/dtype of the most recent completed
  # weight load (or _apply transition). In-place mutation requires refresh.
  self.register_buffer(\"_gate_sigmoid\", None, persistent=False)
  self.register_buffer(\"_residual_scale_sigmoid\", None, persistent=False)
  self._exact_constants_ready = False

  self.final_norm = RMSNorm(hidden_dim) if final_norm else nn.Identity()

 def _invalidate_exact_constants(self):
  self._gate_sigmoid = None
  self._residual_scale_sigmoid = None
  self._exact_constants_ready = False

 def _refresh_exact_constants(self):
  gate = torch.sigmoid(self.gate.detach())
  residual = torch.sigmoid(self.residual_scale.detach())
  # Re-evaluate the original expressions as byte oracles at the lifecycle
  # boundary. This sync is startup-only and is outside serving/graph capture.
  if not torch.equal(
   gate.reshape(-1).view(torch.uint8),
   torch.sigmoid(self.gate.detach()).reshape(-1).view(torch.uint8),
  ):
   raise RuntimeError(\"{CHANGE_B_MARKER}: gate sigmoid byte check failed\")
  if not torch.equal(
   residual.reshape(-1).view(torch.uint8),
   torch.sigmoid(self.residual_scale.detach()).reshape(-1).view(torch.uint8),
  ):
   raise RuntimeError(\"{CHANGE_B_MARKER}: residual sigmoid byte check failed\")
  self._gate_sigmoid = gate
  self._residual_scale_sigmoid = residual
  self._exact_constants_ready = True

 def _apply(self, fn, recurse=True):
  was_ready = self._exact_constants_ready
  self._invalidate_exact_constants()
  result = super()._apply(fn, recurse=recurse)
  if was_ready:
   self._refresh_exact_constants()
  return result
""",
        "Change B gate cache lifecycle",
    )
    source = replace_once(
        source,
        """  gate = torch.sigmoid(self.gate) # FP32
  res = torch.sigmoid(self.residual_scale) # FP32
""",
        """  if not self._exact_constants_ready:
   raise RuntimeError("EarTTS frozen constants were not refreshed after weight load")
  gate = self._gate_sigmoid # FP32
  res = self._residual_scale_sigmoid # FP32
""",
        "Change B cached sigmoid use",
    )
    source = replace_once(
        source,
        """  self.register_buffer("_rvq_embs_padded", None, persistent=False)
  self._exact_constants_ready = False
""",
        """  self.register_buffer("_rvq_embs_padded", None, persistent=False)
  self.register_buffer("_rvq_embs_norms", None, persistent=False)
  self._exact_constants_ready = False
""",
        "Change B norms buffer registration",
    )
    source = replace_once(
        source,
        """ def _invalidate_exact_constants(self):
  self._rvq_embs_padded = None
  self._exact_constants_ready = False
""",
        """ def _invalidate_exact_constants(self):
  self._rvq_embs_padded = None
  self._rvq_embs_norms = None
  self._exact_constants_ready = False
""",
        "Change B norms invalidation",
    )
    source = replace_once(
        source,
        """  self._rvq_embs_padded = padded
  self._exact_constants_ready = True
""",
        f"""  self._rvq_embs_padded = padded
  # Preserve the original per-codebook reduction shape and order. Stacking
  # happens only after each unchanged pow(2).sum(-1) has completed.
  norms = torch.stack([
   self.rvq_embs[i].pow(2).sum(-1)
   for i in range(self.num_quantizers)
  ])
  for i in range(self.num_quantizers):
   dynamic_norm = self.rvq_embs[i].pow(2).sum(-1)
   if not torch.equal(norms[i].view(torch.uint8), dynamic_norm.view(torch.uint8)):
    raise RuntimeError(
     f\"{CHANGE_B_MARKER}: RVQ norm byte check failed at codebook {{i}}\"
    )
  self._rvq_embs_norms = norms
  self._exact_constants_ready = True
""",
        "Change B norms refresh",
    )
    source = replace_once(
        source,
        """    self.rvq_embs[i].pow(2).sum(-1) # [vocab_size]
    - 2 * (r @ self.rvq_embs[i].T) # [B*T, vocab_size]
""",
        """    self._rvq_embs_norms[i] # [vocab_size]
    - 2 * (r @ self.rvq_embs[i].T) # [B*T, vocab_size]
""",
        "Change B cached norms use",
    )
    source = replace_once(
        source,
        """  self.model.sampler._refresh_exact_constants()
  return loaded
""",
        """  self.model.sampler._refresh_exact_constants()
  for module in self.model.total_emb.modules():
   if isinstance(module, GatedProjectedSumRMSNorm):
    module._refresh_exact_constants()
  return loaded
""",
        "Change B post-load refresh",
    )
    return source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("a", "b"), required=True)
    args = parser.parse_args()

    target = (
        Path(sysconfig.get_paths()["purelib"])
        / "vllm/model_executor/models/eartts.py"
    )
    original = target.read_bytes()
    original_sha = hashlib.sha256(original).hexdigest()
    source = original.decode("utf-8")
    if CHANGE_A_MARKER not in source and original_sha != QUALIFIED_EARTTS_SHA256:
        raise RuntimeError(
            f"unqualified EarTTS input: {original_sha} != {QUALIFIED_EARTTS_SHA256}"
        )

    source = apply_change_a(source)
    if args.stage == "b":
        source = apply_change_b(source)
    target.write_text(source, encoding="utf-8")
    py_compile.compile(str(target), doraise=True)
    final_sha = hashlib.sha256(target.read_bytes()).hexdigest()
    print(
        f"Patched EarTTS exact constants stage={args.stage}: "
        f"before={original_sha} after={final_sha} path={target}"
    )


if __name__ == "__main__":
    main()
