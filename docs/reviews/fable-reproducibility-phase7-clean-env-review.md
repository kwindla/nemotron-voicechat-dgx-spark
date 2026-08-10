# Fable review — Phase 7 clean-environment/live-entry boundary

Date: 2026-08-04 · Scope: torch-free-host fix across four conversion test
modules, `run_live_suite.py::run_conversion_tensor_tests`,
`tests/runtime/test_live_suite.py`, `docs/qualification.md`. No files
edited.

## Verdict: APPROVE — the contract is correctly implemented; no false-green
path exists.

- **Visible, explained skips:** the torch-dependent modules guard with
  `pytest.importorskip("torch", reason="conversion tensor tests run in the
  public runtime container")` (confirmed in `test_package_replay_corpus.py:8`
  and `test_nano_attribution.py:9`) — hosted CI and the torch-free host
  suite show them as SKIPPED with an explanation, never as passed.
  Evidence counts corroborate: clean host 203/7 skipped vs dev 246/2.
- **Container lane is genuinely isolated and immutable:** the live-suite
  stage runs `python -m pytest` inside the pinned public runtime image
  with `--network none`, the repo mounted READ-ONLY at
  `/workspace/project`, `-p no:cacheprovider` (no writes to the mount),
  and no `--gpus` flag — offline, no-GPU, image-immutable. 23/23 in the
  exact container lane per evidence; harness tests 4/4 locally.
- **Ordering per docs:** conversion tensor contracts run FIRST in the live
  ladder, before component gates (`docs/qualification.md:34`), so a broken
  conversion contract cannot be masked by later gates.
- **Plan-CI promise:** the plan says hosted CI tests everything not
  requiring GPU/large artifacts. Torch-CPU is technically installable in
  CI, but the Phase 5 host-lock purity contract (Pipecat host is
  torch-free) takes precedence; the deviation is DOCUMENTED in
  `docs/qualification.md:16-17` with the exact compensating lane. Recorded
  as an accepted, documented deviation — not a conflict.
- **No Docker/network/GPU leakage into the host suite:** the four modules
  are the only torch importers; the host suite requires neither Docker nor
  network.

The clean-environment boundary is sound; proceed with live qualification.

---

## Delta — 2026-08-04, root-owned fixture-manifest remap

### Verdict: APPROVE. Evidence integrity and path safety both hold.

- **Container evidence preserved byte-for-byte:** the host only READS
  `fixtures/manifest.json` (root-owned, produced inside the offline
  container) and never rewrites it; the remapped view goes to a NEW
  host-owned `fixture-manifest-host.json` in the run root, with an
  in-code comment stating the immutable-evidence intent.
- **Traversal-proof by construction:** remapping uses
  `Path(recorded).name` — only the basename is joined to the known host
  fixtures directory, so any `../` or absolute container path in the
  manifest collapses harmlessly.
- **No ownership assumptions:** no chmod/chown/user flags anywhere; the
  host reads world-readable root-owned files and writes only its own
  file in the host-owned run root.
- **Stale/collision behavior correct:** the host view is derived data and
  is regenerated per run; the container manifest (the evidence) is never
  touched.
- **Downstream unification:** browser and direct-client consumers read
  `args.fixture_manifest` (the host view) exclusively — one path model,
  host WAV paths confirmed by the real offline container generation.
- Focused tests 4/4 including the remap assertion; lint green.
