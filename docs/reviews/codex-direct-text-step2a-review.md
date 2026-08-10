# Codex adversarial review — direct-text Step 2A

## Boundary SOTC and FP32 manifest review (2026-08-09)

Scope: the post-r6 boundary-only SOTC refinement, qualification-only FP32
manifest relocation, retained NVIDIA Speech patch, telemetry, and focused
tests.

### Findings resolved during review

- The first implementation initialized the deferred-SOTC latch only in the
  session reset path, while `infer_one_step()` can run during model warmup.
  Constructor initialization was added for the latch and all related counters.
- Merely observing `_external_user_eou_requested` does not by itself prove that
  the current position consumes the request. Staging was moved after RNNT turn
  taking and now requires all three observable facts on the same position: the
  raw function prediction was SOTC while the request was pending, the request
  was consumed, and the effective agent token became BOS with the agent state
  open. Failure of that conjunction is fatal instead of preserving an
  ambiguous SOTC.
- Staged and replayed numeric frame indices were added alongside counts,
  and pending state so an async FC continuation cannot hide the original replay
  edge. A live trace later showed that a derived `replayed_this_frame` flag
  stayed true while FC async intentionally held the ordinary context head at
  that position; both derived `this_frame` flags were removed as misleading.

### Invariants checked

- Settlement SOTCs remain effective PAD and are never staged.
- The EOU/BOS position remains effective function PAD. Its actual SOTC is
  staged only after the BOS transition is proven.
- The next Nano position consumes the preceding effective PAD. Only after that
  prediction is the staged SOTC written as the current effective function token
  and passed to the unchanged FC state machine; the recurrent function feedback
  therefore remains coherent.
- The latch clears before FC mutation, cannot be overwritten, is reset at
  session boundaries/mode enablement, is cancelled on mode disablement, and
  blocks direct-position preflight while pending.
- No prompt parsing, keyword routing, tool selection, canned response, or
  semantic exception was introduced.
- The FP32 adapter requires the exact diagnostic manifest kind, exact Nano and
  EarTTS component set, and exact original extraction paths. A deep copy changes
  only the two component paths and adds deterministic audit metadata containing
  the source SHA-256 and path mapping. The immutable source is re-read after the
  atomic write. The adapter and artifacts are separate read-only mounts.
- Production `validate_manifested_model()` was not relaxed; component paths,
  files, aggregate model digests, checkpoint provenance, and reproducibility
  remain server-validated.

### Evidence and residual gate

The complete static suite passes: 422 passed, 3 skipped, with four pre-existing
warnings and three subtests. Ruff, patch/source byte comparison, reverse-apply,
Python compilation, and `git diff --check` (excluding the patch-as-data file)
also pass.

Static tests intentionally cannot prove the model emits the expected boundary
SOTC or that the async call completes. The remaining gate is r7 on the rebuilt
image: c009 must remain tool-free; c013-c019 must show one consumed-boundary
stage, one post-BOS replay, and correct tool completion; FP32 and W8 must each
execute all 20 strict semantic cases.

**Verdict: APPROVE STATIC IMPLEMENTATION; GPU EVIDENCE REQUIRED.**

## r7 live follow-up

Pocket completed 20/20 structurally. Case c009 suppressed its settlement SOTC
without staging; c013-c019 produced seven consumed-boundary stages, seven
post-BOS replays, and seven completed function cycles. The FP32 adapter passed
server validation and the direct lane retained seven response WAVs before c007
hit the fixed 240-frame non-terminal bound.

That bounded abort exposed two state-observability defects. First,
`replayed_this_frame` stayed true while FC async intentionally held the ordinary
context head at the replay model position; the derived staged/replayed booleans
were removed, retaining truthful absolute model frames and counts. Second, the
wrapper-level `_agent_idle` fallback remained false after abort and caused the
next fresh stream's direct preflight to reject before mutation. The existing
transport-session reset now restores `_agent_idle = True`; the 240-frame bound
and strict preflight remain unchanged. r7 is retained as negative/partial
evidence and cannot qualify either direct lane.
