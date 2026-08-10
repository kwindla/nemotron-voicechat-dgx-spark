# Intermittent Nano step stall: investigation and fix plan

## Scope

Eliminate the intermittent production hang in which one Nano step never
returns. This is separate from the continuously-progressing function-call
latency path. The change must preserve the qualified VoiceChat model, turn
protocol, function-calling semantics, EarTTS behavior, and browser input/output
contracts.

## Qualification result

The repair is implemented and qualified on immutable runtime image
`sha256:23ce602a53abb0898a8043d8238dc8433b29d6cd26a9675b32f4e23a964f376d`.
Its network-isolated audit passed, and `/health` reports all four PAD-pair
policy switches disabled.

The decision preflight ran for 220 seconds and completed 2,729 model metrics
with one Nano request-position advance per inference call. Measured server-step
mean/p95 were 71.13/73.99 ms; microphone receipt lag peaked at 255.17 ms and
ended at -1.59 ms. The rebuilt production image then completed a 100-second
paced run with 1,227 model metrics. Its mean/p95 were 69.23/72.99 ms; all 1,225
observable Nano request-position deltas were exactly +1; receipt lag peaked at
227.43 ms and ended at -0.83 ms. Neither run stalled or emitted an inference
error.

Against the closest 220-second PAD-pair baseline, the same-image sequential
preflight changed mean server-step latency from 68.24 to 71.13 ms, while p95
improved from 84.77 to 73.98 ms and p99 improved from 94.79 to 84.64 ms. The
rebuilt production run brought mean back below 70 ms at 69.23 ms. For an 80 ms
real-time cadence, this is a net latency improvement: the small mean tradeoff
is outweighed by materially lower queue-debt-sensitive tail latency.

The browser gate passed an actual function call, spoken input, typed math and
typed recall, and bidirectional WebRTC audio. Its model trace contains exactly
one requested/completed function cycle, EOTR, and post-function BOS, while all
307 fresh timed calls contain exactly one inference position. The browser's
voice-to-voice value from that run is intentionally not used as latency
evidence: the synthetic voice began while the preceding long tool response was
ending, so the intervals overlap. The roughly 23-second ordinary function-call
path remains a separate production-latency problem; it did not exhibit the
missing-completion signature repaired here.

A separate no-tool browser run avoided that overlap and passed the same spoken
input, typed math/recall, and bidirectional-audio assertions. It measured a
1,386.4 ms browser speech-tail-to-first-speech point sample; this reliability
repair neither changes nor requalifies the existing voice-latency target.

## Retained evidence

The failing session is retained at:

`~/.local/state/nemotron-voicechat/traces/model/2d08e743-39bb-4119-866d-8ab5c215581c/events.jsonl`

It starts a typed Pocket carrier and completes model frames 0 through 22. The
Nano request-position sequence ends:

| frame | Nano generated tokens | Nano interface time |
|---:|---:|---:|
| 19 | 21 | 62.89 ms |
| 20 | 21 | 0.45 ms |
| 21 | 23 | 62.51 ms |
| 22 | 23 | 0.41 ms |

Frames 20 and 22 are the custom PAD-pair scheduler's synthetic buffered rows:
they return PAD without advancing the Nano request. Frame 21 accepts the two
positions accumulated by frames 20 and 21. With the function state idle, no
turn-control barrier pending, and both effective channels PAD, frame 23 must
enter `_generate_packed_pad_pair()` for the row buffered at frame 22 plus the
new row. Perception completed for that position, but the packed Nano call never
returned; no frame-23 `model_step` was published and the client eventually hit
its liveness deadline.

This is not the ordinary function-call delay. Successful function cycles show
many completed Nano positions at roughly 80--100 ms each. A PAD-pair hang shows
one missing completion indefinitely.

## NVIDIA comparison

NVIDIA Speech commit `911ec674ab40f04302ef33672be4179f45a7310f`
uses one-position `LLMStreamingEngine.generate_next_token()` continuations.
It does not contain `voicechat_pad_pair_token_id`, the custom speculative
proposal, function-head conjunctive rejection, Mamba rollback shadows, or the
dedicated two-position FULL CUDA graph.

Those mechanisms are local throughput optimizations installed by:

- `runtime_optimizations._install_public_pad_pair_engine()`;
- `patch_public_pad_pair_scheduler.py`; and
- `patch_pair_full_graph.py`.

The evidence therefore localizes the observed hang to a custom production
optimization boundary, not to NVIDIA's ordinary sequential Nano contract. It
does not yet distinguish a scheduler/output wakeup failure from a worker/CUDA
graph failure inside that packed request. That distinction is unnecessary for
the safest production repair because a wedged vLLM/CUDA request cannot be
cancelled and resumed transactionally.

A scan of the retained model traces found one occurrence with this exact
signature among the recent sessions: the session above. That is enough to make
an unrecoverable production path unacceptable, but not enough to treat the
throughput cost of removing it as negligible.

## Decision preflight

Retirement is conditional on proving that NVIDIA's sequential path sustains the
80 ms media cadence on this DGX Spark. Before changing the canonical policy,
run the current immutable image with a temporary reviewed config that sets:

- `VOICECHAT_NANO_PAD_PAIR=0`;
- `VOICECHAT_NANO_PAD_PAIR_CONDITIONAL=0`;
- `VOICECHAT_NANO_PAD_PAIR_CONTROL_BARRIER=0`; and
- `VOICECHAT_NANO_PAIR_FULL_GRAPH=0` when the candidate-2 overlay is present.

The preflight uses the existing C1 stage timings and sustained runner for at
least 1,000 ordinary input positions after warmup. It records every current
`wrapper_total_ms`, `nano_interface_ms`, and `server_step_ms`, plus actual
input send-to-receive lag. The preregistered pass condition is:

- no model-step or liveness timeout;
- no packed-pair decision and exactly one Nano position per inference call;
- settled mean `server_step_ms <= 72 ms`, leaving at least 10% headroom against
  the 80 ms cadence;
- settled p95 `server_step_ms <= 80 ms`;
- non-positive queue-lag slope over the final half of the run; and
- peak measured input queue lag no greater than 400 ms and back below 160 ms
  at the end of the run.

This is a decision gate, not post-deployment observation. If it passes, proceed
with retirement below. If it fails, do not ship either configuration: retain
PAD-pair in the currently deployed image while instrumenting and repairing the
packed request boundary, then stress that repaired path before promotion.

## Fix

Retire Nano PAD-pair speculation from the production runtime and return to
NVIDIA's sequential one-position Nano continuation:

1. Set `VOICECHAT_NANO_PAD_PAIR=0`,
   `VOICECHAT_NANO_PAD_PAIR_CONDITIONAL=0`, and
   `VOICECHAT_NANO_PAD_PAIR_CONTROL_BARRIER=0` in the canonical semantic
   environment. Candidate 2 must also set
   `VOICECHAT_NANO_PAIR_FULL_GRAPH=0`; a graph for a retired packed call must
   not remain enabled.
2. Keep the implementation and converted artifact metadata available only for
   explicit offline diagnosis. Do not delete historical qualification or
   retained evidence.
3. Make startup, health, and runtime provenance report the production policy as
   disabled and reject any contradiction between the canonical environment and
   the running container.
4. Do not add an inference timeout/retry. Once a packed GPU request hangs, the
   model state and output iterator cannot be proven recoverable; retrying could
   duplicate or corrupt recurrent state.
5. Do not change Nano weights, sampling, prompts, tool schemas, turn thresholds,
   EarTTS, codec, or response gates in this repair.

Both `config/production-candidate-1.toml` and
`config/production-candidate-2.toml` are canonical deployment inputs and are in
scope. Add both files to `PUBLIC_RUNTIME_PAYLOAD_FILES`, so their exact bytes
are copied, audited, and included in the public-runtime payload/source hashes.
Together with the in-image canonical values in `provenance.py`, this makes
`./voicechat bootstrap` treat an older PAD-pair-enabled image as stale and
rebuild it after a clone or pull.

## Tests before rebuild

1. Add a regression that reconstructs the failing request-position sequence and
   proves the last completed row is a buffered PAD whose next operation would
   be the packed speculative path.
2. Assert the production environment disables all three PAD-pair flags and the
   model command passes those exact values.
3. Assert startup validation and `/health.nano_pad_pair` report disabled while
   the ordinary Nano artifact and custom heads remain valid.
4. Exercise the installed wrapper with PAD-pair disabled and prove every normal
   call delegates to NVIDIA's original one-position method with no pending
   draft state.
5. Preserve existing opt-in unit coverage for the retired diagnostic path so a
   future investigation cannot silently change its historical semantics.
6. Assert both production TOMLs are exact payload inputs, copied payload audit
   sees them, and changing either file changes checkout/runtime identity,
   forcing bootstrap rebuild rather than stale-image reuse.

## Live qualification

1. Only after the sequential-cadence preflight passes, land the canonical
   policy change. Rebuild and audit a new immutable public runtime; require the
   audit and health provenance to show PAD-pair and candidate-2 full-graph
   execution disabled.
2. Run a bounded multi-session stress campaign spanning typed Pocket input,
   microphone input, no-tool replies, and actual function cycles. Require:
   no model-step/liveness timeout; one Nano request-position advance per real
   inference position; no packed-pair decisions; clean response/session close;
   and unchanged text/audio semantic gates.
3. Retain per-step latency and input queue-lag distributions. Sequential Nano
   may trade throughput for reliability; report that cost honestly and fail the
   promotion if it recreates unbounded input or playback debt.
4. Run the end-to-end browser voice-and-text test, independently checking user
   transcription, model text, received speech-like audio, tool completion, and
   later-turn response latency.
5. Leave the qualified stack running only after all structural and semantic
   gates pass.

## Follow-up, not part of this repair

If sequential Nano is reliable but too close to the 80 ms media cadence, design
a new throughput optimization against NVIDIA's one-position path. It must not
use an unrecoverable speculative packed request without a deterministic
component replay, worker-level stress evidence, and an explicit state recovery
contract. The separate normal function-call latency work resumes after this
reliability gate is green.
