# Fable formal adversarial review — Reproducible clone-to-Playground plan

Date: 2026-08-04 · Target: `docs/reproducible-clone-to-playground-plan.md`
(post-decisions revision, including the CPU-PyTorch/Pocket-TTS clarification at
lines 52–54, 157–159, 295–297) · Reviewed against the current repository,
`docs/provenance.md`, the deployed qualified stack, and the checkpoint's actual
license text. No changes implemented.

## Verdict: APPROVE WITH CHANGES

The plan is structurally sound: the phase ladder is ordered correctly, every
phase has an exit gate, the falsified `VLLM_ALLOW_INSECURE_SERIALIZATION`
hypothesis is now correctly inverted into a must-stay-unset contract (Phase
4.6), and the resolved topology (one CUDA container, host `uv` for Pipecat +
CPU-only-PyTorch Pocket TTS) matches the qualified layout minus systemd. Three
blockers must be incorporated into the plan text before implementation; none
require changing the resolved product decisions.

## Decisive licensing finding (resolves Phase 3.1's biggest unknown)

I read the actual LICENSE shipped with the local
`NVIDIA-NemotronLabs-VoiceChat-11B` checkpoint: it is **OpenMDW-1.1** (Open
Model, Data and Weights License Agreement v1.1). It grants permission "to deal
in the Model Materials without restriction," which permits creating and
redistributing the quantized derivatives, subject to two retained obligations
on distribution: (1) include a copy of the agreement, and (2) retain all
copyright/origin notices; plus a patent-litigation termination clause. There is
no non-commercial restriction, no derivative-naming mandate, and no output
restriction. Publication at `pipecat-ai/NVIDIA-NemotronLabs-VoiceChat-11B-Spark`
is therefore viable. Phase 3.1 should be rewritten from an open-ended "license
review" into these concrete requirements (see I2).

## BLOCKERS (plan-text changes required before implementation)

**B1 — The calibration-corpus identity fork is unresolved and silently breaks
either reproducibility or qualification.** Phase 2.3 packages a
"redistributable, disjoint calibration/evaluation corpus," while the Phase 2
exit gate requires conversions to "match … the release manifest" and Phase 3
publishes "the qualified derived weights." These are jointly satisfiable ONLY
if the published corpus is byte-identical to the corpus that produced
Production Candidate 1. If any fixture is replaced for redistributability, GPTQ
calibration output changes, the published weights are no longer PC1, and every
existing qualification result — including the 15-minute sustained gate — is
invalid for the release. Required text (add to Phase 2 after item 3):

> "Decision fork: if the exact Production Candidate 1 calibration corpus is
> redistributable, ship it and the release weights ARE Production Candidate 1
> (current qualification evidence carries). If any calibration input must be
> substituted, the conversion output is a NEW candidate: its hashes will not
> match PC1, PC1 hashes become comparison-only evidence, and the full Phase 7
> ladder — including the 15-minute sustained qualification — must pass on the
> new weights before publication. The release manifest is generated from
> whichever candidate is actually published, never copied from PC1."

**B2 — Removing the loopback API key (Phase 5.2) is an unacknowledged change
to the qualified server contract.** The deployed strict-v2 server enforces the
key and advertises `auth_required: true`; its sources are hash-pinned in the
production manifest. "Remove the loopback API key" therefore means either a
runtime code/config change (which must go through the same parity gates as any
runtime change) or a launcher-level alternative. Recommended replacement text:

> "The launcher generates a random ephemeral API key at each `./voicechat up`
> invocation and passes it to the model container and the Pipecat process via
> their process environments only. No key is written to disk, no `.env` exists,
> and the key dies with the stack. Alternatively, if the server is patched to
> support an explicit auth-disabled loopback mode, that patch is a runtime
> change and must pass the runtime-parity gates before adoption."

The ephemeral-key path preserves the qualified server behavior byte-for-byte
and still satisfies the "no persisted credential" goal.

**B3 — No stop→restart cycle gate, despite a documented residual-GPU-state
failure mode.** The standalone-cutover investigation proved that unclean model
shutdown can leave GPU state that hangs the NEXT startup at EarTTS prefill
(root-caused and cleared with a clean-memory retry; recorded in
`docs/reviews/fable-final-review.md`). The plan's lifecycle work (Ctrl-C
teardown, container removal on exit, `down` for orphans) is never tested for
exactly this. Add to Phase 7 (and to `./voicechat test`):

> "Restart-cycle gate: with the stack healthy, send SIGINT to `./voicechat up`,
> wait for full teardown, immediately run `./voicechat up` again, and require
> health-ready within the documented cold-start budget. Repeat at least three
> consecutive cycles, plus one cycle where the launcher process is killed with
> SIGKILL and recovery goes through `./voicechat down`. This gate exists
> because of the documented residual-GPU-state startup hang."

## IMPORTANT

**I1 — The env-variable contract does not disappear with `.env`, and the plan
should say so.** `text_input.py` reads `VOICECHAT_TYPED_INPUT_*` and
`VOICECHAT_RNNT_EOU_FRAMES` from the process environment (the typed-input
trailing-silence duration is derived from the EOU profile), and the model
server consumes the `VOICECHAT_*` runtime contract the same way. Phase 5.1
should state explicitly: "the launcher materializes
`config/production-candidate-1.toml` into child process environments (container
`-e` flags; Pipecat process env). 'No `.env` parsing' means no dotenv file, not
no environment variables." Without this sentence, an implementer may refactor
settings plumbing (a runtime-visible change) when only the delivery mechanism
should change.

**I2 — Make Phase 3.1 concrete now that the license is known.** Replace the
open-ended review with: include the OpenMDW-1.1 text and NVIDIA copyright
notice in the HF repo; state derivation from
`nvidia/NVIDIA-NemotronLabs-VoiceChat-11B@fb0f94ea…` (also as `base_model` HF
metadata); include a non-affiliation line in the model card ("community
quantization, not an NVIDIA release") since the repo name embeds NVIDIA
branding; verify separately that no `NVIDIA-Nemotron-Nano-9B-v2` skeleton files
or Pocket TTS assets are republished (bootstrap downloads them from their own
repositories); and record the license conclusion in `docs/provenance.md` and
`THIRD_PARTY_NOTICES.md`. The remaining open item is only the calibration
corpus license (ties to B1), not the weights.

**I3 — Host-side reproducibility needs more than `uv.lock`.** Pin the Python
interpreter version (`uv python pin`; today's lock resolves against one), and
make Phase 6's `--offline` proof include `uv sync --frozen --offline`
succeeding from the bootstrap-populated cache — this forces bootstrap to
pre-fetch all aarch64 wheels and catches any wheel that would be resolved from
the network on a clean machine. `./voicechat doctor` should verify interpreter
version and wheel-cache completeness.

**I4 — The Phase 5 exit gate over-claims.** "No runtime credential" conflicts
with B2's ephemeral key and with optional tunnel settings (a tunnel basic-auth
credential is a credential). Reword to: "no persisted credential: nothing
secret is written to the repository, a `.env`, a config file, or a Docker
layer; per-invocation secrets live only in process environments."

**I5 — Add a model-port exposure negative test.** Phase 7 should assert that
the model WebSocket is bound to `127.0.0.1` only and is unreachable from a
non-loopback interface, and that only the Pipecat port is externally
reachable — parity with the qualified deployment and the plan's own principle
(line 52). Today this is true by construction; after the Docker/publish
refactor it must be proven, since `docker run -p` defaults to `0.0.0.0`.

**I6 — Crash-path container hygiene.** `docker run --rm` plus a deterministic
container name, and `./voicechat up` must detect a stale same-name container
(launcher killed with SIGKILL) and refuse or replace it with a clear message;
`./voicechat down` removes it. Phase 5.3 covers the clean path only.

## SUGGESTIONS

- Phase 6.5's actionable-failure list should explicitly include the gated-
  download case: if the upstream HF repo requires accepting terms, bootstrap
  must detect the 401/403 and print the exact acceptance URL and retry
  command rather than failing generically.
- Record the measured cold-start-to-ready number (Phase 7.3) in the README's
  expected-startup-time line (Phase 8.1) so the two stay linked.
- `./voicechat test --live` should include the typed-input Pocket TTS gate and
  the B3 restart cycle, not only the voice path.
- "Suggested implementation order" items 4–8 switch to lowercase/semicolons —
  trivial, but this list will be quoted in issues; normalize it.
- Phase 4.7's SBOM audit should also grep image history for the retired EA
  image tags and NGC registry paths by name, since "EA/NGC lineage" is the
  specific contamination this project must prove absent
  (`docs/provenance.md:44`).

## What I verified directly (evidence base)

- License: read the checkpoint's shipped `LICENSE` (OpenMDW-1.1, quoted terms
  above).
- Current `.env.example`: 20+ variables — three credential-bearing
  (`VOICECHAT_REALTIME_API_KEY`, `NEMOTRON_VOICECHAT_API_KEY` pair) and the
  rest paths/frozen-contract values; confirms Phase 5's TOML/CLI split is the
  right shape and that I1's env plumbing is real.
- `deploy/stack`: uses `systemd-run`/`systemctl` for Pipecat and ngrok and
  `docker run -d` for the model — confirms the plan's systemd-removal scope.
- `docs/provenance.md`: EA/NGC lower-layer admission (line 44), pinned Speech
  commit + retained patch with startup byte-verification, pinned vLLM fork
  commit, dual-run byte-identical conversion claim, `ea_w8a32` ABI exception —
  all consistent with Phases 1, 2 and 4.
- Live container environment: `VLLM_ALLOW_INSECURE_SERIALIZATION` absent,
  matching Phase 4.6 as corrected.
- Host lock: CPU-only `torch==2.9.1+cpu` via the explicit PyTorch CPU wheel
  index, with Pocket TTS as its consumer — matches the clarified topology text.

With B1–B3 written into the plan (exact text above) and I1–I6 reflected, the
plan credibly delivers its stated outcome and is safe to hand to
implementation. Re-review of the amended plan text can be a fast delta pass.

---

# Delta review — 2026-08-04, revised Pocket TTS boundary (server-side bridge)

Scope: the CURRENT plan text after (a) incorporation of B1–B3/I1–I6 and
(b) the architecture change moving Pocket TTS inside the Voicechat container
as an isolated CPU worker, with typed input crossing the WebSocket as a new
wire event and host `uv` installing only Pipecat.

## Delta verdict: APPROVE WITH CHANGES (one blocker, four IMPORTANT)

First, closure accounting: every item from the formal review is genuinely
incorporated — B1 corpus fork (2.3), B3 restart-cycle gate (7.7), I1 env
materialization (5.1), I2 licensing made concrete with OpenMDW-1.1 (3.1),
I3 `uv python pin` + `--frozen --offline` (5.1/6.4), I4 exit-gate rewording
(Phase 5 gate), I5 loopback negative test (7.4), I6 stale-container hygiene
(5.4), plus the EA-name audit (4.7) and gated-401/403 handling (6.5). B2 was
resolved via my stated alternative — an explicit auth-disabled local mode
gated as a runtime change through all parity and live gates (5.3) — which I
accept as written.

## Question 1: does the isolated CPU worker/IPC design truly avoid
CUDA/CPU PyTorch conflicts?

**Yes, structurally.** A separate Python environment executed as a separate
process over IPC is real isolation: the CUDA server process never imports the
CPU torch build, and this is the same architecture as the already-qualified
CPU codec worker. Two conditions must be stated for it to stay true:

- Launch the worker with a scrubbed environment (its own venv's interpreter,
  no inherited `LD_LIBRARY_PATH`/CUDA paths from the server process), so the
  CPU wheel cannot dynamically bind CUDA-adjacent libraries.
- Re-measure the Pocket prewarm/real-time-factor gate **inside the container
  worker** (Phase 7.3 is location-neutral as written); the 1.95× realtime
  qualification was measured in the host CPU environment and does not carry
  automatically.

## Question 2: does moving typed-input ownership across the WS boundary
preserve exactly-once context / cancellation / turn semantics?

**Only if the plan specifies more than it currently does.** Phase 5.2's
preservation checklist names the right properties but under-specifies the two
mechanisms that made the host bridge correct.

**D1 (BLOCKER) — "one bounded typed-input event" cannot carry the qualified
semantics; the wire contract and the ducking authority must be enumerated.**
The hard-won correctness core of the host bridge was: half-duplex mic
arbitration, cancel-then-replace with mandatory EOU-derived trailing-silence
closure (~2.08 s at the deployed profile) whenever any synthetic audio was
emitted, and serialized replacement. Server-side, Pipecat keeps streaming mic
PCM over the WebSocket during synthesis — so *someone* must arbitrate, and a
single request event gives the client no visibility. Required plan text for
Phase 5.2:

> "The wire contract comprises: typed-input request (bounded text, client job
> id), accept/reject with reason, injection-started, injection-finished with
> disposition (completed | cancelled | replaced | error). The SERVER is the
> sole half-duplex authority: it gates incoming microphone PCM from the
> moment a typed job is accepted until injection plus trailing-silence
> closure completes, so client ducking is advisory UI state only. The server
> ports the qualified closure rules unchanged: trailing silence derived from
> the RNNT EOU profile with margin, emitted whenever any synthetic audio was
> injected — including after cancellation or replacement — before any
> replacement audio begins; typed jobs are serialized; a new typed request
> while one is active follows cancel-then-replace."

Without this, the RNNT partial-utterance merge race that B1 of the original
Pocket plan closed is reopened at the protocol boundary.

**D2 (IMPORTANT) — typed-turn transcript double-entry.** Today the byte-exact
typed text enters Pipecat context once, and RNNT's (lossy) transcription of
the synthetic speech is the model-side record. Server-side synthesis will emit
user-transcription events for typed turns like any other audio. The plan must
state: the server tags transcription events for typed turns with a source
marker and the client job id; Pipecat substitutes the byte-exact typed text
into its context exactly once and does not also commit the RNNT transcript
for that turn. "Transcript-source attribution" (5.2) gestures at this; the
exactly-once rule needs to be explicit or context drifts from qualified
behavior.

**D3 (IMPORTANT) — the CPU contention budget moved inside the container.**
Host-side Pocket competed with the container only through the OS scheduler;
in-container it shares any cpuset/quota with the latency-critical 80 ms frame
loop and the reserved codec-worker cores. Phase 4.4 must add: explicit core
affinity (or cgroup weight) for the Pocket worker that excludes the codec
cores, and Phase 7.5's 15-minute sustained qualification must interleave
typed-input turns, since synthesis load is now inside the qualified realtime
envelope. A voice-only sustained pass no longer covers the shipped
configuration.

**D4 (IMPORTANT) — version the protocol; two stale strict-v2 references.**
Adding a typed-input event with "no legacy mode" is a contract revision.
Name it (strict-v3 or v2.1), advertise it in `/health`, and update Phase 4.2
("strict-v2 server behavior" patch item) and Phase 7.1 ("strict-v2 protocol")
to reference the revised contract; otherwise implementers may bolt the event
onto v2 without a version signal, defeating the "strict" discipline.

**D5 (IMPORTANT) — disposition of the just-qualified host bridge.** Commit
`29d11bc` shipped a host-side bridge, an 8-test suite, and live E2E evidence
hours ago; the plan supersedes all of it and says nothing about it. Add: the
host synthesis path (`text_input.py` synthesis/injection, host `pocket-tts` /
CPU-torch / scipy dependencies) is removed from the Pipecat lock; the 8-test
suite's invariants migrate to (a) hosted-CI Pocket worker + IPC tests
(CPU-only, no GPU needed), (b) Pipecat-side wire-contract tests, and (c) DGX
integration gates; `docs/reviews/fable-pocket-tts-live-e2e.md` is marked as
evidence for the retired host architecture. Also specify worker-crash
resilience: the server detects Pocket worker death, fails typed requests with
error events, leaves the voice path unaffected, and either restarts the
worker or degrades health explicitly.

## Question 3: do any old host-Pocket assumptions remain?

Mostly clean. Remaining traces, all minor:

- Phase 4.5 still says Pocket assets download "into the … host cache" — true,
  but add that the container reaches them via the existing read-only cache
  mount, since the consumer moved inside the image.
- Phase 7.1 runs the "typed-input Pocket TTS bridge" on ordinary CI — now
  split per D5: worker/IPC tests remain hosted-CI-viable (CPU-only); server
  integration is DGX-only.
- Implementation-order item 1 ("Decide the release topology … below") is
  stale — the decisions are resolved; reword to past tense.
- The IPC serialization should be constitutionally boring: length-prefixed
  JSON or flat binary over a unix socket, never pickle — the same
  insecure-serialization principle the plan already enforces for vLLM (4.6),
  applied to the new boundary.

With D1 written into Phase 5.2 and D2–D5 reflected, the revised architecture
is sound — arguably stronger than the host-side design, since one process now
owns audio arbitration and the RNNT closure invariant instead of two
processes cooperating across a WebSocket.

---

# Final disposition — 2026-08-04, delta verification of the amended plan

## Verdict: APPROVE. The plan is implementation-ready.

Every delta item is verified present in the current plan text, not merely
acknowledged:

- **D1 (wire contract / ducking authority) — CLOSED.** Phase 5.2 names the
  single supported contract `strict-v3` (advertised by `/health`, no v2
  mode), enumerates the full typed-input event family (request with bounded
  text + client job ID, accept/reject with reason, `injection_started`,
  `injection_finished` with disposition completed/cancelled/replaced/error),
  makes the server the sole half-duplex authority from acceptance through
  trailing-silence closure with client ducking demoted to advisory UI state,
  and ports the qualified closure rules by name (EOU-profile-derived silence
  with margin, silence whenever any synthetic audio was injected including
  cancellation/replacement, closure completes before replacement audio,
  serialized jobs, cancel-then-replace).
- **D2 (transcript exactly-once) — CLOSED.** Typed-turn transcription events
  carry source + client job ID; Pipecat commits the byte-exact typed text
  exactly once and does not also commit the RNNT transcript (5.2).
- **D3 (in-container CPU budget) — CLOSED.** Worker cores exclude the
  latency-critical codec cores (4.4), and the 15-minute sustained
  qualification now interleaves typed turns so synthesis, affinity, codec
  work, and 80 ms pacing are tested together in the shipped configuration
  (7.5).
- **D4 (protocol versioning) — CLOSED.** `strict-v3` appears in the Phase 4.2
  patch inventory and Phase 7.1 test list; no stale strict-v2 references
  remain anywhere in the plan.
- **D5 (host-bridge disposition + worker resilience) — CLOSED.** Phase 7.1
  migrates the eight qualified host-bridge invariants into worker/IPC,
  Pipecat-contract, and DGX-integration tiers; removes host `pocket-tts`,
  CPU-PyTorch, and SciPy dependencies and synthesis code from Pipecat; and
  marks the former host-bridge live report as retired-architecture evidence.
  Phase 4.4 specifies worker-death detection, typed-input error without
  harming the voice path, and restart/re-prewarm or explicit health
  degradation; Phase 5.2 fails health readiness if the worker cannot load
  and prewarm.
- **All minor items — CLOSED.** Non-pickle length-prefixed JSON/flat-binary
  Unix-socket IPC (4.4); scrubbed CUDA/library-path worker environment
  (4.4); Pocket assets mounted read-only into the container from the
  verified host cache (4.5); Pocket RTF gate measured inside the container
  worker (7.3); implementation-order item 1 rephrased to record the resolved
  decisions (356–359).

One non-gating note for implementation time: `docs/known-limitations.md`'s
half-duplex entry describes the retired host-bridge mechanism and should be
reworded to the server-owned gating when Phase 5.2 lands — Phase 8.4 already
covers honest limitation documentation, so no plan change is needed.

This concludes the plan-review cycle: formal review (APPROVE WITH CHANGES,
B1–B3/I1–I6), delta review of the server-side Pocket boundary (APPROVE WITH
CHANGES, D1–D5), and this final verification (APPROVE). Subsequent reviews
should target implementation checkpoints against this plan's exit gates.
