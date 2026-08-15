# Tool-call latency fix plan

Status: benchmarked and production-qualified through implementation step two.
This plan is based on the retained v2 latency/mechanism benchmark. The twelve
preregistered safety baseline cells were retained before the first
behavior-changing experiment and were rerun against each promoted behavior.

## Outcome and scope

The slow tool path is real and is primarily serialized model work. It is not
external API latency, Pipecat queueing, client pacing, or the previously fixed
intermittent Nano stall.

For the representative 72-token result, the path from SOTC to first audible
post-tool speech is approximately:

| component | measured time |
|---|---:|
| autoregressive tool-call emission, 27 positions | 2.32 s |
| external tool | less than 0.02 s |
| unconditional layer-one on-hold utterance | 2.08 s |
| forced result plus EOTR, 73 positions | 6.24 s |
| post-EOTR boundary to first audible PCM | 0.56 s |
| total from SOTC | about 11.2 s |

The first target is not the 400 ms no-tool voice-to-voice goal. Tool use has an
additional autoregressive decision, an external operation, and result ingestion.
The target for this project is to remove avoidable fixed waits and make result
ingestion scale much better while preserving the existing no-tool path byte for
byte when the new features are disabled.

## Benchmark findings

The retained root is
`~/.local/state/nemotron-voicechat/traces/tool-call-latency/benchmark-v2`.
Important immutable report file hashes are:

- payload manifest: `edd6cb1844468dee145893b2f5f2b083dabf8a212db549787e9c11d21f18ba5d`;
- 40-cell primary report: `8d9e9a15430bebbe7bd434692401f6b8ad491aa82a5ae40769af3a015add6202`;
- primary analysis: `dae390f6abcdc7389d15589c7055c3d18becb617ef89a8fb3826df82b9f9e047`;
- schema report: `e07e394014aca0168c4e5a8349635942fc177cd543121ed4f0725804f9a352ef`;
- five-cell long-schema supplement: `bf8424ca6ec48fc67b24b8df8ee47aac5d8a1938cd4128f7059622da78fc90f5`;
- speech report: `d3cfe8a819e850b6e3334d4e7eea9c094ee7de453d23b252ec0708d61b8b4dc2`;
- carrier report: `eba95cdfa03ff3eadbd2fff18e3e180c44d39458b09c3197f7d730bfd8c393c2`;
- same-session red report: `f602cb802369760eccb966bdfb1b01c9f48bab6db45a8c5e2edaa552dede3865`;
- vLLM child ledger: `3c07f9f2b11341fc02d39298cf9ba227d53b01c468fedccdfdab848d5440e4aa`.

The primary 40 cells all passed. Result-phase wall time fits
`355.9 ms + 80.64 ms * hidden positions` with R-squared 0.9992. Mean result
phase times were 2.04, 4.32, 6.24, and 10.11 seconds at 20, 48, 72, and 120
wrapped result tokens. Nano took 4.33 of the mean 5.68 seconds across the mixed
result sizes; EarTTS took 1.10 seconds. Inside Nano, 99 percent of the interface
time was waiting for vLLM output. Input copy, append submission, and result
parsing were each negligible.

The child-process calibration resolves that wait. During the 73-position result
phase, vLLM scheduling averaged 0.103 ms, scheduler update averaged 0.030 ms, and
Nano model execution averaged 60.724 ms per position. EarTTS scheduling/update
were likewise below 0.2 ms while its model executor averaged 12.348 ms. Scheduler
or Python submission work is therefore not a material optimization target; the
GPU model executions are the cost.

Tool schema length has the same cost. The short schema emitted 27 call positions
in 2.265 seconds; the long schema emitted 37 in 3.081 seconds. Ten extra emitted
tokens cost 816 ms.

Ten qualified Pocket speech sessions all passed with exact input transcription.
Their 73-position result phase averaged 6.269 seconds versus 6.213 seconds in the
typed schema arm. Input modality therefore does not explain the slow result path.
Bursting identical speech increased call-emission time from 2.361 to 2.977
seconds, but left result injection unchanged at 6.20 versus 6.17 seconds. Fast
input is a contention and queueing diagnostic, not a latency optimization.

The ten-call single-session arm is intentionally red. Calls one through three
had stable 27/73-position phases, but call three already failed the response
content gate. Starting at call four, the model declined to call the tool and said
it had already been used in the turn. Content degradation therefore precedes
decision degradation. This is a separate long-context tool-use reliability
finding, not evidence of progressively slower completed inference. NVIDIA
documents that tool use degrades with repeated tools and recommends no more than
five tools per session. We still need to distinguish a prompt/turn-history defect
from that model limitation on a realistic mixed conversation.

The NVIDIA implementation at pinned Speech commit
`911ec674ab40f04302ef33672be4179f45a7310f` has the same serial architecture as
ours: every hidden function position advances Nano, EarTTS, codec, and RNNT. The
local patch adds correctness and lifecycle hardening; it did not create this
latency. NVIDIA's own README requires concise, ASCII-only, TTS-friendly tool
responses and explicitly warns that long tool responses delay speech.

One additional fixed cost is visible in both source and timing. The two-phase
worker starts the external tool in a thread and then unconditionally renders the
layer-one reminder before checking whether the future already completed. Even an
approximately 10 ms tool therefore waits about 2.08 seconds before phase two.

## Implementation order

### Step one: skip the on-hold utterance for fast tools

Add a prospectively configured fast-tool grace interval, initially 100 ms. Start
the tool immediately as today, wait up to the grace interval for its future, and
only render layer one if the future is still incomplete. A slow tool starts its
reminder at most 100 ms later than today; a fast tool proceeds directly to result
injection. Layers two and three, timeouts, authorization errors, and explicit
per-tool reminders retain their existing semantics once the grace expires.

Add structured evidence for tool-start, tool-complete, grace-expired, reminder
start/end, and phase-two start. Do not infer reminder playback from text events.
Before changing the policy, establish the actual baseline with a transport-level
reminder witness: correlate reminder start/end and the selected reminder variant
to PCM bytes written to the client channel, client receipt, and audible frames in
the retained WAV. The current model-step trace has no delivered or non-silent
audio in the measured reminder window, so it cannot prove that the reminder is
audible today. If the transport witness confirms that production currently drops
the reminder, treat that as a separate output defect rather than making the bug a
new compatibility requirement.

Record or deterministically pin the randomly selected reminder variant so its
length cannot confound slow-tool comparisons. The fast-tool promotion gate is
phase-two start within 250 ms p95 of result availability, down from the measured
1.92--2.16 seconds. A deliberately slow tool must retain the established,
transport-observed reminder behavior and complete audio. An explicit per-tool
"always acknowledge" policy bypasses the grace shortcut; its behavior is named
and capability-tested rather than silently changed. Record the race in which a
tool completes after grace expiry but before reminder playback begins.

This is the smallest, lowest-risk production fix and should land independently.

#### Step-one qualification result

The final candidate used immutable image
`sha256:bd4c4d340266e955843a7c900de78266a5501d73fb6503c94707a68648dc2658`
and canonical Speech patch
`6953c5952debfb405769f0cd2dd1e82b9de8d6e71869c64ec2b255c6b22e674f`.
All final reports below use benchmark driver
`de1e83caae0fc67ef473e9ecd866a25a4625a99e44f72eda232f348933748c07`:

- fast tools: 10/10 cells passed; result-available to phase-two p95 was
  0.957 ms; no reminder was rendered. Report SHA-256:
  `bd0b4f69df975b1165a531e1ce8fc29e56c6b69b9097abbf48ce393f016da058`;
- deliberately slow tool: grace expiry and reminder rendering were preserved;
  phase two began 0.937 ms after the result became available. Report SHA-256:
  `63735ca40607fee033060436079245bb4fab79d347963d4abb62a43a86bbd345`;
- always-acknowledge tool: the explicit override rendered the reminder despite
  fast completion. Report SHA-256:
  `26157d0b6beecbe1edd5f9fc3955f65b78b0b8e31397bfeedca52935fc569057`;
- safety: no-tool, call-emission interruption, and result-injection
  interruption each passed 3/3. Chained calls reproduced the existing one-call
  limitation at 0/3 and did not change. Report SHA-256:
  `ec5b0827e934ab9802e93723d06f5577d4a267dc400bc6719892863a1e8285f9`.

The reminder transport witness found a separate production defect: reminder PCM
is rendered internally but is not delivered to the client. Zero delivery is
recorded evidence, not a compatibility gate. The defect and its exact evidence
are retained in `docs/reminder-audio-transport-defect.md`.

### Step two: separate full tool data from concise model-visible context

Status: production-qualified.

Extend the versioned strict protocol, with explicit capability negotiation, with
an optional, explicitly client-authored `model_output` alongside the full
`output`. The server retains and correlates the full result, but injects
`model_output` when supplied. It must never silently summarize, truncate, or
select fields from an arbitrary result. Both values get separate hashes, byte
counts, and token counts in retained evidence. The existing full-output byte cap
continues to apply independently of the model-visible token cap; an older peer
that did not negotiate the field follows the existing single-output path.

Update our Pipecat tools to provide one short ordinary-English sentence in
`model_output`; keep their structured result in `output`. Preserve the existing
128-token hard limit and ASCII/TTS checks. The default path without
`model_output` remains exactly compatible.

The initial product target is approximately 20 wrapped model-visible tokens for
simple lookup tools. The unchanged benchmark shows this alone reduces phase two
from 6.24 seconds at 72 tokens to 2.04 seconds at 20 tokens, a 4.20-second saving.
Correctness is evaluated against the declared concise value, not hidden fields
from the full result. The production clock sentence realized 25 wrapped tokens;
the separately enforced injection safety limit remains 128 tokens.

The concrete boundary is host-only. Protocol v3 advertises
`function_output_model_output` and a client echoes `client_authored_v1` during
`session.update`. A negotiated `function_call_output` may then carry both
`output` and `model_output`. The Pipecat tool API represents them explicitly as
`VoicechatToolResult`; its context aggregator keeps the full result in normal
Pipecat history and carries the concise value in a call-ID-correlated sidecar.
An older server that does not advertise the capability receives only the full
result. A client that sends the optional field without negotiation is rejected.

The runtime validates the full result's existing 16 KiB bound independently.
It measures both wrapped token counts, but the existing 128-token injection hard
limit applies to the value that is actually injected. Empty or whitespace-only
`model_output` is rejected rather than falling back to the full value. The
runtime records, per call ID, separate hashes, byte counts, and token counts plus
the selected injection source. It does not summarize, truncate, or select fields.

`ExternalToolBridge` returns only the selected injection value to the unmodified
NVIDIA Speech pipeline. Thus Speech still consumes exactly one ordinary
`<TOOL_RESPONSE>[...]</TOOL_RESPONSE>` sequence, and its forced-token, EOTR, KV,
EarTTS/codec synchronization, and post-function BOS contracts do not change.
Any Speech patch in this step is a design violation.

Promotion reuses the complete Step-one benchmark and safety matrix, adds a
short-result interruption cell and a realized 72-to-near-20 token slope check,
then repeats independent response ASR and the browser-through-Pipecat real-tool
smoke. The first adopter is the clock tool: its structured clock object remains
the full `output`, while its short spoken sentence is `model_output`. Tool
authors must understand that fields retained only in `output` are intentionally
not visible to the model.

Before the first Step-two live run, the result-injection interruption outcome is
preregistered as follows. Reducing the injected result from about 72 wrapped
tokens to approximately 20 shrinks the injection interval from roughly six
seconds to roughly two seconds, so the client barge trigger may be transported
after EOTR rather than during forced injection. That timing change is
descriptive, not a waiver: the trigger must be sent before the result-applied
acknowledgement, the deferred-input queue must preserve it byte-for-byte, both
correlated responses must complete, and their input lifecycle acknowledgements
must remain correctly ordered. Qualification reports the observed phase without
requiring the shorter implementation to recreate an obsolete six-second
interruption window.

Pre-final-build attribution clarification: the preregistered byte/order facts
above remain unchanged, but “deferred-input queue” was too narrow a mechanism
name. Before function output is received, the application queue owns deferred
input. While `advance_function_output_recovery` runs inline, the server receive
loop cannot consume new WebSocket messages, so an input sent during the observed
`tool_response` phase is first held by the ordered transport and is consumed
only after recovery. New reports therefore label that trigger
`client_send_during_output_recovery` and retain the actual turn-start send,
function-output send, applied-ack, and input-lifecycle timestamps. This is a
serialization/ordering safety probe, not evidence that the model observed a
mid-injection interruption.

#### Step-two qualification result

The final behavioral candidate used immutable runtime image
`sha256:ec64f5f02f963e72e5225f84ccba6bab49587a61b806c287cf8e32d347ea9320`,
benchmark driver
`f6fa137a600d3629aeb63e8ef8e775d324c007cc17eccf42cb705eb4d8b2f355`,
and the unchanged Speech patch
`6953c5952debfb405769f0cd2dd1e82b9de8d6e71869c64ec2b255c6b22e674f`.
The frozen 40-cell quantitative matrix ran on the preceding Step-two image
`sha256:41a434c1127584d6f1144107ce5e6477a1366d3f9234424df0c8b01f9f5602fa`;
the final image reran every behavior affected by its later readiness and bridge
cleanup changes.

- The 40/40 primary cells passed text, audibility, tool-cycle, and exact
  position-count gates. Full results of 20, 48, 72, and 120 wrapped tokens all
  injected the same 20-token `model_output` in exactly 21 positions. Result
  recovery averaged 1.760 seconds, with p95 1.876 seconds and p99 2.225 seconds;
  mean end-to-end cell time was 14.845 seconds. Report and independently
  rederived analysis SHA-256 values are
  `004b4e2c77d5fc32c929d0a5a50a9f86dff3571609f98266598d6508a55532b6`
  and `428e6f5623b18ad5c08a9ae9c5c3e90d650d0f32fbb63e89ff99448c59498e06`.
- On the final image, the zero-delay reminder arm passed 10/10 with no reminder
  and 10.30 ms result-available-to-phase-two p95, still about 24 times inside
  the 250 ms gate. Its report SHA-256 is
  `3030e26606bbdf6b6ef94b43a65513801a2a19409471e6dc1b1439024ae76f48`.
  The value is higher than Step-one's 0.957 ms point result and is retained as
  observed image/load variance, not hidden.
- The delayed three-second tool preserved the reminder path and passed 1/1;
  report SHA-256
  `fbfde243b1b989386be0621200248069e8f9ec2085bfca80eecc02542c300d94`.
  Step two does not alter reminder selection or the always-acknowledge override,
  so the Step-one always-acknowledge artifact remains the binding qualification
  for that path and was not silently removed.
- Final-image safety passed 3/3 for no-tool, call-emission interruption, and
  result-recovery serialization. The known one-call chained-tool limitation
  remained 0/3 and non-promotable; report SHA-256
  `2feb72c92d711df4f1e40fa913e44809dfca5823fb67664a0a53116f6733bf5a`.
- The browser-through-Pipecat smoke passed a real clock tool, voice input, typed
  input, and audible output. It measured a 1.387-second warm voice-to-voice point
  sample, 0.587-second warm TTFB, and 0.831-second warm TTFA. The model-ready
  trace preceded the first accepted typed input by 12 ms, directly exercising
  the readiness-race fix. The retained model trace SHA-256 is
  `279adf895d1cc262c0e3c61307c6598bdb833d9a0523971d734933704513aa5c`.

The promoted immutable Nemotron English evaluator image
`sha256:5985421433c37aa558f0938bd11e8714b1bbf9a74b0ebf6a6226696582de482d`
classified and transcribed all 63 retained WAVs. The first complete ASR report
is intentionally retained red: 57/60 required rows passed because three
otherwise correct `Harbor` responses were rendered by the evaluator as the
British spelling `harbour`. Their audio classification was speech and their WER
was 0.2, inside the unchanged 0.5 bound; exact required-word spelling was the
only failed conjunct. Its file and self SHA-256 values are
`0a3d487162c7c1666ed5dfd212a338d4a0e912fb0728d3aafc3b6e2d2bae1ed5`
and `c12d5dac76c348c9e492870fafdb118397b8c627c749b3602a3886df9cb84e67`.

A reviewed `english_spelling_v2` normalizer maps only `harbour` to `harbor`,
and only for the required-word presence conjunct. It cannot affect WER or speech
classification. The successor report embeds the complete one-entry table and
both predecessor hashes. All transcripts, token IDs, classifications, and WERs
are byte-for-byte/number-for-number unchanged; only those three presence flags
and their dependent pass flags differ. The resulting 60/60 required green
artifact has file SHA-256
`2f4fc7ff05f9f6ba0233422bff61b41847b2a6053dd92a318c8d789342d93493`
and self SHA-256
`1e3314d9c4d991b68f70e9d962fdaf62033b1ef50f4daed8b84ccecb0ebf1f13`.
Future verification-word manifests should exclude words with common regional
English spelling variants; frozen manifests and their hash chains remain
unchanged.

### Step three: bypass EarTTS only on hidden forced-result positions

Prototype an opt-in extension of the already qualified idle-PAD bypass. It may
run only during phase-two forced function injection when the agent channel is
PAD, no acknowledgement token is being spoken, real audio capture is disabled,
the installed decoder-silence policy is attested, and a post-FC legacy BOS reset
is mandatory. Append the same codec silence tokens and advance codec history,
but do not issue a real EarTTS call or mutate its recurrent code/PKV state.

The controller must mark the prepared epoch dirty and require the existing
post-FC legacy abort/prefill. Any ambiguity falls back to the byte-identical
serial path. Phase one cannot use this shortcut while acknowledgement or reminder
tokens are being rendered.

An interrupted phase two must also leave the epoch dirty. With the bypass active,
a mid-injection barge-in is a mandatory safety cell, and the next EarTTS
activation must be evidence-verified as `legacy_fallback`, never
`prepared_reuse`. This explicitly covers NVIDIA's documented choppy-audio failure
mechanism on paths where a normal terminal BOS might not repair recurrent state.

The measured ceiling is about 1.2 seconds at 73 positions. Promotion requires
exact function token history and EOTR, exact decoder-silence bytes, the expected
single post-FC reset, full text/audio semantic agreement, independent response
ASR, and no new playout artifact.

### Step four: batch known forced Nano inputs

Build a diagnostic-only teacher-forced prototype for phase two. After the first
forced position, the remaining function tokens are already known. Construct the
causally shifted fused input embeddings for those forced tokens, append them as a
multi-position custom-input block to Nano, and request one natural prediction at
the end. EarTTS/codec/RNNT state handling remains separate and explicit. Do not
label this a byte-exact fused kernel: Nano has hybrid Mamba/KV state, and a
multi-position prefill may use numerically different kernels from repeated
single-position decode.

The prototype may advance only in prospectively bounded chunks. Choose a chunk
size whose preregistered p95 is below the existing 240 ms function-call interrupt
window, poll abort/live-perception state between chunks, and fall back or fail
closed without publishing a tool side effect if that bound is exceeded. It must
return and verify every intermediate agent/function head output that the current
sequential state machine consumes or suppresses; matching only the final EOTR and
post-EOTR continuation is insufficient. The proof must also establish that the
fused input for each hidden position depends only on the same prior forced token
and static silence-audio path as the sequential implementation.

Add a dedicated vLLM API rather than weakening the current assertion that normal
streaming returns exactly one generated token. Track custom-input positions
separately from sampled-output count. The prototype must prove:

- exact forced function IDs and function-history placement;
- exact EOTR ID on the first natural position;
- exact logical/physical request generation and position accounting;
- no stale output can mutate the active request;
- identical per-position raw/effective agent and function IDs, suppression
  decisions, abort points, and publication ordering;
- identical next 32 greedy agent/function token IDs after EOTR;
- bounded logit differences declared before results are observed;
- correct hybrid Mamba, KV-cache, RNNT, codec, and post-FC BOS state;
- deterministic fallback to the existing sequential path on any mismatch.

Run 20/48/72/120-token A/B cells in both orders. Promotion requires all current
structural, text, retained-audio, and Nemotron English ASR gates. The latency
target is a 72-token result phase p95 below 1.5 seconds; it is a target, not a
waiver if numerical or semantic equivalence fails.

### Step five: reduce autoregressive call-emission cost

Use the child vLLM ledger to separate scheduler/update time from model-executor
time. Submission is already approximately 0.2 ms, so queue plumbing is not the
leading hypothesis. Preserve concise public tool names and descriptions; the
schema benchmark establishes an 81.6 ms cost per extra emitted token.

Do not unconditionally teacher-force a canonical call after SOTC. SOTC proves
only that a function cycle began; it does not prove the exact tool name,
serialization, or EOTC that the target model would greedily emit. Instead, after
the forced-result batch is qualified, prototype target-verified speculative
decoding for the narrow case where exactly one no-argument tool is available.
Draft the canonical serialization, but accept each position only when Nano's own
target logits select the same greedy token; otherwise resume ordinary decoding
from the first mismatch. Apply the same bounded-chunk, per-position-output, and
abort polling contract as step four. Multi-tool or argument-bearing calls remain
autoregressive unless a separately reviewed constrained-decoding/trie design
proves equivalence.

This optimization changes the duration of the existing pre-publication
cancellation opportunity even if model tokens remain exact. Before promotion,
either preserve an equivalent decision window or obtain explicit product signoff
for the semantics change. The call-emission interruption test must have a
prospectively specified expectation for the speculative path; it cannot rely on
observing a protocol state that an effectively instantaneous draft consumes
before the client can react. Phase-two batching likewise shortens the weaker
timing opportunity for result-injection interruption and must report that change
without weakening the abort/state-safety gate.

### Step six: diagnose repeated-tool context reliability

Add a retained 15-turn mixed-conversation test with ordinary dialogue between
four distinct, semantically motivated tool turns. Record rendered prompt/history
boundaries, Nano positions, tool decisions, phase timings, and response text/audio.
Compare it with four fresh-session controls and NVIDIA's unchanged prompt/template
path. This distinguishes a missing turn delimiter or stale local FC state from
the model's documented repeated-tool limitation.

The test must fail when a required call is skipped; a refusal containing an old
result is not a pass. Record the first response-content degradation separately
from the first call-decision failure, because the retained ten-call arm shows
those at calls three and four respectively. Report response-time distributions
only for actually completed cycles and report decision reliability separately,
so skipped calls cannot make latency look better.

## Qualification and rollout

Each step lands behind a separate disabled-by-default flag and is promoted
separately. Default-disabled code must add no clocks, tensor reads, CUDA
synchronization, queue operations, or request-state mutation. Any enabled-path
failure falls back only where fallback is proven state-safe; otherwise it fails
closed and ends the session.

Before promotion, run:

- the 40-cell token-slope benchmark and the short/long schema arm;
- ten fresh typed and ten qualified-Pocket speech calls;
- the paced/burst carrier diagnostic;
- no-tool, chained-call, call-emission interruption, and result-injection
  interruption safety cells, triggered by observed protocol state rather than
  fixed sleeps;
- the 15-turn mixed-context regression;
- browser-through-Pipecat typed, speech, real-tool, and barge-in tests;
- independent Nemotron English ASR on every retained response WAV;
- exact EOTR/cardinality/post-FC-BOS, audio-tail, prebuffer, and clean-close gates.

The twelve frozen v2 safety baseline cells (three each for no-tool, chained,
call-emission interruption, and result-injection interruption) were not part of
the completed latency/mechanism sweeps. Run and retain them before the first
behavior-changing implementation so comparisons cannot be defined after a
candidate result is visible. Candidate-specific interruption tests then add the
explicit expectations described above.

No WER, semantic, margin, EOTR, timeout, or audio-integrity threshold is relaxed.
Report p50/p95/p99 for SOTC-to-call-publication, result-received-to-phase-two,
phase-two duration, EOTR-to-first-text, EOTR-to-first-audible, and full browser
voice-to-voice latency. A promotion artifact binds the exact immutable runtime,
Speech commit/patch, benchmark source, payloads, fixture, ASR evaluator, and all
source report hashes.

The rollout order is step one, step two, step three, then step four. Step five is
independent optimization after the result path is safe. Step six may identify a
separate correctness fix and must not be laundered into a latency success.
