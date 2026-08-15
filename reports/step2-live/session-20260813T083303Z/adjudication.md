# Step 2 live campaign post-campaign adjudication

Session: `session-20260813T083303Z`  
Campaign window: `2026-08-13T08:33:15.525Z` through `2026-08-13T08:36:50.747Z`  
Preregistered contract: `preregistration.md`, gates 1–2

## Verdict

- **Gate 1 — FAIL campaign-wide.** Three of five long-bin fixtures pass both
  preregistered conjuncts. `L1-2` is audibly truncated after “green dragons”
  (WER 0.6250), and `capture-off-L1` renders unrelated content (WER 0.9688).
- **Gate 2 — PASS.** No `no_text_since_bos_watchdog` request occurs in any
  retained metrics row. Every completed audible long-bin rendition closes via
  `decoded_silence_watchdog`, and each corresponding request row has
  `audible_seen=true` and `audio_delivered=false`.
- **I1 replacement check — PASS (supplemental, not gate 1).** The retained
  evaluator transcript is exactly “Two plus two is four.” The original response
  is cancelled and the accepted audio ends at the clear boundary.

## Retained server traces and audio derivation

The two newest model trace sessions matching the campaign's two Pipecat
connections were copied byte-for-byte into `server-traces/`:

| Pipecat artifact | Model trace session | File | SHA-256 |
|---|---|---|---|
| `driver/pipecat-playout-0001.jsonl` | `42a2717a-e863-4d2b-ae10-1de3fd3888f6` | `events.jsonl` | `5c5c1a5c6f477d0dd28b8098e1de4957e16f4430b9b45959e105d0584440617d` |
| `driver/pipecat-playout-0001.jsonl` | `42a2717a-e863-4d2b-ae10-1de3fd3888f6` | `output-22050-mono-s16le.pcm` | `c65a933880035373ef334194f923efca3868c9373c8481de99cdeea1deb90cc3` |
| `driver/pipecat-playout-0002.jsonl` | `a543b72d-a616-4582-b79e-69bcf6686159` | `events.jsonl` | `f3cfffc87e16b78508a18dfbf438000c5fb48ceb7c1ae43f45771f1cc1ce45eb` |
| `driver/pipecat-playout-0002.jsonl` | `a543b72d-a616-4582-b79e-69bcf6686159` | `output-22050-mono-s16le.pcm` | `f059f343643962fcfa96f37784a9d20b87940ef3ad99212cb6cfab63b1d1bfc0` |

The per-response WAVs are mono PCM16 at 22,050 Hz. They were reconstructed
from the ordered `response.output_audio.delta` payloads in the retained Pipecat
artifacts, grouped by explicit `response_id`. This is the exact downstream
accepted/rendered stream. The capture-on server PCM has eight additional
22,050-Hz frames (14,112 samples, 0.64 s) after the I1 interruption clear and
before the replacement response. Pipecat accepted none of those frames, so
they remain in the immutable server PCM but are excluded from both I1 response
WAVs. `rendered-audio/index.json` records every response boundary and identity.

## Gate 1: tail completeness and WER

Scoring uses the standard normalization implemented by
`tools/qualification/transcribe_sustained.py`: lowercase alphanumeric tokens
and Levenshtein word error rate. “Appear, in order” is checked as an ordered
token sequence; each passing tail below is also contiguous. The threshold is
WER <= 0.30.

| Fixture | Final 8 normalized script words | Tail | WER | WER gate | Gate 1 |
|---|---|---:|---:|---:|---:|
| `L1-1` | `every flag that flies above the northern gate` | PASS | 0.0625 (2/32) | PASS | **PASS** |
| `L1-2` | `every flag that flies above the northern gate` | FAIL | 0.6250 (20/32) | FAIL | **FAIL** |
| `L2-1` | `clouds and the exact minute the tide turned` | PASS | 0.0811 (3/37) | PASS | **PASS** |
| `L2-2` | `clouds and the exact minute the tide turned` | PASS | 0.0541 (2/37) | PASS | **PASS** |
| `capture-off-L1` | `every flag that flies above the northern gate` | FAIL | 0.9688 (31/32) | FAIL | **FAIL** |

### Retained evaluator transcripts

Evaluator image:
`pipecat-ai/nemotron-voicechat-asr-evaluator:english-0.6b` =
`sha256:5985421433c37aa558f0938bd11e8714b1bbf9a74b0ebf6a6226696582de482d`.
It ran with GPU access, host IPC, and `--network none`. Audio was linearly
resampled to 16 kHz exactly as in
`tools/qualification/transcribe_sustained.py:read_and_resample`. Full decoder
provenance and token IDs are retained in `asr/asr-results.json` (SHA-256
`3db13dfb41634ffa8b2e8677322f74febf86d9867c5fea1d4c3a41f8d0c3befc`).

- `L1-1`: “The quick brown fox jumped over the lazy dog while seventeen green
  dragons circled the ancient stone tower, counting every window, every door,
  and every flag that flies above the northern gate.”
- `L1-2`: “The quick brown fox jumped over the lazy dog while seventeen green
  dragons.”
- `L2-1`: “On Tuesday morning, the harbor filled with small boats carrying
  baskets of silverfish and the old lighthouse keeper wrote down the number of
  goals, the color of the clouds, and the exact minute the tide turned.”
- `L2-2`: “On Tuesday morning, the harbour filled with small boats carrying
  baskets of silver fish, and the old lighthouse keeper wrote down the number
  of goals, the color of the clouds, and the exact minute the tide turned.”
- `capture-off-L1`: “I cannot visualize or process that scene, but I can tell
  you the current UTC time if you would like.”
- `I1-replacement`: “Two plus two is four.” Supplemental reference WER: 0.0000
  (0/5).

## Gate 2: watchdog evidence

These are the only non-null `turn_state.agent_silence_watchdog.request_reason`
rows associated with the five long-bin response IDs. There are zero
`no_text_since_bos_watchdog` values in either complete Pipecat metrics trace.

| Fixture | Metrics frame | `request_reason` | `audible_seen` | `audio_delivered` | Gate 2 |
|---|---:|---|---:|---:|---:|
| `L1-1` | 608 | `decoded_silence_watchdog` | true | false | **PASS** |
| `L1-2` | 863 | `decoded_silence_watchdog` | true | false | **PASS** |
| `L2-1` | 1213 | `decoded_silence_watchdog` | true | false | **PASS** |
| `L2-2` | 1570 | `decoded_silence_watchdog` | true | false | **PASS** |
| `capture-off-L1` | 235 | `decoded_silence_watchdog` | true | false | **PASS** |

`L1-2` therefore fails audio completeness even though gate 2 passes: its EOS
was requested by decoded-silence detection, not by the prohibited no-text
watchdog.

## Per-fixture audio identities and verdicts

| Fixture / response | Duration | WAV SHA-256 | Transcript source | Verdict |
|---|---:|---|---|---|
| `S1-1` | 5.44 s | `ff292399b00156fbb66ca33cc64d97b021885b81d5e496b8cd4fb770f5c6ff01` | Pipecat text: “The red balloon floated over the quiet village square, a simple yet vivid image.” | Gate 1/2 N/A (short control) |
| `S1-2` | 5.60 s | `1aa08d8b0a94590178944ef183f5de74cf65ebd81c3e5ab249ca5c13b1139547` | Pipecat text: “The red balloon floated over the quiet village square, a simple yet vivid image.” | Gate 1/2 N/A (short control) |
| `L1-1` | 11.92 s | `6bf6a4f7f93618b10989394b6f93b2f56eafcb88693c3fecffc25423586e6d13` | Retained evaluator; transcript above | Gate 1 PASS; gate 2 PASS |
| `L1-2` | 5.44 s | `25587e74dda72b9add81ddec9922919c8e4d7585a3dc76c5fcd9e4a053757b74` | Retained evaluator; transcript above | **Gate 1 FAIL**; gate 2 PASS |
| `L2-1` | 12.48 s | `3468a3f25400691c16ada989e818e09e75c63cb795853f8e64de38bc693c4d20` | Retained evaluator; transcript above | Gate 1 PASS; gate 2 PASS |
| `L2-2` | 12.40 s | `13c2f30afdc1aa9cafbdae2c481ae12b7fbc636de2bfee4c795f84b7cf896bbe` | Retained evaluator; transcript above | Gate 1 PASS; gate 2 PASS |
| `I1-original` | 4.00 s | `04deabb87a40ef95c48a1a3680c407be78913afe5df271beab257e04aa0530b3` | Pipecat text: “The quick brown fox jumped over the lazy dog, while seventeen green dragons” | Gate 1/2 N/A (intentionally interrupted); cancelled at clear |
| `I1-replacement` | 2.56 s | `28493c1d60fedc5bf89b727816724732a45622de882a96664500eab6b86e8b3f` | Retained evaluator: “Two plus two is four.” | Supplemental replacement check PASS |
| `capture-off-L1` | 7.44 s | `16b52bfaa1d10bc2bcb9c8c182ffef26f036e109b2ae4cf7a03e65ff34856947` | Retained evaluator; transcript above | **Gate 1 FAIL**; gate 2 PASS |

`manifest.sha256` covers every regular file in the session directory other
than the manifest itself, which cannot self-hash by construction. It is sorted
by relative pathname and is verified with `sha256sum --check`.
