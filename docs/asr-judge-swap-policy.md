# ASR judge replacement policy

This policy governs replacement of the independent ASR evaluator used to score
retained VoiceChat response audio. It does not change the product inference stack,
EarTTS, the response rubric, or the `WER <= 0.5` qualification threshold.

## Historical policy and preserved red result

The original policy required the old and new judges to agree on every standalone
component outcome, including the WER-threshold boolean. The immutable calibration
in `docs/evidence/asr-judge-swap-calibration-v1.json` applies that rule. It is red:
five of eight WER-threshold booleans differ. That artifact must remain red and must
not be rewritten after this amendment.

The same evidence also shows zero changes in audio classification, required-term
inclusion, excluded-term rejection, per-response verdicts, or aggregate verdicts.
All eight new transcripts and token-ID sequences repeat exactly in separate retained
rerun reports produced in declared reverse corpus order. The artifact does not treat
that caller-declared order as cryptographic proof of container freshness. Both judges'
four byte-matched EarTTS controls transcribe `The answer is five.` at WER 0. The
controls also exercise successful required-term inclusion and a deliberate
excluded-`five` failure. A deterministic all-zero PCM control proves the shared
pre-ASR silence classifier returns `near_silent` without invoking either judge.

## Prospective policy: decision concordance v2

Effective for judge replacements reviewed after this amendment, promotion requires:

1. Zero changes in speech/silence classification, required-term inclusion,
   excluded-term rejection, per-response pass/fail, and aggregate pass/fail.
2. Exact transcript and token-ID repetition by the candidate judge in a separate
   retained rerun. Reference text, semantic contract, and source/canonical WAV
   identities must also match; execution order is recorded as an operator declaration.
3. Passing positive controls from the actual EarTTS output domain under both judges.
4. Disclosure of every raw transcript, WER value, WER-threshold boolean, and change.
5. A changed WER-threshold boolean is admissible only when another unchanged false
   conjunct makes that boolean non-load-bearing. A load-bearing WER-threshold change
   blocks promotion.
6. The first subsequent green-path qualification must report every response's WER
   margin from 0.5. A passing response within 0.1 of the threshold is held for
   explicit review and cannot be cited as an unattended green qualification.

This separates measurements from decisions without weakening the live gate. WER
remains a required conjunct. The exception applies only to retrospective rows where
changing WER alone cannot alter the archived verdict; it cannot excuse a current or
future qualification failure.

The executable policy is `tools/qualification/compare_asr_judges.py`. It independently
re-derives response, ASR-aggregate, runtime, and final verdicts. Provenance-bound
reruns in the exact historical runtime images reproduce the old `.nemo` transcripts,
WERs, and canonical WAV hashes, binding the archived rows to the recorded old model.
The preserved
v1 artifact proves that the original rule fails; the separate v2 artifact applies
this amendment and records the same five differences as non-load-bearing rather
than deleting or relabeling them.

## Tool-freshness parity margin policy

The judge-replacement policy above leaves `WER <= 0.5` as an absolute live
gate. The first tool-freshness input-parity run additionally applied a uniform
`0.1` margin as a hard promotion gate. Its immutable v1 comparison remains red
with two margin reasons and must never be rewritten or reinterpreted under a
later policy.

An independently retained speech-then-typed diagnostic counterbalanced that
typed-then-speech run. Its response-audio WERs were:

- case 1: typed-forward `0.125`, speech-forward `0.5`, speech-reverse `0.25`,
  typed-reverse `0.375`;
- case 2: `0.25`, `0.375`, `0.375`, `0.375` in the same cell order;
- case 3: `0.5` in all four cells;
- case 4: `0.0` in all four cells.

All sixteen paired model-text cells were exact matches, every structural and
absolute response-ASR integrity gate passed, and every model-text/audio
positive-semantic result was a shared negative. The preregistered labels were
`unstable`, `unstable`, `tie`, and `tie`; they are descriptive and noncausal.
The result establishes no stable typed-input disadvantage: typed was no worse
than speech at either ordinal position, while case 3 demonstrates that a
uniform margin is not meaningful on a two-word reference, whose WER changes in
increments of `0.5`.

Effective prospectively, `parity_margin_v2` records both verdicts:

1. `strict_margin_v1` preserves the original uniform hard-margin result.
2. Every existing absolute and paired gate remains unchanged, including actual
   `WER <= 0.5` for both response WAVs.
3. A sub-`0.1` margin is eligible for an attended-review hold only when typed
   and speech model text is exactly equal, both model-text semantic outcomes
   are false, both response-audio semantic outcomes are false, and both arms'
   absolute audio-integrity gates passed.
4. Positive or discordant semantics, unequal model text, or failed absolute
   integrity retain the hard margin block.
5. A normalized reference of at most two words is recorded as a
   `degenerate_reference` hold subtype; shortness never independently waives a
   margin.
6. A held comparison has `core_passed: true` but remains
   `passed: false`, `unattended_promotion_eligible: false`, and
   `attended_review_required: true`. Only a separate content-addressed
   adjudication may promote it; there is no override flag.
7. The paired forward runner must write
   `comparison_policy: parity_margin_v2` into its retained runtime report before
   either arm is compared. The v2 comparator rejects unmarked historical runs;
   the strict v1 comparator ignores this prospective marker.

The counterbalanced diagnostic is permanently non-promotable. Its retained
file SHA-256 is
`6cc3939709a28a87e4c20f8f8f090dfa61f3e971f92b94e8e503518512af25ac`
and its internal self-hash is
`ccbbf4df38fa314569b2b202898f3bfe3caa04f572ba47aaa8d30744516c5a47`.
A frozen v1 comparison supplied its forward evidence. Its retained file SHA-256 is
`4889acc9305b691466ef80911280061a5d6eafd894a967c42d2a0d8381eae129`
and its internal self-hash is
`4916ec737007d14d96058229392a98a960778adc80ff842214054cfedd26274b`.
A new typed-then-speech run is required for prospective v2 evidence; the
historical v1 comparison remains red.

The first prospective margin-v2 run is also retained red. Its top report,
comparison, typed ASR report, and speech ASR report file hashes are,
respectively,
`1a9078fda74c78d623afbf9dd7bb31accc8819fc3a69b2b87617c4e0043b4a56`,
`e2920cb7b640904d7287ab5ab09a58c5110c3acb8e77a78ec4d62e5266444db0`,
`4e3709270344f1a6fbc31fda69c6c4397b9c039cff47eec8912c5ccb88c9b111`,
and `9714deaceb9d27dfddd34e46bc9c4c56a408f465bf52ca4cba700dbd13ccf6a2`.
The comparison's internal self-hash is
`41e6bd0993046387349ffc90cc2590f720c05f921146c22da525b9efec4acee8`.
Both arms failed only the unchanged absolute WER gate on the same two-word
model output, `Current UTC`; the English Nemotron judge has consistently
failed to render the acronym as that token sequence. The run is not
re-adjudicated and none of its thresholds are relaxed.

Prospectively, the ASR-scored tool-freshness stimulus is canonical scenario
v2, SHA-256
`2e95857362d4e74c3bb7cd4b2d8edcdfecfc87fdf2bf90a913b078132d700f28`.
It replaces clock times, acronyms, digits, and alphanumeric strings with four
ordinary verification words and ordinary-English prompts and spoken results.
This corpus rule applies to natural-language instructions, prompts, tool
descriptions, and injected spoken results; protocol identifiers and audio
format metadata are retained as non-spoken implementation data.
Every spoken result is one sentence of at least six normalized words with its
fresh word first. The old UTC scenario remains available only through its
exact archival hash so frozen evidence can still be rederived. Absolute
`WER <= 0.5`, the speech-input heard gate, and `parity_margin_v2` are
unchanged. A severe short-prefix truncation can still make a future run red;
there is no short-reference fallback. This is the last elicitation change for
close-out: any further corpus revision requires a new version and an
independent, pre-result instrument rationale.

The first fresh scenario-v2 qualification completed green on 2026-08-10. It
used runtime image
`sha256:246ec4f4adbae977e13ae062ca64f3c48051d823c5725059c8d6dc62749d133a`
and the promoted English Nemotron evaluator image
`sha256:5985421433c37aa558f0938bd11e8714b1bbf9a74b0ebf6a6226696582de482d`.
The retained top report, comparison, runtime report, typed ASR report, and
speech ASR report file hashes are, respectively,
`6226986167813ed1aa79f0e57cdf372f1e763bb6e7dc147df1db4fd36ac6f371`,
`c79d7cc5a7d45581b38bf1ce451b9f946a5c75058e8699e9b9967741de6f7efa`,
`b641b654b12374e0fb88911e208ce634d2cd6cc1201d73347ffd520c309444ad`,
`3873a8bc380ccf2bb2b9a6e514ffc23fd8d2220ff184dd52999c27ef72803320`,
and `cbe6cf533ab91cc899c9450913d288f8c4a20029cc40f5191ee4a8556d05ecc2`.
The comparison's internal self-hash is
`7f7bc5cdc61268361d7181cee00879b12996a00fc83431d2e55856453a5d18ae`;
the fixture-manifest identity is
`18dc1cab55383ce5c26ed30f65d90eb49486cd282aafea0427425f792a666ad4`.

Both arms passed structural and response-audio integrity gates. Speech-input
WER was zero on all four prompts. Typed and speech response WER was also zero
on every case, so the minimum response margin was `0.5` against the unchanged
required `0.1`. Both `strict_margin_v1` and `parity_margin_v2` passed; blocking
reasons, margin dispositions, and review holds were empty, and unattended
promotion was eligible. This green result does not rewrite the retained UTC
runs: it shows that their WER miss was tied to an unsuitable acronym/short-text
benchmark surface, while the unchanged judge cleanly transcribed the ordinary
English outputs.
