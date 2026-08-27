# attnW8-GPTQ seal-protocol criterion deviation

Status: **PROSPECTIVE DEVIATION — REQUIRES RESULTS ADVERSARIAL-REVIEW
ADJUDICATION**

Effective boundary: the first Seal-0 revision that hashes this document
(revision 26 or later). This deviation does not apply retroactively.
Every prior invalidation stands, no failed evidence is resurrected, and no partial baseline
from an earlier seal is eligible for Seal-1.

## Amended expected-transcript criterion

Expected and observed transcripts are accepted when their normalized word
sequences are equal. Normalization is deterministic:

1. Unicode NFKC normalization and Unicode case folding.
2. Remove Unicode punctuation characters, including terminal punctuation and
   hyphens, without inserting whitespace. This gives hyphenated and closed
   compounds the same representation (for example, `fresh-word` and
   `freshword`).
3. Collapse all whitespace and compare the resulting word sequence for exact
   equality.

### Prospective implementation clarification (revision 35 or later)

Seal-0 revision 34 exposed that ASR may render the same explicitly hyphenated
compound as an open compound: sealed `benchmark-word` was transcribed as
`benchmark word`. To implement the already recorded requirement that
hyphens/compounds be normalized, the first Seal-0 revision hashing this
clarification canonicalizes an explicitly hyphenated compound across its
hyphenated, closed, and open forms (for example, `benchmark-word`,
`benchmarkword`, and `benchmark word`). It does not merge arbitrary adjacent
words: the equivalence is enabled only by a hyphenated compound present in one
of the two compared strings. This clarification is prospective only. Revision
34 remains invalid, every earlier invalidation stands, and no failed evidence
is resurrected. This clarification is also explicitly subject to the
adversarial results review.

Only transcript comparison changes. Tool name, parsed arguments, call count,
injection count/outcome, fatal/error status, and all loop bounds remain exact
and fail closed.

## Rationale and evidence

The pre-Seal-1 curation loop exposed an acceptance-criterion defect rather
than invalid tool scenarios:

- `seal0-revision-r21-reason.md`: a correct one-call/one-injection scenario
  was invalidated because ASR added a terminal period.
- `seal0-revision-r22-reason.md`: regenerated audio for the replacement was
  invalidated in the opposite direction because ASR omitted the sealed
  terminal period.
- `seal0-revision-r24-reason.md`: a correct one-call/one-injection scenario
  was invalidated because ASR rendered `fresh-word` as `freshword`.

These outcomes consumed full baseline cycles while leaving the categorical
tool behavior unchanged. The amended criterion removes punctuation and
hyphenation nondeterminism from transcript qualification while retaining exact
word-sequence and tool-behavior requirements.

This deviation must be called out in the final campaign result and is
explicitly submitted to the planned adversarial results review for acceptance
or rejection.

## Prospective pooled-selection and similarity deviation (revision 36 or later)

Status: **PROSPECTIVE DEVIATION — REQUIRES RESULTS ADVERSARIAL-REVIEW
ADJUDICATION**

This is one prospective protocol deviation with two inseparable changes. It
first becomes effective in the Seal-0 revision that hashes this section. Every
prior invalidation stands, no failed or superseded evidence is resurrected,
and no earlier partial baseline is eligible for Seal-1.

### 1. Pooled scenario selection

Seal-0 seals an ordered pool of at least 40 whole, audio-hashed candidates,
including at least 20 positive-intent and 20 no-call-intent candidates. It
also seals this deterministic selection rule: run one baseline capture over
the entire pool; classify every candidate; then, preserving sealed pool order,
select the first 12 validated positive-intent candidates and the first 12
validated no-call-intent candidates. Seal-1 records that exact selected set.

A pool member that does not validate is retained with its transcript, score,
and exact categorical outcomes as propensity evidence. It does not invalidate
the pool or trigger scenario replacement. If fewer than 12 candidates validate
in either intent class, A0 records the shortfall and the campaign stops for
adjudication; there is no further one-scenario replacement loop.

For revision 36 the preregistered order is the existing fixture-manifest order
over all `a2` and `c2` records. The sealed pool contains 48 candidates: 24
positive-intent and 24 no-call-intent.

### 2. Transcript similarity criterion

Candidate audio SHA-256 remains exact. Tool call count, tool name, parsed
arguments, injection count, fatal/error status, and loop bounds remain exact.
The observed user transcript is retained verbatim and is accepted when its
deterministic normalized word-sequence similarity is at least `0.8`.

Normalization uses Unicode NFKC, case folding, punctuation removal, and the
sealed hyphenated/closed/open-compound canonicalization above. Similarity is
`1 - D/max(n,m)`, where `n` and `m` are the normalized token counts and `D` is
the deterministic token-sequence edit distance with insertion/deletion cost
`1` and substitution cost `1 - SequenceMatcher(token_a, token_b).ratio()`.
Thus single-token spelling variance such as `harbour`/`harbor` is tolerated,
while gross truncation such as sealed `give me a synonym` becoming observed
`y` remains below threshold and is recorded as a non-validating candidate.

### Rationale and adjudication evidence

`seal0-revision-r31-reason.md` records gross word-level truncation (`give me a
synonym` to `y`), while `seal0-revision-r32-reason.md` records benign
single-token ASR spelling variance (`harbor` to `harbour`). Treating both as
binary exact-match failures caused full baseline cycles to behave like
one-scenario discovery trials. A sealed pool measures those propensities in
one run, retains every outcome, and prevents adaptive replacement churn.

Both the pooled-selection rule and the `0.8` transcript-similarity criterion
are explicitly flagged for the results adversarial review to accept or reject.
