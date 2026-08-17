# Seal-0 revision 35 reason

Seal-0 revision 34 is invalidated prospectively because C2 positive scenario
`c2-pos-10` rendered sealed `benchmark-word` as ASR `benchmark word`. Its one
exact `get_benchmark_word` call, empty parsed arguments, and one injection all
passed. The fail-closed monitor stopped at record 34, and no r34 evidence is
eligible for Seal-1.

This failure exposed an implementation gap in the recorded compound-
normalization deviation rather than scenario invalidity. Revision 35 retains
the scenario and audio unchanged, but prospectively hashes comparator version
`punctuation_hyphenation_normalized_v2`: only a compound explicitly hyphenated
in one compared string is canonicalized across hyphenated, closed, and open
renderings. Arbitrary word boundaries are not removed. The clarification is
recorded in `reports/attnw8-gptq/seal-criterion-deviation.md` and is explicitly
subject to the results adversarial review.

The partial live reports, raw public-hook capture, and capture-stack log are
retained under `reports/attnw8-gptq/a0-r34/`. The fence-r2 stack was restored
before this revision. All prior invalidations remain in force; nothing failed
is resurrected.
