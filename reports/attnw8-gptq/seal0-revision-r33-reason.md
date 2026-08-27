# Seal-0 revision 33 reason

Seal-0 revision 32 is invalidated prospectively because the first C2 positive
scenario, `c2-pos-01`, produced the expected normalized word sequence but made
zero tool calls and zero result injections. This is an exact tool-outcome
failure under both the original and amended criteria, not punctuation or
hyphenation variation. The fail-closed monitor stopped at record 25; no r32
evidence is eligible for Seal-1.

The partial live reports, raw public-hook capture, and capture-stack log are
retained under `reports/attnw8-gptq/a0-r32/`. The fence-r2 stack was restored
before this revision. The entire failed scenario is replaced by independently
generated and audio-hashed `c2-pos-r33-01`, whose prompt explicitly names the
registered benchmark-word function. All prior invalidations remain in force;
nothing failed is resurrected.

The punctuation- and hyphenation-normalized expected-transcript criterion in
`reports/attnw8-gptq/seal-criterion-deviation.md` remains prospective from r26
and is explicitly subject to the results adversarial review. Tool name,
arguments, call count, and injection outcomes remain exact.
