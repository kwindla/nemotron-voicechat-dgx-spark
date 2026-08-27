# Seal-0 revision 34 reason

Seal-0 revision 33 is invalidated prospectively because C2 positive scenario
`c2-pos-02` produced its expected normalized word sequence but made zero tool
calls and zero result injections. The fail-closed monitor stopped at record 26.
This is an exact tool-outcome failure under both the original and amended
criteria, not punctuation or hyphenation variation. No r33 evidence is
eligible for Seal-1.

The partial live reports, raw public-hook capture, and capture-stack log are
retained under `reports/attnw8-gptq/a0-r33/`. The fence-r2 stack was restored
before this revision. The entire failed scenario is replaced by independently
generated and audio-hashed `c2-pos-r34-02`, whose prompt explicitly names the
registered `get_benchmark_word` function. All prior invalidations remain in
force; nothing failed is resurrected.

The punctuation- and hyphenation-normalized expected-transcript criterion in
`reports/attnw8-gptq/seal-criterion-deviation.md` remains prospective from r26
and is explicitly subject to the results adversarial review. Tool name,
arguments, call count, and injection outcomes remain exact.
