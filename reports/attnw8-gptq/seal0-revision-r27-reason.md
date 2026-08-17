# Seal-0 revision 27 reason

Seal-0 revision 26 recorded and hashed the prospective transcript-criterion
deviation, but it was superseded before any baseline scenario replay. A
post-seal audit found that the amended A0 verifier's no-call branch did not
explicitly assert both an empty call list and zero injections, even though the
deviation requires injection outcomes and all categorical tool behavior to
remain exact.

Revision 27 hashes the corrected verifier. The transcript normalization and
its prospective boundary are unchanged; the no-call branch now explicitly
requires call count zero, empty call evidence, and injection count zero. No
revision-26 live or model-call evidence exists, and nothing from any earlier
invalidated baseline is eligible for Seal-1.
