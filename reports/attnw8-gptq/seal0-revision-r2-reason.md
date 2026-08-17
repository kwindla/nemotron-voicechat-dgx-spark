# Seal-0 revision 2 reason

The initial baseline capture window is invalidated in full. The live service
accepted the first A2 scenario, but no Nano replay payload was written because
the hash-sealed `install_public_nano_capture` wrapper was not installed by the
fence-r2 server entrypoint. The preceding host-Python attempt was also invalid:
it failed before WebSocket attachment because that interpreter lacks the
`websockets` package.

No Hessian, quantized weight, candidate artifact/output, threshold change, or
qualification measurement was produced. Fence-r2 was restored and proved
ready before this revision.

Revision 2 retains every partition, scenario, artifact, algorithm, threshold,
and implementation source from revision 1. It additionally freezes the
already-sealed wrapper installation entrypoint, the read-only repository mount,
the in-container capture root, and a 12,000-call capture ceiling. The complete
baseline-only A0/index stage restarts in new directories; no evidence from the
invalidated attempts may enter Seal-1.
