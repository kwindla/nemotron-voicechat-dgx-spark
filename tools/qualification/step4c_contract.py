"""Immutable constants for the preregistered Step 4c incidence fixture."""

from __future__ import annotations

import hashlib

QUALIFICATION_FIXTURE = "step4c-l1"
QUALIFICATION_MODE = "step4c-eartts-incidence-v1"
SYSTEM_INSTRUCTION = (
    "Qualification latency fixture: when the user supplies a script, speak the "
    "supplied script aloud completely and exactly, then stop."
)
L1_TEXT = (
    "The quick brown fox jumps over the lazy dog while seventeen green dragons "
    "circle the ancient stone tower, counting every window, every door, and every "
    "flag that flies above the northern gate."
)
WARMUP_SECONDS = 10.0
DURATION_SECONDS = 120.0
DRAIN_SECONDS = 15.0
PACED_SOURCE_FRAME_SECONDS = 0.08
EXPECTED_SOURCE_FRAMES = round(DURATION_SECONDS / PACED_SOURCE_FRAME_SECONDS)

# The browser retains a monotonic connected-window measurement.  Permit only a
# one-millisecond clock-edge discrepancy when proving the duration plus drain.
BROWSER_CONNECTED_WINDOW_TOLERANCE_SECONDS = 0.001

# Chrome's file-backed fake microphone has been observed to turn exact-zero WAV
# samples into +/-1 PCM16 LSB after its WebRTC path.  Future browser campaigns
# must preregister this server-ingress predicate; it does not retroactively
# relax the retained v1 campaign's exact-zero condition.
BROWSER_ACOUSTIC_CONTRACT_ID = "chrome-fake-mic-webrtc-one-lsb-v1"
BROWSER_SOURCE_PCM = "exact-zero-mono-pcm16"
BROWSER_SERVER_INGRESS_MAX_ABS_PCM16 = 1
BROWSER_FIXTURE_LIMITATION = (
    "Chrome file-backed fake microphone/WebRTC may transform exact-zero source WAV "
    "into server-ingress PCM with absolute amplitude one PCM16 LSB."
)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


L1_SHA256 = sha256_text(L1_TEXT)
