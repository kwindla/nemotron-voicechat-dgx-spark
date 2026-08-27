"""Negative control: inject a known mid-speech stall, then run the bot.

A counter that reads zero is worth nothing until it has been shown to read
non-zero when the fault it looks for is actually present. The output transport
feeds the track serially, so sleeping inside one write withholds audio from the
track while its 10 ms clock keeps draining -- exactly a mid-stream underflow,
in order, with no reordering. The track holds at most one chunk-size of slack
(40 ms), so each injection should surface as roughly
(STALL_SECONDS - 0.04) / 0.01 ticks in a single run.
"""

import asyncio
import runpy
import sys

from pipecat.transports.smallwebrtc.transport import SmallWebRTCOutputTransport

STALL_SECONDS = 0.20
STALL_EVERY = 20  # writes; well inside a response, not at its edges

_writes = 0
_original = SmallWebRTCOutputTransport.write_audio_frame


async def write_audio_frame(self, frame):
    global _writes
    _writes += 1
    if _writes % STALL_EVERY == 0:
        await asyncio.sleep(STALL_SECONDS)
    return await _original(self, frame)


SmallWebRTCOutputTransport.write_audio_frame = write_audio_frame
sys.argv = ["nemotron_voicechat_pipecat.demo", "-t", "webrtc", "--host", "0.0.0.0", "--port", "7860"]
# alter_sys installs the module as sys.modules["__main__"], which is where the
# Pipecat runner looks for the bot factory.
runpy.run_module("nemotron_voicechat_pipecat.demo", run_name="__main__", alter_sys=True)
