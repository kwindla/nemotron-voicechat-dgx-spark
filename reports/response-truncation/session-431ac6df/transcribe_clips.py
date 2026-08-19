"""Independently transcribe each response clip with the pinned Nemotron ASR.

The claim under test is that two responses stopped decoding before their text
was spoken. That was inferred from characters-per-second, which is an argument
about rate, not about content. Transcribing the audio the model actually
emitted and diffing it against the text the model actually generated settles it
directly.
"""

import glob
import json
import sys
import wave

import numpy as np
import soxr

sys.path.insert(0, "/opt/voicechat-asr")
from nemotron_voicechat_asr_evaluator.backend import NemotronEnglishAsr

asr = NemotronEnglishAsr()
out = {}
for path in sorted(glob.glob("/clips/response_*.wav")):
    with wave.open(path, "rb") as w:
        rate = w.getframerate()
        pcm = w.readframes(w.getnframes())
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    if rate != 16000:
        audio = soxr.resample(audio, rate, 16000).astype(np.float32)
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    name = path.rsplit("/", 1)[-1]
    if not np.abs(audio).max():
        out[name] = {"transcript": "", "note": "all-zero audio"}
        continue
    result = asr.transcribe(audio)
    out[name] = {"transcript": result["transcript"], "seconds": round(len(audio) / 16000, 2)}
    print(name, "->", json.dumps(out[name]), flush=True)

with open("/clips/transcripts.json", "w") as fh:
    json.dump(out, fh, indent=1)
