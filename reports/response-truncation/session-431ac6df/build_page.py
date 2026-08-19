"""Render the generated-vs-spoken comparison page next to the clips.

Written to be re-runnable after a machine restart: it reads only the retained
playout trace and the committed transcripts, so the page can always be rebuilt
from evidence rather than from a scratch directory.
"""

import collections
import difflib
import html
import json
import pathlib
import re

TRACE = pathlib.Path.home() / (
    ".local/state/nemotron-voicechat/playout-traces/pipecat-playout-r3-0002.jsonl"
)
HERE = pathlib.Path(__file__).parent
WATCHDOG_RESPONSES = {2, 3, 5}

text = collections.defaultdict(list)
created, status, frames = {}, {}, collections.Counter()
for line in TRACE.open():
    try:
        record = json.loads(line)
    except ValueError:
        continue
    kind, rid = record.get("type"), record.get("response_id")
    if kind == "response.output_text.delta":
        text[rid].append(record["delta"])
    elif kind == "response.created":
        created[rid] = record["client_received_monotonic_s"]
    elif kind == "response.done":
        status[rid] = record["status"]
    elif kind == "response.output_audio.delta":
        frames[rid] += 1

transcripts = json.loads((HERE / "transcripts.json").read_text())


def words(value: str) -> list[str]:
    return re.sub(r"[^a-z0-9 ]", "", value.lower().replace("-", " ")).split()


sections = [
    "<h1>Session 431ac6df &mdash; generated text vs audio actually emitted</h1>",
    "<p>Each clip is a frame-exact cut of the model's own output PCM. The transcript beside it comes "
    "from the pinned Nemotron ASR evaluator (<code>english-0.6b</code>, image-owned backend, offline), "
    "not from the voice model. Where the two disagree, the model generated text it never spoke.</p>",
]
for index, rid in enumerate(sorted(created, key=created.get), 1):
    generated = "".join(text[rid]).strip()
    spoken = transcripts[f"response_{index}.wav"]["transcript"]
    left, right = words(generated), words(spoken)
    blocks = difflib.SequenceMatcher(None, left, right).get_matching_blocks()
    matched = sum(block.size for block in blocks)
    tail = max((b.a + b.size for b in blocks if b.size), default=0)
    percent = 100 * matched / len(left) if left else 0
    cancelled = status.get(rid) == "cancelled"
    flags = []
    if index in WATCHDOG_RESPONSES:
        flags.append("decoded_silence_watchdog fired")
    if cancelled:
        flags.append("cancelled by user interruption")
    tone = "ok" if percent > 95 else ("warn" if cancelled else "bad")
    lost = " ".join(left[tail:])
    body = f"<span class='kept'>{html.escape(' '.join(left[:tail]))}</span>"
    if lost:
        body += f" <span class='lost'>{html.escape(lost)}</span>"
    sections.append(f"""
<div class="r {tone}">
  <h2>response_{index} &mdash; {frames[rid] * 0.08:.2f}s audio &mdash;
      <span class="pct">{percent:.0f}% of generated words spoken</span></h2>
  {'<p class="flag">' + ' &middot; '.join(flags) + '</p>' if flags else ''}
  <audio controls preload="none" src="response_{index}.wav"></audio>
  <p class="lbl">generated text, never-spoken tail highlighted</p>
  <blockquote>{body}</blockquote>
  <p class="lbl">independent ASR of the emitted audio</p>
  <blockquote class="asr">{html.escape(spoken) or "<i>(silence)</i>"}</blockquote>
</div>""")

STYLE = """<style>
body{font:16px/1.6 system-ui,sans-serif;max-width:880px;margin:2rem auto;padding:0 1rem;background:#111;color:#eee}
h1{font-size:1.35rem;line-height:1.3} h2{font-size:1rem;font-weight:600;margin:.2rem 0}
.r{border-left:4px solid #444;padding:.9rem 1.1rem;margin:1.3rem 0;background:#1a1a1a;border-radius:5px}
.r.bad{border-left-color:#e0533d} .r.ok{border-left-color:#3d8b5f} .r.warn{border-left-color:#c9922e}
.pct{font-variant-numeric:tabular-nums}
.r.bad .pct{color:#ff8a75} .r.ok .pct{color:#7fd6a4} .r.warn .pct{color:#e8b84b}
.flag{color:#e8b84b;font-size:.85rem;margin:.3rem 0}
.lbl{font-size:.78rem;text-transform:uppercase;letter-spacing:.05em;color:#777;margin:.8rem 0 .2rem}
blockquote{margin:0;padding-left:.9rem;border-left:2px solid #333;color:#ccc;font-size:.93rem}
blockquote.asr{color:#9fb8d0}
.kept{color:#ccc} .lost{background:#4a1d16;color:#ff9d88;padding:.05em .2em;border-radius:3px}
audio{width:100%;margin:.5rem 0}
</style>"""
(HERE / "index.html").write_text(
    f"<!doctype html><meta charset=utf-8><title>generated vs spoken</title>{STYLE}" + "".join(sections)
)
print(f"wrote {HERE / 'index.html'}")
