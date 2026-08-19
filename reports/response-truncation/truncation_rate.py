"""Measure how often responses lose their spoken tail, on a live stack.

One session told us truncation happens; it cannot tell us how often. Any fix
has to beat a baseline, and with a two-in-seven observation a lucky run would
look like a cure. This drives repeated sessions, transcribes what each response
actually said with the pinned offline evaluator, and diffs that against what the
model generated.

Usage:  python3 truncation_rate.py --sessions 3 [--seconds 95]

Requires a bot already serving on 127.0.0.1:7860 with playout tracing enabled.
"""

from __future__ import annotations

import argparse
import collections
import difflib
import json
import pathlib
import re
import subprocess
import sys
import time
import wave

REPO = pathlib.Path(__file__).resolve().parents[2]
STATE = pathlib.Path.home() / ".local/state/nemotron-voicechat"
TRACES = STATE / "traces/model"
PLAYOUT = STATE / "playout-traces"
ASR_IMAGE = "pipecat-ai/nemotron-voicechat-asr-evaluator:english-0.6b"
FRAME_BYTES = 1764 * 2
COMPLETE_THRESHOLD = 95.0


def newest(directory: pathlib.Path, pattern: str = "*") -> pathlib.Path:
    entries = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not entries:
        raise SystemExit(f"nothing matching {pattern} under {directory}")
    return entries[-1]


def build_microphone_wav(destination: pathlib.Path) -> pathlib.Path:
    """Reuse real captured speech; synthetic prompts do not elicit long answers."""
    import audioop

    source = REPO / "reports/response-truncation/session-431ac6df"
    pcm_candidates = sorted(TRACES.glob("*/input-16000-mono-s16le.pcm"))
    if not pcm_candidates:
        raise SystemExit(f"no retained input audio under {TRACES}")
    # Prefer the session that is known to have produced long responses.
    known = TRACES / "431ac6df-573c-45af-8746-7cd9f66bd8c3/input-16000-mono-s16le.pcm"
    pcm_path = known if known.exists() else pcm_candidates[-1]
    upsampled, _ = audioop.ratecv(pcm_path.read_bytes(), 2, 1, 16000, 48000, None)
    with wave.open(str(destination), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(48000)
        handle.writeframes(upsampled)
    print(f"microphone fixture from {pcm_path.parent.name} ({source.name} session)")
    return destination


def responses_from_playout(trace: pathlib.Path) -> list[dict]:
    text = collections.defaultdict(list)
    created, status, frames = {}, {}, collections.Counter()
    for line in trace.open():
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
    return [
        {
            "response_id": rid,
            "text": "".join(text[rid]).strip(),
            "frames": frames[rid],
            "status": status.get(rid),
        }
        for rid in sorted(created, key=created.get)
    ]


def cut_clips(pcm: pathlib.Path, responses: list[dict], out: pathlib.Path) -> None:
    """Cut frame-exact clips; the PCM holds emitted frames in emission order."""
    data = pcm.read_bytes()
    offset = 0
    for index, response in enumerate(responses, 1):
        count = response["frames"]
        segment = data[offset * FRAME_BYTES : (offset + count) * FRAME_BYTES]
        offset += count
        with wave.open(str(out / f"response_{index}.wav"), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(22050)
            handle.writeframes(segment)
    expected = offset * FRAME_BYTES
    if abs(len(data) - expected) > FRAME_BYTES:
        print(f"  WARNING: pcm {len(data)}B vs {expected}B from frame counts", file=sys.stderr)


def transcribe(clip_dir: pathlib.Path) -> dict:
    script = REPO / "reports/response-truncation/session-431ac6df/transcribe_clips.py"
    (clip_dir / "transcribe_clips.py").write_bytes(script.read_bytes())
    subprocess.run(
        [
            "docker", "run", "--rm", "--gpus", "all", "--ipc", "host", "--network", "none",
            "-e", "CUBLAS_WORKSPACE_CONFIG=:4096:8",
            "-v", f"{clip_dir}:/clips",
            ASR_IMAGE, "python3", "-P", "/clips/transcribe_clips.py",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return json.loads((clip_dir / "transcripts.json").read_text())


def words(value: str) -> list[str]:
    return re.sub(r"[^a-z0-9 ]", "", value.lower().replace("-", " ")).split()


def classify(responses: list[dict], transcripts: dict) -> list[dict]:
    results = []
    for index, response in enumerate(responses, 1):
        generated = words(response["text"])
        spoken = words(transcripts[f"response_{index}.wav"]["transcript"])
        blocks = difflib.SequenceMatcher(None, generated, spoken).get_matching_blocks()
        matched = sum(block.size for block in blocks)
        tail = max((b.a + b.size for b in blocks if b.size), default=0)
        percent = 100 * matched / len(generated) if generated else 100.0
        results.append(
            {
                "index": index,
                "generated_words": len(generated),
                "spoken_words": len(spoken),
                "percent": round(percent, 1),
                "status": response["status"],
                # An interrupted response is expected to be short; only a
                # completed response that lost its tail counts as a defect.
                "truncated": percent <= COMPLETE_THRESHOLD and response["status"] != "cancelled",
                "lost": " ".join(generated[tail:]),
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", type=int, default=3)
    parser.add_argument("--seconds", type=float, default=95.0)
    parser.add_argument("--out", type=pathlib.Path,
                        default=REPO / "reports/response-truncation/rate-baseline")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    microphone = build_microphone_wav(args.out / "microphone.wav")
    with wave.open(str(microphone), "rb") as handle:
        fixture_seconds = handle.getnframes() / handle.getframerate()
    if args.seconds > fixture_seconds - 5:
        # Past the end of the fixture the microphone is silent and the pipeline
        # idle timeout closes the session, cutting the last response short and
        # contaminating the very measurement this makes.
        args.seconds = max(10.0, fixture_seconds - 5)
        print(f"replay clamped to {args.seconds:.1f}s ({fixture_seconds:.1f}s fixture)")
    probe = REPO / "reports/audio-crackle/replay_probe.py"
    every = []

    for session in range(1, args.sessions + 1):
        print(f"\n=== session {session}/{args.sessions} ===", flush=True)
        before = {p.name for p in TRACES.iterdir()}
        subprocess.run(
            [sys.executable, str(probe), str(microphone), str(args.seconds)],
            check=True, cwd=str(args.out),
        )
        time.sleep(4)  # let the server finish flushing its trace
        fresh = [p for p in TRACES.iterdir() if p.name not in before]
        if not fresh:
            print("  no new model trace; skipping", flush=True)
            continue
        trace_dir = max(fresh, key=lambda p: p.stat().st_mtime)
        playout = newest(PLAYOUT, "*.jsonl")
        responses = responses_from_playout(playout)
        if not responses:
            print("  no responses in this session; skipping", flush=True)
            continue
        clips = args.out / f"session-{session:02d}-{trace_dir.name[:8]}"
        clips.mkdir(exist_ok=True)
        cut_clips(trace_dir / "output-22050-mono-s16le.pcm", responses, clips)
        results = classify(responses, transcribe(clips))
        (clips / "classification.json").write_text(json.dumps(results, indent=1))
        every.extend(results)
        for row in results:
            if row["truncated"]:
                mark = "TRUNCATED"
            else:
                mark = "cancelled" if row["status"] == "cancelled" else "ok"
            print(f"  _{row['index']}: {row['percent']:5.1f}% "
                  f"({row['spoken_words']}/{row['generated_words']} words) {mark}", flush=True)
            if row["truncated"]:
                print(f"      lost: {row['lost'][:100]}", flush=True)

    completed = [r for r in every if r["status"] != "cancelled"]
    truncated = [r for r in completed if r["truncated"]]
    summary = {
        "sessions": args.sessions,
        "completed_responses": len(completed),
        "truncated_responses": len(truncated),
        "rate": round(len(truncated) / len(completed), 3) if completed else None,
        "long_responses": len([r for r in completed if r["generated_words"] >= 30]),
        "long_truncated": len([r for r in truncated if r["generated_words"] >= 30]),
    }
    payload = {"summary": summary, "responses": every}
    (args.out / "summary.json").write_text(json.dumps(payload, indent=1))
    print(f"\n=== baseline ===\n{json.dumps(summary, indent=1)}")


if __name__ == "__main__":
    main()
