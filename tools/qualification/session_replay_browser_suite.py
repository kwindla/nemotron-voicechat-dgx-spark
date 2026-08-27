#!/usr/bin/env python3
"""Replay a retained microphone session through Chromium and adjudicate its trace.

Capture and ASR are separate phases because the production model and the qualified
ASR evaluator cannot safely share this machine's GPU.  ``capture`` drives the live
Playground and retains browser/RTVI/model/Pipecat evidence.  After stopping the
production stack, ``asr`` runs the immutable evaluator over the browser-rendered
audio and closes the audio/text fidelity gate.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import wave
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
QUALIFICATION_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(QUALIFICATION_ROOT))

from response_completion_gate import gate_trace  # noqa: E402

DEFAULT_TRACE_ID = "c42c6701-f14d-4960-83fc-3c24bd269194"
DEFAULT_TRACE_ROOT = Path.home() / ".local/state/nemotron-voicechat/traces/model"
DEFAULT_PCM = DEFAULT_TRACE_ROOT / DEFAULT_TRACE_ID / "input-16000-mono-s16le.pcm"
DEFAULT_ASR_IMAGE = "pipecat-ai/nemotron-voicechat-asr-evaluator:english-0.6b"
INPUT_RATE = 16_000
MAX_AUDIO_TEXT_WER = 0.35
SENTENCE_BOUNDARY = re.compile(r"[.!?][\"')\]]?\s*$")
TIME_RESULT = re.compile(
    r"It is currently\s+(\d{1,2}):(\d{2})\s+(AM|PM)\s+([A-Z]{2,5})\.", re.I
)


def atomic_json(path: Path, value: Any) -> None:
    staging = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    staging.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(staging, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve().relative_to(root.resolve())),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def refresh_manifest_artifact(
    manifest: dict[str, Any], path: Path, root: Path
) -> None:
    """Replace one retained artifact record after its final rewrite."""

    relative = str(path.relative_to(root))
    matches = [
        index
        for index, record in enumerate(manifest.get("artifacts", []))
        if record.get("path") == relative
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"capture manifest must contain exactly one {relative!r} artifact"
        )
    manifest["artifacts"][matches[0]] = file_record(path, root)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, 1):
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"non-object JSONL at {path}:{line_number}")
            records.append(value)
    return records


def pcm_to_wav(source: Path, destination: Path, *, leading_silence_seconds: float) -> float:
    payload = source.read_bytes()
    if not payload or len(payload) % 2:
        raise ValueError("retained microphone PCM must be non-empty aligned PCM16")
    if not math.isfinite(leading_silence_seconds) or leading_silence_seconds < 0:
        raise ValueError("connection preroll must be finite and non-negative")
    preroll = b"\x00\x00" * round(INPUT_RATE * leading_silence_seconds)
    with wave.open(str(destination), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(INPUT_RATE)
        output.writeframes(preroll + payload)
    return (len(preroll) + len(payload)) / (2 * INPUT_RATE)


def write_rendered_wav(raw_path: Path, wav_path: Path, sample_rate: int) -> None:
    if sample_rate <= 0 or not raw_path.is_file() or raw_path.stat().st_size % 2:
        raise ValueError("browser rendered audio capture is malformed")
    with wave.open(str(wav_path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(raw_path.read_bytes())


def browser_probe_script() -> str:
    return r"""(() => {
      const NativePeerConnection = window.RTCPeerConnection;
      const observedChannels = new WeakSet();
      window.__voicechatReplayPC = null;
      window.__voicechatRtviTimedMessages = [];
      const observeChannel = (channel) => {
        if (observedChannels.has(channel)) return;
        observedChannels.add(channel);
        channel.addEventListener('message', (event) => {
          if (typeof event.data !== 'string') return;
          try {
            window.__voicechatRtviTimedMessages.push({
              receivedPerformanceMs: performance.now(),
              receivedWallTimeS: Date.now() / 1000,
              message: JSON.parse(event.data),
            });
          } catch (_error) {}
        });
      };
      window.RTCPeerConnection = class extends NativePeerConnection {
        constructor(...args) {
          super(...args);
          window.__voicechatReplayPC = this;
          this.addEventListener('datachannel', (event) => observeChannel(event.channel));
        }
        createDataChannel(...args) {
          const channel = super.createDataChannel(...args);
          observeChannel(channel);
          return channel;
        }
      };
    })();"""


def rendered_audio_capture_script() -> str:
    return r"""async () => {
      const pc = window.__voicechatReplayPC;
      if (!pc) throw new Error('replay peer connection is unavailable');
      const deadline = performance.now() + 10000;
      let track = null;
      while (performance.now() < deadline && !track) {
        track = pc.getReceivers().map((receiver) => receiver.track)
          .find((candidate) => candidate?.kind === 'audio');
        if (!track) await new Promise((resolve) => setTimeout(resolve, 20));
      }
      if (!track) throw new Error('remote audio track is unavailable');
      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      const context = new AudioContextClass({latencyHint: 'interactive'});
      const processorSource = `
        class ReplayPcmCapture extends AudioWorkletProcessor {
          constructor() {
            super(); this.sequence = 0; this.stopped = false;
            this.port.onmessage = (event) => {
              if (event.data?.kind === 'stop') {
                this.stopped = true;
                this.port.postMessage({kind: 'stopped', sequence: this.sequence});
              }
            };
          }
          process(inputs) {
            if (this.stopped) return false;
            const samples = inputs[0]?.[0] || new Float32Array();
            const pcm = new Int16Array(samples.length);
            for (let index = 0; index < samples.length; index++) {
              const value = Math.max(-1, Math.min(1, samples[index]));
              pcm[index] = value < 0 ? Math.round(value * 32768) : Math.round(value * 32767);
            }
            this.port.postMessage({kind: 'pcm', sequence: ++this.sequence,
              frame: currentFrame, pcm}, [pcm.buffer]);
            return true;
          }
        }
        registerProcessor('replay-pcm-capture', ReplayPcmCapture);
      `;
      const moduleUrl = URL.createObjectURL(new Blob([processorSource],
        {type: 'application/javascript'}));
      await context.audioWorklet.addModule(moduleUrl);
      URL.revokeObjectURL(moduleUrl);
      await context.resume();
      const source = context.createMediaStreamSource(new MediaStream([track]));
      const worklet = new AudioWorkletNode(context, 'replay-pcm-capture');
      const silent = context.createGain(); silent.gain.value = 0;
      let deliveredSequence = 0;
      let resolveStopped;
      const stopped = new Promise((resolve) => { resolveStopped = resolve; });
      worklet.port.onmessage = (event) => {
        if (event.data?.kind === 'stopped') { resolveStopped(event.data); return; }
        if (event.data?.kind !== 'pcm') return;
        if (event.data.sequence !== deliveredSequence + 1)
          throw new Error('rendered audio sequence is not contiguous');
        deliveredSequence = event.data.sequence;
        const bytes = new Uint8Array(event.data.pcm.buffer);
        let binary = '';
        for (let index = 0; index < bytes.length; index++)
          binary += String.fromCharCode(bytes[index]);
        window.__voicechatRenderedChunk({sequence: deliveredSequence,
          frame: event.data.frame, payload: btoa(binary)});
      };
      source.connect(worklet); worklet.connect(silent); silent.connect(context.destination);
      window.__voicechatReplayCapture = {
        sampleRate: context.sampleRate,
        stop: async () => {
          worklet.port.postMessage({kind: 'stop'});
          const result = await Promise.race([stopped,
            new Promise((_, reject) => setTimeout(
              () => reject(new Error('capture stop timeout')), 5000))]);
          source.disconnect(); worklet.disconnect(); silent.disconnect();
          await context.close(); return {...result, deliveredSequence};
        },
      };
      return {sampleRate: context.sampleRate, trackId: track.id};
    }"""


def _response_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    responses: list[dict[str, Any]] = []
    active: dict[str, Any] | None = None
    recognized_since_bos = ""
    for record_index, record in enumerate(records):
        if record.get("event") != "model_step":
            continue
        turn_state = record.get("turn_state") or {}
        boundary = turn_state.get("response_boundary") or {}
        step_user_text = str(record.get("user_text") or "").strip()
        if record.get("user_text_changed") is True and step_user_text:
            recognized_since_bos = step_user_text
        if boundary.get("event") == "start":
            active = {
                "ordinal": len(responses) + 1,
                "record_index": record_index,
                "bos_frame": record.get("frame"),
                "eos_frame": None,
                "completion_reason": None,
                "user_text": step_user_text,
                "recognized_user_text_before_bos": recognized_since_bos,
                "response_attribution": turn_state.get("response_attribution"),
                "assistant_text": "",
                "tail": [],
                "sentence_boundary": False,
                "i3_classification": None,
            }
            recognized_since_bos = ""
        if active is None:
            continue
        active["assistant_text"] += str(record.get("assistant_delta") or "")
        audio = record.get("model_output_audio") or record.get("output_audio") or {}
        active["tail"].append(
            {
                "frame": record.get("frame"),
                "agent_control": turn_state.get("agent_control"),
                "rms_dbfs": audio.get("rms_dbfs"),
                "assistant_delta": record.get("assistant_delta"),
            }
        )
        active["tail"] = active["tail"][-10:]
        if boundary.get("event") != "end":
            continue
        active["eos_frame"] = record.get("frame")
        active["completion_reason"] = boundary.get("eos_reason")
        text = active["assistant_text"].strip()
        active["assistant_text"] = text
        active["sentence_boundary"] = bool(SENTENCE_BOUNDARY.search(text))
        if not active["sentence_boundary"]:
            reason = active["completion_reason"]
            audible_tail = any(
                isinstance(item.get("rms_dbfs"), (int, float))
                and float(item["rms_dbfs"]) > -60.0
                for item in active["tail"][-5:]
            )
            if reason in {"no_text_since_bos_watchdog", "no_audio_since_bos_watchdog"}:
                active["i3_classification"] = (
                    "runtime_boundary_while_productive" if audible_tail else "model_silence"
                )
            elif reason == "decoded_silence_watchdog":
                active["i3_classification"] = "model_silence"
            else:
                active["i3_classification"] = "model_turn_taking"
        responses.append(active)
        active = None
    if active is not None:
        active["assistant_text"] = active["assistant_text"].strip()
        active["sentence_boundary"] = bool(
            SENTENCE_BOUNDARY.search(active["assistant_text"])
        )
        active["i3_classification"] = "session_terminated_open_response"
        responses.append(active)
    return responses


def _tool_results(records: list[dict[str, Any]]) -> list[str]:
    function_text = ""
    for record in records:
        if record.get("event") == "model_step":
            candidate = record.get("function_text")
            if isinstance(candidate, str) and len(candidate) >= len(function_text):
                function_text = candidate
    values = re.findall(
        r"\[INJECTED_RESPONSE\]\s*<TOOL_RESPONSE>\[(.*?)\]</TOOL_RESPONSE>",
        function_text,
    )
    return values


ONES = {
    0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
    6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
    11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen",
    15: "fifteen", 16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen",
}


def spoken_number(value: int) -> str:
    if value < 20:
        return ONES[value]
    tens = {20: "twenty", 30: "thirty", 40: "forty", 50: "fifty"}
    return tens[value - value % 10] + ("" if value % 10 == 0 else f" {ONES[value % 10]}")


def stale_time_variants(result: str) -> list[str]:
    match = TIME_RESULT.search(result)
    if match is None:
        return [result.casefold().rstrip(".")]
    hour, minute, meridiem, zone = match.groups()
    numeric = f"{int(hour)}:{minute} {meridiem} {zone}".casefold()
    spoken = f"{spoken_number(int(hour))} {spoken_number(int(minute))} {meridiem} {zone}".casefold()
    return [numeric, spoken]


def analyze_trace(trace_path: Path, *, expected_phase: str) -> dict[str, Any]:
    records = load_jsonl(trace_path)
    responses = _response_records(records)
    calls = [
        {"name": record.get("name"), "arguments": record.get("arguments"), "record_index": index}
        for index, record in enumerate(records)
        if record.get("event") == "function_call_requested"
    ]
    outputs = [
        (index, record)
        for index, record in enumerate(records)
        if record.get("event") == "function_call_output_received"
    ]
    results = _tool_results(records)
    sequence = [call["name"] for call in calls]
    expected_sequence = [
        "get_current_time", "set_timezone", "get_current_time"
    ]
    sequence_observed = sequence == expected_sequence and len(outputs) >= 3
    pacific_result = results[2] if sequence_observed and len(results) >= 3 else ""
    pacific_conversion_passed = bool(
        TIME_RESULT.search(pacific_result)
        and re.search(r"\b(?:PDT|PST)\b", pacific_result, re.I)
        and not re.search(r"\bUTC\b", pacific_result, re.I)
    )
    latest_output_index = outputs[2][0] if sequence_observed else len(records)
    post_latest = (
        [r for r in responses if r["record_index"] > latest_output_index]
        if pacific_conversion_passed
        else []
    )
    old_result = results[0] if results else ""
    variants = stale_time_variants(old_result) if old_result else []
    stale = [
        {"ordinal": response["ordinal"], "assistant_text": response["assistant_text"]}
        for response in post_latest
        if any(variant in response["assistant_text"].casefold() for variant in variants)
    ]
    stale_ordinals = {response["ordinal"] for response in stale}
    fresh = [
        {"ordinal": response["ordinal"], "assistant_text": response["assistant_text"]}
        for response in post_latest
        if response["ordinal"] not in stale_ordinals
    ]
    freshness_eligible = bool(
        sequence_observed and pacific_conversion_passed and post_latest
    )
    freshness = {
        "status": (
            "not_eligible"
            if not freshness_eligible
            else "stale_observed" if stale else "fresh"
        ),
        "eligibility": {
            "eligible": freshness_eligible,
            "reason": (
                None
                if freshness_eligible
                else (
                    "no Pacific-converted result from the directive sequence"
                    if not pacific_conversion_passed
                    else "no assistant response followed the Pacific-converted result"
                )
            ),
        },
        "old_result": old_result,
        "latest_result": pacific_result,
        "eligible_responses": len(post_latest),
        "per_response": [
            {
                "ordinal": response["ordinal"],
                "assistant_text": response["assistant_text"],
                "classification": (
                    "stale_superseded_result"
                    if response["ordinal"] in stale_ordinals
                    else "fresh_no_superseded_result"
                ),
            }
            for response in post_latest
        ],
        "stale_responses": stale,
        "fresh_responses": fresh,
        "stale_rate": round(len(stale) / len(post_latest), 4) if post_latest else None,
        "fresh_rate": round(len(fresh) / len(post_latest), 4) if post_latest else None,
        "observed_without_stale": not stale if freshness_eligible and post_latest else None,
    }
    call_windows: list[dict[str, Any]] = []
    for offset, response in enumerate(responses):
        next_index = (
            responses[offset + 1]["record_index"]
            if offset + 1 < len(responses)
            else len(records)
        )
        response_calls = [
            call
            for call in calls
            if response["record_index"] <= call["record_index"] < next_index
        ]
        call_windows.append(
            {
                "ordinal": response["ordinal"],
                "bos_frame": response["bos_frame"],
                "utterance": response["user_text"],
                "call_requested": bool(response_calls),
                "calls": [
                    {"name": call["name"], "arguments": call["arguments"]}
                    for call in response_calls
                ],
                "assistant_text": response["assistant_text"],
            }
        )
    utterance_turns = [turn for turn in call_windows if turn["utterance"]]
    called_turns = sum(1 for turn in utterance_turns if turn["call_requested"])
    attribution_failures = [
        response["ordinal"]
        for response in responses
        if response["recognized_user_text_before_bos"]
        and not response["user_text"]
        and not (
            isinstance(response.get("response_attribution"), dict)
            and response["response_attribution"].get("kind") == "post_tool_continuation"
        )
    ]
    runtime_i3 = [
        response["ordinal"]
        for response in responses
        if response.get("i3_classification") == "runtime_boundary_while_productive"
    ]
    completion = gate_trace(trace_path)
    lifecycle = [
        {key: record.get(key) for key in ("event", "reason", "monotonic_s") if key in record}
        for record in records
        if record.get("event") in {"terminal_session_claimed", "session_trace_stopped"}
    ]
    watchdog_audible = completion.get("watchdog_closures_while_audible")
    pre_reproduced = bool(
        completion.get("passed") is False
        and any(
            "response_open_at_eof" in error
            for error in completion.get("input_schema_errors", [])
        )
        and attribution_failures
    )
    hard_gates = {
        "completion": completion.get("passed") is True,
        "audible_watchdog": watchdog_audible == 0,
        "response_attribution": not attribution_failures,
        "i3_runtime": not runtime_i3,
    }
    post_passed = all(hard_gates.values())
    return {
        "schema": "nemotron_voicechat.session_replay_analysis.v2",
        "expected_phase": expected_phase,
        "trace": str(trace_path),
        "response_completion_gate": completion,
        "responses": responses,
        "response_attribution": {
            "failures": attribution_failures,
            "passed": not attribution_failures,
        },
        "tool_propensity": {
            "classification": "reported_metric_not_acceptance_gate",
            "expected_sequence": expected_sequence,
            "observed_sequence": sequence,
            "calls": calls,
            "results": results,
            "calls_elicited": len(calls),
            "utterance_turns": len(utterance_turns),
            "turns_with_calls": called_turns,
            "per_turn_call_rate": (
                round(called_turns / len(utterance_turns), 4) if utterance_turns else None
            ),
            "per_turn": call_windows,
            "sequence_observed": sequence_observed,
            "pacific_conversion_passed": pacific_conversion_passed,
        },
        "freshness": freshness,
        "freshness_classification": "reported_metric_not_acceptance_gate",
        "sentence_completion": {
            "runtime_caused_incomplete_ordinals": runtime_i3,
            "model_behavior_incomplete_ordinals": [
                response["ordinal"]
                for response in responses
                if not response["sentence_boundary"]
                and response.get("i3_classification")
                in {"model_silence", "model_turn_taking"}
            ],
            "passed_runtime_half": not runtime_i3,
        },
        "session_lifecycle": lifecycle,
        "pre_fix_reproduced": pre_reproduced,
        "hard_gates": hard_gates,
        "runtime_gates_passed": post_passed,
        "phase_expectation_passed": pre_reproduced if expected_phase == "pre" else post_passed,
    }


async def wait_http(url: str, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2):  # noqa: S310 - loopback qualification
                return
        except Exception as exc:  # noqa: BLE001 - retained in terminal failure
            last_error = exc
            await asyncio.sleep(0.25)
    raise RuntimeError(f"Playground did not become ready: {last_error!r}")


async def capture(args: argparse.Namespace) -> dict[str, Any]:
    from playwright.async_api import async_playwright

    if args.final_user_stop_count <= 0:
        raise ValueError("final user-stop count must be positive")
    if not math.isfinite(args.disconnect_lead_seconds) or args.disconnect_lead_seconds < 0:
        raise ValueError("disconnect lead must be finite and non-negative")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing stale replay output: {output}")
    output.mkdir(parents=True)
    input_pcm = args.input_pcm.expanduser().resolve()
    trace_root = args.trace_root.expanduser().resolve()
    before = {path.name for path in trace_root.iterdir() if path.is_dir()}
    input_wav = output / "retained-input-16000.wav"
    duration = pcm_to_wav(
        input_pcm,
        input_wav,
        leading_silence_seconds=args.connection_preroll_seconds,
    )
    browser_log: list[str] = []
    chunks_path = output / "rendered-bot-raw-remote-track.pcm"
    chunk_manifest_path = output / "rendered-bot-chunks.jsonl"
    rendered_stream = chunks_path.open("xb")
    chunk_manifest = chunk_manifest_path.open("x", encoding="utf-8")
    expected_sequence = 0
    capture_description: dict[str, Any] | None = None
    capture_stop: dict[str, Any] | None = None
    rtvi: list[dict[str, Any]] = []
    screenshot = output / "playground.png"
    await wait_http(args.url.rstrip("/") + "/")

    async def rendered_chunk(_source: Any, record: dict[str, Any]) -> None:
        nonlocal expected_sequence
        sequence = record.get("sequence")
        if sequence != expected_sequence + 1:
            raise ValueError(f"rendered audio chunk sequence drifted: {sequence}")
        payload = base64.b64decode(record.get("payload", ""), validate=True)
        if len(payload) % 2:
            raise ValueError("rendered audio chunk is not aligned PCM16")
        expected_sequence = sequence
        rendered_stream.write(payload)
        chunk_manifest.write(
            json.dumps(
                {
                    "sequence": sequence,
                    "frame": record.get("frame"),
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                },
                sort_keys=True,
            )
            + "\n"
        )

    started = time.time()
    async with async_playwright() as playwright:
        executable = args.chromium or playwright.chromium.executable_path
        if not Path(executable).is_file():
            raise RuntimeError(f"real Playwright Chromium is missing: {executable}")
        browser = await playwright.chromium.launch(
            executable_path=executable,
            headless=True,
            args=[
                "--no-sandbox",
                "--use-fake-device-for-media-stream",
                "--use-fake-ui-for-media-stream",
                f"--use-file-for-fake-audio-capture={input_wav}%noloop",
                "--autoplay-policy=no-user-gesture-required",
            ],
        )
        context = await browser.new_context(permissions=["microphone", "camera"])
        await context.add_init_script(browser_probe_script())
        page = await context.new_page()
        page.on("console", lambda message: browser_log.append(message.text))
        page.on("pageerror", lambda error: browser_log.append(f"pageerror: {error}"))
        await page.expose_binding("__voicechatRenderedChunk", rendered_chunk)
        try:
            await page.goto(args.url.rstrip("/") + "/", wait_until="load", timeout=30_000)
            connect = page.get_by_role("button", name="Connect", exact=True)
            disconnect = page.get_by_role("button", name="Disconnect", exact=True)
            for _ in range(6):
                await connect.click()
                try:
                    await disconnect.wait_for(timeout=5_000)
                    break
                except Exception:  # noqa: BLE001 - bounded upstream connect retry
                    await page.wait_for_timeout(300)
            else:
                raise RuntimeError(f"Playground failed to connect: {browser_log[-50:]}")
            capture_description = await page.evaluate(rendered_audio_capture_script())
            # The source replay's final utterance begins near 74 seconds, after
            # the explicit connection preroll. Terminate during its answer to
            # deterministically exercise I1.
            await asyncio.sleep(max(1.0, duration - args.disconnect_lead_seconds))
            user_deadline = time.monotonic() + args.final_response_timeout
            while time.monotonic() < user_deadline:
                user_stops = await page.evaluate(
                    "() => (window.__voicechatRtviTimedMessages || []).filter(x => "
                    "x.message?.type === 'user-stopped-speaking').length"
                )
                if user_stops >= args.final_user_stop_count:
                    break
                await asyncio.sleep(0.1)
            else:
                raise TimeoutError(
                    "retained replay did not deliver the expected final user turn"
                )
            starts_before = await page.evaluate(
                "() => (window.__voicechatRtviTimedMessages || []).filter(x => "
                "x.message?.type === 'bot-started-speaking').length"
            )
            deadline = time.monotonic() + args.final_response_timeout
            while time.monotonic() < deadline:
                starts = await page.evaluate(
                    "() => (window.__voicechatRtviTimedMessages || []).filter(x => "
                    "x.message?.type === 'bot-started-speaking').length"
                )
                stops = await page.evaluate(
                    "() => (window.__voicechatRtviTimedMessages || []).filter(x => "
                    "x.message?.type === 'bot-stopped-speaking').length"
                )
                if starts >= starts_before and starts > stops:
                    break
                await asyncio.sleep(0.1)
            else:
                raise TimeoutError("final replay response did not open before disconnect")
            await asyncio.sleep(args.disconnect_after_bos_seconds)
            rtvi = await page.evaluate("() => window.__voicechatRtviTimedMessages || []")
            await page.screenshot(path=str(screenshot), full_page=True)
            await disconnect.click()
            await connect.wait_for(timeout=10_000)
            await asyncio.sleep(1.0)
        finally:
            try:
                capture_stop = await page.evaluate(
                    "() => window.__voicechatReplayCapture?.stop?.() || null"
                )
            except Exception as exc:  # noqa: BLE001
                browser_log.append(f"capture-stop-error: {exc!r}")
            await context.close()
            await browser.close()
    rendered_stream.flush()
    os.fsync(rendered_stream.fileno())
    rendered_stream.close()
    chunk_manifest.flush()
    os.fsync(chunk_manifest.fileno())
    chunk_manifest.close()
    (output / "rtvi-messages.json").write_text(
        json.dumps(rtvi, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "browser.log").write_text("\n".join(browser_log) + "\n", encoding="utf-8")
    if not capture_description or not isinstance(capture_description.get("sampleRate"), int):
        raise RuntimeError("browser audio capture did not report its sample rate")
    rendered_wav = output / "rendered-bot-remote-track.wav"
    write_rendered_wav(chunks_path, rendered_wav, capture_description["sampleRate"])
    deadline = time.monotonic() + 20
    created: list[Path] = []
    while time.monotonic() < deadline:
        created = [
            path for path in trace_root.iterdir()
            if path.is_dir() and path.name not in before
        ]
        complete = [path for path in created if (path / "events.jsonl").is_file()]
        if len(complete) == 1 and any(
            record.get("event") == "session_trace_stopped"
            for record in load_jsonl(complete[0] / "events.jsonl")
        ):
            created = complete
            break
        await asyncio.sleep(0.25)
    if len(created) != 1:
        raise RuntimeError(f"replay did not produce exactly one completed model trace: {created}")
    model_dir = output / "model-trace"
    shutil.copytree(created[0], model_dir)
    analysis = analyze_trace(model_dir / "events.jsonl", expected_phase=args.expect)
    atomic_json(output / "analysis.json", analysis)
    pipecat_candidates = sorted(output.glob("pipecat-playout-*.jsonl"))
    if len(pipecat_candidates) != 1:
        raise RuntimeError(
            "replay requires exactly one Pipecat playout artifact; restart the bot with "
            f"NEMOTRON_VOICECHAT_PLAYOUT_TRACE={output}/pipecat-playout-{{connection}}.jsonl"
        )
    report = {
        "schema": "nemotron_voicechat.session_replay_browser_suite.v2",
        "phase": args.expect,
        "status": "captured_pending_asr",
        "started_wall_time_s": started,
        "completed_wall_time_s": time.time(),
        "source_trace_id": DEFAULT_TRACE_ID,
        "source_pcm": {
            "bytes": input_pcm.stat().st_size,
            "sha256": sha256_file(input_pcm),
            "unmodified_audio_seconds": round(input_pcm.stat().st_size / 2 / INPUT_RATE, 6),
        },
        "input_duration_seconds": round(duration, 6),
        "connection_preroll_seconds": args.connection_preroll_seconds,
        "browser": {
            "engine": "real_playwright_chromium",
            "executable": executable,
            "fake_microphone_flag": f"--use-file-for-fake-audio-capture={input_wav}%noloop",
            "capture": capture_description,
            "capture_stop": capture_stop,
            "rendered_chunks": expected_sequence,
        },
        "analysis": analysis,
        "asr": None,
        "passed": False,
    }
    atomic_json(output / "report.json", report)
    artifacts = [
        input_wav, chunks_path, chunk_manifest_path, rendered_wav,
        output / "rtvi-messages.json", output / "browser.log", screenshot,
        model_dir / "events.jsonl", pipecat_candidates[0], output / "analysis.json",
        output / "report.json",
    ]
    atomic_json(
        output / "capture-manifest.json",
        {"schema": 1, "artifacts": [file_record(path, output) for path in artifacts]},
    )
    if not analysis["phase_expectation_passed"]:
        raise RuntimeError(f"{args.expect}-fix replay expectation failed: {output}/analysis.json")
    return report


def normalize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").casefold())


def word_error_rate(reference: str, hypothesis: str) -> float:
    expected, actual = normalize(reference), normalize(hypothesis)
    if not expected:
        return 0.0 if not actual else 1.0
    previous = list(range(len(actual) + 1))
    for index, token in enumerate(expected, 1):
        current = [index]
        for offset, observed in enumerate(actual, 1):
            current.append(
                min(current[-1] + 1, previous[offset] + 1,
                    previous[offset - 1] + (token != observed))
            )
        previous = current
    return previous[-1] / len(expected)


def read_wav_resampled(path: Path, rate: int = INPUT_RATE) -> np.ndarray:
    with wave.open(str(path), "rb") as source:
        channels = source.getnchannels()
        width = source.getsampwidth()
        source_rate = source.getframerate()
        payload = source.readframes(source.getnframes())
    if width != 2:
        raise ValueError("ASR worker accepts PCM16 WAV only")
    values = np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        values = values.reshape(-1, channels).mean(axis=1)
    if source_rate != rate:
        count = round(values.size * rate / source_rate)
        values = np.interp(
            np.arange(count, dtype=np.float64) * source_rate / rate,
            np.arange(values.size, dtype=np.float64), values,
        ).astype(np.float32)
    return values


def asr_worker(run_dir: Path) -> dict[str, Any]:
    from nemotron_voicechat_asr_evaluator.backend import NemotronEnglishAsr

    analysis = json.loads((run_dir / "analysis.json").read_text(encoding="utf-8"))
    reference = " ".join(
        response["assistant_text"] for response in analysis["responses"]
        if response.get("assistant_text")
    )
    audio = read_wav_resampled(run_dir / "rendered-bot-remote-track.wav")
    evaluator = NemotronEnglishAsr()
    decoded = evaluator.transcribe(audio)
    wer = word_error_rate(reference, decoded["transcript"])
    result = {
        "schema": "nemotron_voicechat.session_replay_asr.v1",
        "reference": reference,
        "hypothesis": decoded["transcript"],
        "word_error_rate": round(wer, 4),
        "max_word_error_rate": MAX_AUDIO_TEXT_WER,
        "passed": bool(reference and decoded["transcript"] and wer <= MAX_AUDIO_TEXT_WER),
        "decode": decoded,
        "provenance": evaluator.provenance(),
    }
    atomic_json(run_dir / "asr.json", result)
    return result


def image_id(image: str) -> str:
    result = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", image],
        text=True, capture_output=True, check=False,
    )
    identity = result.stdout.strip()
    if result.returncode or not re.fullmatch(r"sha256:[0-9a-f]{64}", identity):
        raise RuntimeError(f"ASR evaluator image is unavailable: {image}")
    return identity


def run_asr(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.run_dir.expanduser().resolve()
    report_path = run_dir / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "captured_pending_asr" or report.get("asr") is not None:
        raise RuntimeError("replay is not in the captured_pending_asr state")
    identity = image_id(args.asr_image)
    subprocess.run(
        [str(REPO_ROOT / "container/audit-asr-evaluator.sh"), identity],
        cwd=REPO_ROOT, check=True,
    )
    command = [
        "docker", "run", "--rm", "--gpus", "all", "--ipc", "host",
        "--network", "none",
        "-e", f"VOICECHAT_ASR_EVALUATOR_IMAGE_ID={identity}",
        "-v", f"{REPO_ROOT}:/workspace/project:ro",
        "-v", f"{run_dir}:/qualification",
        identity, "python3", "-P",
        "/workspace/project/tools/qualification/session_replay_browser_suite.py",
        "--asr-worker", "/qualification",
    ]
    log_path = run_dir / "asr.log"
    with log_path.open("x", encoding="utf-8") as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    if completed.returncode:
        raise RuntimeError(f"qualified ASR evaluator failed: {log_path}")
    evaluated = json.loads((run_dir / "asr.json").read_text(encoding="utf-8"))
    if evaluated.get("provenance", {}).get("evaluator_image_id") != identity:
        raise RuntimeError("ASR evaluator provenance does not match the immutable image")
    report["asr"] = evaluated
    report["acceptance_gates"] = {
        **report["analysis"].get("hard_gates", {}),
        "asr_audio_text_fidelity": evaluated.get("passed") is True,
    }
    report["status"] = "completed" if evaluated.get("passed") else "failed"
    report["passed"] = bool(
        report["analysis"].get("phase_expectation_passed")
        and evaluated.get("passed")
    )
    atomic_json(report_path, report)
    capture_manifest_path = run_dir / "capture-manifest.json"
    manifest = json.loads(capture_manifest_path.read_text(encoding="utf-8"))
    refresh_manifest_artifact(manifest, report_path, run_dir)
    atomic_json(capture_manifest_path, manifest)
    manifest["asr_artifacts"] = [
        file_record(run_dir / "asr.json", run_dir),
        file_record(log_path, run_dir),
    ]
    atomic_json(run_dir / "manifest.json", manifest)
    if not report["passed"]:
        raise RuntimeError(f"replay ASR/final gate failed: {report_path}")
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--asr-worker", type=Path, help=argparse.SUPPRESS)
    subparsers = result.add_subparsers(dest="command")
    capture_parser = subparsers.add_parser("capture")
    capture_parser.add_argument("--output", type=Path, required=True)
    capture_parser.add_argument("--expect", choices=("pre", "post"), required=True)
    capture_parser.add_argument("--input-pcm", type=Path, default=DEFAULT_PCM)
    capture_parser.add_argument("--trace-root", type=Path, default=DEFAULT_TRACE_ROOT)
    capture_parser.add_argument("--url", default="http://127.0.0.1:7860/client")
    capture_parser.add_argument("--chromium")
    capture_parser.add_argument("--disconnect-lead-seconds", type=float, default=15.0)
    capture_parser.add_argument("--connection-preroll-seconds", type=float, default=20.0)
    capture_parser.add_argument("--disconnect-after-bos-seconds", type=float, default=0.05)
    capture_parser.add_argument("--final-response-timeout", type=float, default=45.0)
    capture_parser.add_argument("--final-user-stop-count", type=int, default=9)
    asr_parser = subparsers.add_parser("asr")
    asr_parser.add_argument("--run-dir", type=Path, required=True)
    asr_parser.add_argument("--asr-image", default=DEFAULT_ASR_IMAGE)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.asr_worker is not None:
        result = asr_worker(args.asr_worker)
    elif args.command == "capture":
        result = asyncio.run(capture(args))
    elif args.command == "asr":
        result = run_asr(args)
    else:
        parser().error("choose capture or asr")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
