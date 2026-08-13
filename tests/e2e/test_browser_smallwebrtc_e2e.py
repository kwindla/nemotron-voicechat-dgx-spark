"""Full browser -> SmallWebRTC -> bot -> Voicechat WebSocket qualification."""

from __future__ import annotations

import asyncio
import base64
import copy
import hashlib
import json
import math
import os
import re
import socket
import sys
import time
import urllib.error
import urllib.request
import wave
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from playwright.async_api import async_playwright
from websockets.asyncio.server import serve

QUALIFICATION_TOOLS = Path(__file__).resolve().parents[2] / "tools" / "qualification"
sys.path.insert(0, str(QUALIFICATION_TOOLS))

from step2_live_fixture_driver import (
    RTVI_QUIESCENCE_MIN_WINDOW_MS,
    RTVI_QUIESCENCE_WINDOW_MS,
    RTVI_TRANSCRIPT_IDENTITY,
    RTVI_WALL_CLOCK_SKEW_TOLERANCE_S,
    STEP4C_SUMMARY_SCHEMA,
    SUMMARY_SCHEMA,
    FixtureSpec,
    Step2PathReservation,
    WaitCriteria,
    blank_fixture_result,
    canonical_plan_json,
    connection_playout_trace_path,
    correlate_response_lifecycles,
    fixture_marker,
    load_browser_artifact,
    parse_fixture_plan,
    parse_rtvi_quiescence_window_ms,
    parse_step4c_fixture_plan,
    plan_document,
    read_playout_trace,
    resolve_step2_paths,
    rtvi_message_diagnostics,
    rtvi_transcript_evidence,
    second_answer_gate_evidence,
    transcript_gate_evidence,
    validate_browser_record,
    validate_fixture_summary,
    validate_marker,
    validate_rtvi_message_clocks,
    validate_rtvi_quiescence_window_ms,
    validate_step4c_summary,
    wait_for_interruption_deadline,
    wait_for_playout_trace_publication,
)

CHROMIUM = os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE")


def _rtc_probe_script(peer_connection_slot: str) -> str:
    """Observe RTC stats and browser-received RTVI messages without altering them."""

    return f"""(() => {{
      const NativePeerConnection = window.RTCPeerConnection;
      const observedChannels = new WeakSet();
      window.__voicechatRtviMessages = [];
      window.__voicechatRtviTimedMessages = [];
      const observeChannel = (channel) => {{
        if (observedChannels.has(channel)) return;
        observedChannels.add(channel);
        channel.addEventListener("message", (event) => {{
          if (typeof event.data !== "string") return;
          try {{
            const message = JSON.parse(event.data);
            window.__voicechatRtviMessages.push(message);
            window.__voicechatRtviTimedMessages.push({{
              receivedPerformanceMs: performance.now(),
              receivedWallTimeS: Date.now() / 1000,
              message,
            }});
          }}
          catch (_error) {{}}
        }});
      }};
      window.RTCPeerConnection = class extends NativePeerConnection {{
        constructor(...args) {{
          super(...args);
          window[{json.dumps(peer_connection_slot)}] = this;
          this.addEventListener("datachannel", (event) => observeChannel(event.channel));
        }}
        createDataChannel(...args) {{
          const channel = super.createDataChannel(...args);
          observeChannel(channel);
          return channel;
        }}
      }};
    }})();"""


def _voice_latency_probe_script() -> str:
    """Measure browser microphone-to-playout speech latency on one clock."""

    return """(() => {
      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      const context = new AudioContextClass({latencyHint: 'interactive'});
      const state = window.__voicechatV2V = {
        schema: 1,
        method: 'browser_webaudio_pcm_rms',
        thresholdRms: 0.01,
        thresholdDbfs: -40,
        processorSamples: 256,
        sampleRate: context.sampleRate,
        outboundSpeechFirstMs: null,
        outboundSpeechLastMs: null,
        inboundSpeechFirstMs: null,
        latencyMs: null,
        maxOutboundRms: 0,
        maxInboundRms: 0,
        stopped: false,
      };
      window.__voicechatResetV2V = () => Object.assign(state, {
        outboundSpeechFirstMs: null,
        outboundSpeechLastMs: null,
        inboundSpeechFirstMs: null,
        latencyMs: null,
        maxOutboundRms: 0,
        maxInboundRms: 0,
        stopped: false,
      });
      const monitoredTracks = new Set();
      const monitor = (stream, direction) => {
        const track = stream?.getAudioTracks?.()[0];
        if (!track || monitoredTracks.has(track.id)) return;
        monitoredTracks.add(track.id);
        const source = context.createMediaStreamSource(new MediaStream([track]));
        const processor = context.createScriptProcessor(state.processorSamples, 1, 1);
        const silent = context.createGain();
        silent.gain.value = 0;
        processor.onaudioprocess = (event) => {
          if (state.stopped) return;
          const samples = event.inputBuffer.getChannelData(0);
          let sumSquares = 0;
          for (let index = 0; index < samples.length; index++)
            sumSquares += samples[index] * samples[index];
          const rms = Math.sqrt(sumSquares / samples.length);
          const now = performance.now();
          if (direction === 'outbound') {
            state.maxOutboundRms = Math.max(state.maxOutboundRms, rms);
            if (rms > state.thresholdRms) {
              if (state.outboundSpeechFirstMs === null) state.outboundSpeechFirstMs = now;
              state.outboundSpeechLastMs = now;
            }
          } else {
            state.maxInboundRms = Math.max(state.maxInboundRms, rms);
            if (state.outboundSpeechFirstMs !== null &&
                state.inboundSpeechFirstMs === null &&
                rms > state.thresholdRms) {
              state.inboundSpeechFirstMs = now;
              state.latencyMs = now - state.outboundSpeechLastMs;
            }
          }
        };
        source.connect(processor);
        processor.connect(silent);
        silent.connect(context.destination);
      };
      const nativeGetUserMedia = navigator.mediaDevices.getUserMedia.bind(
        navigator.mediaDevices
      );
      navigator.mediaDevices.getUserMedia = async (...args) => {
        const stream = await nativeGetUserMedia(...args);
        monitor(stream, 'outbound');
        return stream;
      };
      const findRemoteAudio = () => {
        context.resume();
        for (const element of document.querySelectorAll('audio'))
          if (element.srcObject) monitor(element.srcObject, 'inbound');
      };
      setInterval(findRemoteAudio, 20);
    })();"""


def _audio_worklet_playout_capture_script(peer_connection_slot: str) -> str:
    """Batch a parallel render tap without touching the served audio element."""

    return f"""async () => {{
      const EXPORT_CHUNK_RECORDS = 128;
      const MAX_PENDING_BINDING_WRITES = 16;
      const exportBuffer = [];
      const pendingBindingWrites = new Set();
      let exportFailure = null;
      let exportedChunks = 0;
      let exportedRecords = 0;
      let stopped = false;
      const boundedWait = async (promise, label) => {{
        let timer;
        try {{
          return await Promise.race([
            promise,
            new Promise((_, reject) => {{
              timer = setTimeout(() => reject(new Error(`${{label}} timed out`)), 5000);
            }}),
          ]);
        }} finally {{
          clearTimeout(timer);
        }}
      }};
      const flushExport = () => {{
        while (exportBuffer.length) {{
          if (pendingBindingWrites.size >= MAX_PENDING_BINDING_WRITES)
            throw new Error('browser playout binding write bound exceeded');
          const chunk = exportBuffer.splice(0, EXPORT_CHUNK_RECORDS);
          exportedChunks++;
          exportedRecords += chunk.length;
          const pending = Promise.resolve(window.__voicechatEmitPlayoutTraceBatch(chunk))
            .catch((error) => {{ exportFailure ||= error; }})
            .finally(() => pendingBindingWrites.delete(pending));
          pendingBindingWrites.add(pending);
        }}
      }};
      const awaitBindingWrites = async () => {{
        await boundedWait(Promise.all([...pendingBindingWrites]), 'binding writes');
        if (exportFailure) throw exportFailure;
      }};
      const emit = (record) => {{
        exportBuffer.push({{
          trace_schema: 'nemotron_voicechat.browser_playout.v1',
          ...record,
        }});
        if (exportBuffer.length >= EXPORT_CHUNK_RECORDS) flushExport();
      }};
      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      if (!AudioContextClass || !window.AudioWorkletNode) {{
        emit({{type: 'browser.playout.capability', audio_worklet: false}});
        flushExport();
        await awaitBindingWrites();
        return;
      }}
      const processorSource = `
        class VoicechatPlayoutCapture extends AudioWorkletProcessor {{
          constructor(options) {{
            super();
            this.ordinal = 0;
            this.underrunCount = 0;
            this.audible = false;
            this.batch = [];
            this.captureStopped = false;
            this.workletId = options.processorOptions.workletId;
            this.port.onmessage = (event) => {{
              if (event.data?.kind !== 'flush') return;
              this.captureStopped = true;
              this.flush();
              this.port.postMessage({{
                kind: 'flush-ack',
                flushId: event.data.flushId,
                workletId: this.workletId,
                finalOrdinal: this.ordinal,
              }});
            }};
          }}
          flush() {{
            if (!this.batch.length) return;
            this.port.postMessage({{kind: 'render-batch', records: this.batch}});
            this.batch = [];
          }}
          process(inputs) {{
            if (this.captureStopped) return false;
            const channels = inputs[0] || [];
            const samples = channels[0] || [];
            let sumSquares = 0;
            let nonzero = false;
            for (let index = 0; index < samples.length; index++) {{
              const sample = samples[index];
              sumSquares += sample * sample;
              if (sample !== 0) nonzero = true;
            }}
            const rms = samples.length ? Math.sqrt(sumSquares / samples.length) : 0;
            const underrun = !samples.length || !nonzero;
            if (underrun) this.underrunCount++;
            const audible = rms > 0.01;
            this.batch.push({{
              ordinal: ++this.ordinal,
              audioContextTimeS: currentTime,
              audioContextFrame: currentFrame,
              sampleCount: samples.length,
              channelCount: channels.length,
              rms,
              underrun,
              underrunCount: this.underrunCount,
              onset: audible && !this.audible,
            }});
            if (this.batch.length >= 64) this.flush();
            this.audible = audible;
            return true;
          }}
        }}
        registerProcessor('voicechat-playout-capture', VoicechatPlayoutCapture);
      `;
      const moduleUrl = URL.createObjectURL(new Blob([processorSource], {{
        type: 'application/javascript',
      }}));
      const context = new AudioContextClass({{latencyHint: 'interactive'}});
      await context.audioWorklet.addModule(moduleUrl);
      URL.revokeObjectURL(moduleUrl);
      await context.resume();
      const clockMapping = (phase) => {{
        const output = context.getOutputTimestamp?.() || {{}};
        emit({{
          type: 'browser.clock_mapping',
          phase,
          main_thread_performance_ms: performance.now(),
          audio_context_time_s: context.currentTime,
          output_context_time_s: output.contextTime ?? null,
          output_performance_time_ms: output.performanceTime ?? null,
        }});
      }};
      const monitored = new Set();
      const readers = [];
      const nodes = [];
      const worklets = [];
      let workletSequence = 0;
      const monitor = (track) => {{
        if (!track || track.kind !== 'audio' || monitored.has(track.id)) return;
        monitored.add(track.id);
        const source = context.createMediaStreamSource(new MediaStream([track]));
        const workletId = `worklet-${{++workletSequence}}`;
        const worklet = new AudioWorkletNode(context, 'voicechat-playout-capture', {{
          processorOptions: {{workletId}},
        }});
        let resolveFlush;
        let rejectFlush;
        const workletState = {{
          worklet,
          workletId,
          deliveredLastOrdinal: 0,
          finalOrdinal: null,
          flushPromise: new Promise((resolve, reject) => {{
            resolveFlush = resolve;
            rejectFlush = reject;
          }}),
        }};
        worklets.push(workletState);
        emit({{type: 'browser.audio_worklet.registered', worklet_id: workletId}});
        const silent = context.createGain();
        silent.gain.value = 0;
        worklet.port.onmessage = (event) => {{
          if (event.data?.kind === 'flush-ack') {{
            workletState.finalOrdinal = event.data.finalOrdinal;
            emit({{
              type: 'browser.audio_worklet.flush_ack',
              worklet_id: workletId,
              final_ordinal: event.data.finalOrdinal,
              delivered_last_ordinal: workletState.deliveredLastOrdinal,
            }});
            if (workletState.deliveredLastOrdinal !== event.data.finalOrdinal) {{
              rejectFlush(new Error(
                `${{workletId}} delivered ${{workletState.deliveredLastOrdinal}} ` +
                `but acknowledged ${{event.data.finalOrdinal}}`
              ));
            }} else {{
              resolveFlush(event.data.finalOrdinal);
            }}
            return;
          }}
          if (event.data?.kind !== 'render-batch') return;
          const mainThreadReceivedMs = performance.now();
          for (const item of event.data.records) {{
            if (item.ordinal !== workletState.deliveredLastOrdinal + 1) {{
              rejectFlush(new Error(`${{workletId}} render ordinal is not contiguous`));
              return;
            }}
            workletState.deliveredLastOrdinal = item.ordinal;
            const base = {{
              worklet_id: workletId,
              main_thread_received_performance_ms: mainThreadReceivedMs,
              render_audio_context_time_s: item.audioContextTimeS,
              render_audio_context_frame: item.audioContextFrame,
              ordinal: item.ordinal,
              sample_count: item.sampleCount,
              channel_count: item.channelCount,
              rms: item.rms,
              underrun: item.underrun,
              underrun_count: item.underrunCount,
            }};
            emit({{type: 'browser.audio_worklet.dequeue', ...base}});
            if (item.onset)
              emit({{type: 'browser.audio_worklet.onset', ...base}});
          }}
        }};
        source.connect(worklet);
        worklet.connect(silent);
        silent.connect(context.destination);
        nodes.push(source, worklet, silent);

        if ('MediaStreamTrackProcessor' in window) {{
          const clonedTrack = track.clone();
          const processor = new MediaStreamTrackProcessor({{track: clonedTrack}});
          const reader = processor.readable.getReader();
          readers.push({{reader, clonedTrack}});
          void (async () => {{
            let ordinal = 0;
            while (true) {{
              const {{done, value}} = await reader.read();
              if (done) break;
              emit({{
                type: 'browser.remote_audio.enqueue',
                main_thread_observed_performance_ms: performance.now(),
                ordinal: ++ordinal,
                media_timestamp_us: value.timestamp,
                media_duration_us: value.duration,
                sample_count: value.numberOfFrames,
                sample_rate_hz: value.sampleRate,
                channel_count: value.numberOfChannels,
              }});
              value.close();
            }}
          }})();
        }} else {{
          emit({{type: 'browser.playout.capability',
            audio_worklet: true, media_stream_track_processor: false}});
        }}
      }};
      const scan = () => {{
        const pc = window[{json.dumps(peer_connection_slot)}];
        if (pc) for (const receiver of pc.getReceivers()) monitor(receiver.track);
        for (const element of document.querySelectorAll('audio'))
          for (const track of element.srcObject?.getAudioTracks?.() || []) monitor(track);
      }};
      const timer = setInterval(scan, 20);
      window.__voicechatStopPlayoutCapture = async () => {{
        if (stopped) return {{exportedChunks, exportedRecords}};
        stopped = true;
        clearInterval(timer);
        const flushId = `flush-${{performance.now()}}`;
        for (const state of worklets)
          state.worklet.port.postMessage({{kind: 'flush', flushId}});
        const finalOrdinals = await boundedWait(
          Promise.all(worklets.map((state) => state.flushPromise)),
          'worklet flush acknowledgements',
        );
        for (const {{reader, clonedTrack}} of readers) {{
          await reader.cancel();
          clonedTrack.stop();
        }}
        clockMapping('teardown');
        await awaitBindingWrites();
        flushExport();
        await awaitBindingWrites();
        for (const node of nodes) node.disconnect();
        await context.close();
        return {{
          exported_chunks: exportedChunks,
          exported_records: exportedRecords,
          export_chunk_records: EXPORT_CHUNK_RECORDS,
          worklet_batch_records: 64,
          worklet_final_ordinals: Object.fromEntries(
            worklets.map((state, index) => [state.workletId, finalOrdinals[index]])
          ),
          max_pending_binding_writes: MAX_PENDING_BINDING_WRITES,
        }};
      }};
      emit({{
        type: 'browser.playout.capability',
        audio_worklet: true,
        media_stream_track_processor: 'MediaStreamTrackProcessor' in window,
        zero_pcm_classification: 'underrun_candidate',
        audio_context_sample_rate_hz: context.sampleRate,
        render_path: 'parallel_silent_audio_context_tap',
        audible_element_render_path_observed: false,
        render_clock: 'render_audio_context_time_s/render_audio_context_frame',
        main_thread_clock: 'main_thread_received_performance_ms',
        caveat: 'parallel tap scheduling/resampling may differ from the audible element',
        export_chunk_records: EXPORT_CHUNK_RECORDS,
        worklet_batch_records: 64,
        max_pending_binding_writes: MAX_PENDING_BINDING_WRITES,
      }});
      clockMapping('startup');
      scan();
    }}"""


class _BrowserPlayoutTraceSink:
    """Persist bounded page batches and expose capture-overhead counters."""

    def __init__(
        self,
        path: Path,
        *,
        capture_enabled: bool = True,
        reservation: Step2PathReservation | None = None,
    ):
        if reservation is None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._stream = path.open("x", encoding="utf-8")
        else:
            descriptor = reservation.open_child(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            self._stream = os.fdopen(descriptor, "w", encoding="utf-8")
        self.path = path
        self.capture_enabled = capture_enabled
        self._chunks = 0
        self._records = 0
        self._marker_records = 0
        self._handler_seconds = 0.0
        self._closed = False
        self._registered_worklets: set[str] = set()
        self._last_ordinals: dict[str, int] = {}
        self._acknowledged_ordinals: dict[str, int] = {}

    def emit_batch(self, _source: Any, records: list[dict[str, Any]]) -> None:
        started = time.perf_counter()
        if not isinstance(records, list) or not 0 < len(records) <= 128:
            raise TypeError("browser playout trace batch must contain 1..128 records")
        received = time.monotonic()
        lines = []
        for record in records:
            if not isinstance(record, dict):
                raise TypeError("browser playout trace record must be an object")
            validate_browser_record(record, persisted=False)
            record_type = record.get("type")
            worklet_id = record.get("worklet_id")
            if record_type == "browser.audio_worklet.registered":
                if not isinstance(worklet_id, str) or worklet_id in self._registered_worklets:
                    raise ValueError("invalid or duplicate browser worklet registration")
                self._registered_worklets.add(worklet_id)
                self._last_ordinals[worklet_id] = 0
            elif record_type == "browser.audio_worklet.dequeue":
                ordinal = record.get("ordinal")
                if worklet_id not in self._registered_worklets or ordinal != (
                    self._last_ordinals[worklet_id] + 1
                ):
                    raise ValueError("browser worklet dequeue ordinal is not contiguous")
                self._last_ordinals[worklet_id] = ordinal
            elif record_type == "browser.audio_worklet.flush_ack":
                final_ordinal = record.get("final_ordinal")
                if (
                    worklet_id not in self._registered_worklets
                    or worklet_id in self._acknowledged_ordinals
                    or not isinstance(final_ordinal, int)
                ):
                    raise ValueError("invalid or duplicate browser worklet flush acknowledgement")
                self._acknowledged_ordinals[worklet_id] = final_ordinal
            lines.append(
                json.dumps(
                    {**record, "playwright_batch_received_monotonic_s": received},
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
        self._stream.write("".join(lines))
        self._stream.flush()
        self._chunks += 1
        self._records += len(records)
        self._handler_seconds += time.perf_counter() - started

    def emit_marker(self, marker: dict[str, Any]) -> None:
        """Persist a driver boundary in the same ordered browser artifact."""

        validate_marker(marker)
        record = {
            **marker,
            "playwright_batch_received_monotonic_s": time.monotonic(),
            "playwright_batch_received_wall_time_s": time.time(),
        }
        self._stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        self._stream.flush()
        self._marker_records += 1

    def overhead(self) -> dict[str, int | float]:
        return {
            "playwright_binding_chunks": self._chunks,
            "playwright_binding_records": self._records,
            "playwright_binding_handler_ms": self._handler_seconds * 1000.0,
            "fixture_marker_records": self._marker_records,
        }

    def ordinal_summary(self) -> dict[str, Any]:
        return {
            "persisted_last_ordinals": dict(sorted(self._last_ordinals.items())),
            "acknowledged_final_ordinals": dict(sorted(self._acknowledged_ordinals.items())),
        }

    def _assert_final_ordinals(self) -> None:
        if not self.capture_enabled:
            return
        if self._registered_worklets != self._acknowledged_ordinals.keys():
            raise AssertionError("not every browser worklet persisted a flush acknowledgement")
        if self._last_ordinals != self._acknowledged_ordinals:
            raise AssertionError(
                "persisted browser worklet ordinal does not equal acknowledged final ordinal"
            )

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            try:
                self._assert_final_ordinals()
            finally:
                self._stream.close()


async def _install_browser_playout_capture(
    context: Any,
    page: Any,
    artifact: Path,
    *,
    peer_connection_slot: str,
    reservation: Step2PathReservation | None = None,
) -> _BrowserPlayoutTraceSink:
    sink = _BrowserPlayoutTraceSink(artifact, reservation=reservation)
    try:
        await context.expose_binding("__voicechatEmitPlayoutTraceBatch", sink.emit_batch)
        await page.evaluate(_audio_worklet_playout_capture_script(peer_connection_slot))
    except Exception:
        sink.close()
        raise
    return sink


async def _shutdown_browser_playout_capture(
    page: Any, sink: _BrowserPlayoutTraceSink | None
) -> dict[str, Any]:
    """Always stop the page producer before closing its artifact sink."""

    page_summary: dict[str, Any] = {}
    shutdown_error: str | None = None
    sink_close_error: str | None = None
    ordinal_summary: dict[str, Any] = {}
    try:
        if sink is not None and sink.capture_enabled and not page.is_closed():
            page_summary = await page.evaluate(
                "() => window.__voicechatStopPlayoutCapture?.() || {}"
            )
    except Exception as exc:  # noqa: BLE001 - preserve the browser-test failure path
        shutdown_error = repr(exc)
    finally:
        if sink is not None:
            try:
                sink.close()
            except Exception as exc:  # noqa: BLE001 - report artifact teardown failure
                sink_close_error = repr(exc)
            ordinal_summary = sink.ordinal_summary()
    return {
        "capture_enabled": sink.capture_enabled if sink is not None else False,
        "shutdown_error": shutdown_error,
        "sink_close_error": sink_close_error,
        **page_summary,
        **ordinal_summary,
        **(sink.overhead() if sink is not None else {}),
    }


async def _disconnect_step2_page(page: Any) -> dict[str, Any]:
    """Await the Playground client's clean SmallWebRTC disconnect path."""

    result: dict[str, Any] = {
        "attempted": False,
        "completed": False,
        "peer_connection_present": False,
        "connection_state": None,
        "signaling_state": None,
        "error": None,
    }
    try:
        if page.is_closed():
            raise RuntimeError("page closed before WebRTC disconnect")
        disconnect = page.get_by_role("button", name="Disconnect", exact=True)
        connect = page.get_by_role("button", name="Connect", exact=True)
        if await disconnect.count() != 1 or not await disconnect.is_visible():
            raise RuntimeError("Playground disconnect control is not uniquely visible")
        result["attempted"] = True
        # The bundled Playground ConnectButton awaits handleDisconnect(), which
        # invokes PipecatClient.disconnect() and SmallWebRTCTransport.stop().
        # Waiting for Connect to return therefore waits for tracks/transceivers
        # and the peer connection to be closed before the context disappears.
        await disconnect.click()
        await connect.wait_for(state="visible", timeout=10_000)
        peer_state = await page.evaluate(
            """() => {
              const pc = window.__voicechatStep2PC;
              return {
                peer_connection_present: Boolean(pc),
                connection_state: pc?.connectionState ?? null,
                signaling_state: pc?.signalingState ?? null,
              };
            }"""
        )
        result.update(peer_state)
        if not result["peer_connection_present"]:
            raise RuntimeError("Step 2 peer connection was not observed")
        if result["signaling_state"] != "closed":
            raise RuntimeError(f"Step 2 peer connection did not close cleanly: {peer_state}")
        result["completed"] = True
    except Exception as exc:  # noqa: BLE001 - retain the complete teardown record
        result["error"] = repr(exc)
    return result


async def _teardown_step2_session(
    *,
    page: Any,
    sink: _BrowserPlayoutTraceSink | None,
    context: Any,
    browser: Any,
    artifact: Path,
    capture_enabled: bool,
    artifact_loader: Any = load_browser_artifact,
    expected_fixtures: tuple[FixtureSpec, ...] | None = None,
    reservation: Step2PathReservation | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, list[str]]:
    """Attempt every teardown boundary and retain each independent failure."""

    shutdown: dict[str, Any] | None = None
    validation: dict[str, Any] | None = None
    errors: list[str] = []
    if page is not None:
        shutdown = await _shutdown_browser_playout_capture(page, sink)
        if shutdown.get("shutdown_error") or shutdown.get("sink_close_error"):
            errors.append(f"capture shutdown: {shutdown}")
        disconnect = await _disconnect_step2_page(page)
        shutdown["webrtc_disconnect"] = disconnect
        if disconnect["error"] is not None:
            errors.append(f"WebRTC disconnect failed: {disconnect['error']}")
    elif sink is not None:
        try:
            sink.close()
        except Exception as exc:  # noqa: BLE001 - retain sink-only setup failure
            errors.append(f"sink.close failed: {exc!r}")
    artifact_exists = (
        reservation.child_exists(artifact) if reservation is not None else artifact.exists()
    )
    if artifact_exists:
        try:
            if reservation is None:
                validation = artifact_loader(
                    artifact,
                    capture_enabled=capture_enabled,
                    **(
                        {"expected_fixtures": expected_fixtures}
                        if expected_fixtures is not None
                        else {}
                    ),
                )
            else:
                validation = artifact_loader(
                    artifact,
                    capture_enabled=capture_enabled,
                    **(
                        {"expected_fixtures": expected_fixtures}
                        if expected_fixtures is not None
                        else {}
                    ),
                    reservation=reservation,
                )
        except Exception as exc:  # noqa: BLE001 - checked evidence failure
            errors.append(f"browser artifact validation failed: {exc!r}")
    if context is not None:
        try:
            await context.close()
        except Exception as exc:  # noqa: BLE001 - continue remaining cleanup
            errors.append(f"context.close failed: {exc!r}")
    if browser is not None:
        try:
            await browser.close()
        except Exception as exc:  # noqa: BLE001 - outer terminalization records it
            errors.append(f"browser.close failed: {exc!r}")
    return shutdown, validation, errors


async def _stop_playwright_recorded(
    playwright: Any, terminalization_errors: list[str]
) -> str | None:
    try:
        await playwright.stop()
    except Exception as exc:  # noqa: BLE001 - retain Playwright exit failure
        error = f"Playwright exit failed: {exc!r}"
        terminalization_errors.append(error)
        return error
    return None


async def _start_playwright(manager: Any) -> Any:
    """Expose Playwright entry as an injectable awaited orchestration boundary."""

    return await manager.start()


def _terminalize_step2_summary(summary: dict[str, Any], failure: str | None, persist: Any) -> None:
    """Write exactly one terminal state from the outermost campaign finally."""

    summary["completed_wall_time_s"] = time.time()
    summary["status"] = "failed" if failure is not None else "completed"
    persist()


async def _setup_step2_session(
    *,
    state: dict[str, Any],
    playwright: Any,
    url: str,
    microphone_wav: Path,
    artifact: Path,
    capture_mode: str,
    browser_log: list[str],
    reservation: Step2PathReservation | None = None,
) -> Any:
    """Build one isolated browser session while exposing partial state to teardown."""

    state["browser"] = await playwright.chromium.launch(
        executable_path=CHROMIUM,
        headless=True,
        args=[
            "--no-sandbox",
            "--use-fake-device-for-media-stream",
            "--use-fake-ui-for-media-stream",
            f"--use-file-for-fake-audio-capture={microphone_wav}%noloop",
            "--autoplay-policy=no-user-gesture-required",
        ],
    )
    state["context"] = await state["browser"].new_context(permissions=["microphone", "camera"])
    await state["context"].add_init_script(_rtc_probe_script("__voicechatStep2PC"))
    state["page"] = await state["context"].new_page()
    state["page"].on("console", lambda message: browser_log.append(message.text))
    state["page"].on("pageerror", lambda error: browser_log.append(f"pageerror: {error}"))
    await state["page"].goto(url.rstrip("/") + "/", wait_until="load", timeout=30_000)
    if capture_mode == "on":
        state["sink"] = await _install_browser_playout_capture(
            state["context"],
            state["page"],
            artifact,
            peer_connection_slot="__voicechatStep2PC",
            reservation=reservation,
        )
    else:
        state["sink"] = _BrowserPlayoutTraceSink(
            artifact, capture_enabled=False, reservation=reservation
        )

    connect = state["page"].get_by_role("button", name="Connect", exact=True)
    disconnect = state["page"].get_by_role("button", name="Disconnect", exact=True)
    for _ in range(6):
        await connect.click()
        try:
            await disconnect.wait_for(timeout=4_000)
            break
        except PlaywrightTimeoutError:
            await state["page"].wait_for_timeout(300)
    else:
        raise AssertionError(f"Step 2 Playground did not connect: {browser_log[-50:]}")

    text_box = state["page"].locator('[placeholder="Type message..."]:visible')
    await text_box.wait_for(state="visible", timeout=10_000)
    if not await text_box.is_enabled():
        raise AssertionError(f"Step 2 text input never became ready: {browser_log[-50:]}")
    return text_box


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _write_fake_microphone(path: Path) -> None:
    """Write speech-like 48 kHz mono audio accepted by Chrome's fake mic."""
    rate = 48_000
    duration = 4.0
    t = np.arange(int(rate * duration), dtype=np.float64) / rate
    envelope = np.minimum(1.0, np.maximum(0.0, np.sin(math.pi * t / duration) * 3))
    samples = (
        0.16 * envelope * (np.sin(2 * math.pi * 180 * t) + 0.35 * np.sin(2 * math.pi * 360 * t))
    )
    pcm = np.rint(np.clip(samples, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm.tobytes())


def _write_silent_microphone(path: Path, duration: float = 120.0) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(48_000)
        wav.writeframes(bytes(round(48_000 * duration) * 2))


def _write_fixture_summary(
    path: Path,
    summary: dict[str, Any],
    *,
    reservation: Step2PathReservation | None = None,
) -> None:
    """Atomically retain every state transition, including partial failures."""

    validate_fixture_summary(summary, reservation=reservation)
    serialized = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if reservation is None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(serialized, encoding="utf-8")
        temporary.replace(path)
        return
    temporary = path.with_name(path.name + ".tmp")
    descriptor = reservation.open_child(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            descriptor = -1
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        reservation.replace_child(temporary, path)
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            reservation.unlink_child(temporary)
        except FileNotFoundError:
            pass
        raise


def _write_step4c_summary(
    path: Path, summary: dict[str, Any], reservation: Step2PathReservation
) -> None:
    """Atomically retain the separately validated Step 4c qualification state."""

    validate_step4c_summary(summary)
    temporary = path.with_name(path.name + ".tmp")
    serialized = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    descriptor = reservation.open_child(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            descriptor = -1
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        reservation.replace_child(temporary, path)
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            reservation.unlink_child(temporary)
        except FileNotFoundError:
            pass
        raise


async def _browser_clock(page: Any) -> dict[str, float | int]:
    return await page.evaluate(
        "() => ({performance_ms: performance.now(), wall_time_s: Date.now() / 1000, "
        "rtvi_message_index: (window.__voicechatRtviTimedMessages || []).length})"
    )


async def _emit_fixture_boundary(
    page: Any,
    sink: _BrowserPlayoutTraceSink,
    fixture: FixtureSpec,
    boundary: str,
    *,
    detail: str | None = None,
) -> dict[str, float | int]:
    clock = await _browser_clock(page)
    sink.emit_marker(
        fixture_marker(
            fixture,
            boundary,
            browser_performance_ms=clock["performance_ms"],
            browser_wall_time_s=clock["wall_time_s"],
            detail=detail,
        )
    )
    return clock


async def _emit_fixture_boundary_recorded(
    page: Any,
    sink: _BrowserPlayoutTraceSink,
    fixture: FixtureSpec,
    result: dict[str, Any],
    boundary: str,
    persist: Any,
    *,
    detail: str | None = None,
) -> dict[str, float | int]:
    """Emit a boundary while retaining either its success or its failure."""

    try:
        clock = await _emit_fixture_boundary(page, sink, fixture, boundary, detail=detail)
    except Exception as exc:
        result["marker_emission"][boundary] = {
            "status": "failed",
            "error": repr(exc),
        }
        persist()
        raise
    result["marker_emission"][boundary] = {"status": "succeeded", "error": None}
    persist()
    return clock


async def _timed_rtvi_messages(page: Any, after: int) -> list[dict[str, Any]]:
    return await page.evaluate(
        "(after) => (window.__voicechatRtviTimedMessages || []).slice(after)", after
    )


async def _await_rtvi_quiescence_barrier(
    page: Any,
    terminal_event: dict[str, Any],
    *,
    start_message_count: int,
    quiet_window_ms: float,
    sleep: Any = asyncio.sleep,
) -> dict[str, Any]:
    """Retain a fixed post-terminal window on the ordered RTVI data channel."""

    quiet_window_ms = validate_rtvi_quiescence_window_ms(quiet_window_ms)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + quiet_window_ms / 1000.0
    while (remaining := deadline - loop.time()) > 0:
        await sleep(min(0.05, remaining))
    observed = await _timed_rtvi_messages(page, start_message_count)
    validate_rtvi_message_clocks(observed)
    bot_types = [
        message_type
        for item in observed
        if isinstance(item.get("message"), dict)
        and isinstance((message_type := item["message"].get("type")), str)
        and message_type.startswith("bot-")
    ]
    end = await _browser_clock(page)
    start_performance = float(terminal_event["receivedPerformanceMs"])
    start_wall = float(terminal_event["receivedWallTimeS"])
    end_performance = float(end["performance_ms"])
    end_wall = float(end["wall_time_s"])
    performance_window_complete = end_performance >= start_performance + quiet_window_ms
    wall_window_complete = (
        end_wall + RTVI_WALL_CLOCK_SKEW_TOLERANCE_S >= start_wall + quiet_window_ms / 1000.0
    )
    return {
        "identity": RTVI_TRANSCRIPT_IDENTITY,
        "quiet_window_ms": float(quiet_window_ms),
        "window_start_performance_ms": start_performance,
        "window_end_performance_ms": end_performance,
        "window_start_wall_time_s": start_wall,
        "window_end_wall_time_s": end_wall,
        "start_message_count": start_message_count,
        "end_message_count": start_message_count + len(observed),
        "observed_message_count": len(observed),
        "bot_message_count": len(bot_types),
        "bot_message_types": bot_types,
        "passed": not bot_types and performance_window_complete and wall_window_complete,
    }


@pytest.mark.parametrize(
    "quiet_window_ms", [0.0, 499.0, -1.0, float("nan"), float("inf"), 60_001.0]
)
@pytest.mark.asyncio
async def test_rtvi_quiescence_barrier_rejects_invalid_runtime_window(
    quiet_window_ms: float,
) -> None:
    with pytest.raises(ValueError, match="RTVI quiescence window"):
        await _await_rtvi_quiescence_barrier(
            object(),
            {},
            start_message_count=0,
            quiet_window_ms=quiet_window_ms,
        )


class _QuiescenceBarrierProbeLoop:
    def __init__(self) -> None:
        self.now = 10.0

    def time(self) -> float:
        return self.now


class _QuiescenceBarrierProbePage:
    def __init__(self, *, performance_ms: float, wall_time_s: float) -> None:
        self.performance_ms = performance_ms
        self.wall_time_s = wall_time_s

    async def evaluate(self, expression: str, argument: object = None) -> object:
        if "slice(after)" in expression:
            return []
        if "performance.now()" in expression and "Date.now()" in expression:
            return {"performance_ms": self.performance_ms, "wall_time_s": self.wall_time_s}
        raise AssertionError(f"unexpected browser probe: {expression!r}, {argument!r}")


async def _runtime_quiescence_barrier_probe(
    monkeypatch: pytest.MonkeyPatch,
    *,
    performance_ms: float,
    wall_time_s: float,
) -> dict[str, Any]:
    loop = _QuiescenceBarrierProbeLoop()

    async def advance_loop(seconds: float) -> None:
        loop.now += seconds

    monkeypatch.setattr(asyncio, "get_running_loop", lambda: loop)
    return await _await_rtvi_quiescence_barrier(
        _QuiescenceBarrierProbePage(
            performance_ms=performance_ms,
            wall_time_s=wall_time_s,
        ),
        {"receivedPerformanceMs": 1000.0, "receivedWallTimeS": 100.0},
        start_message_count=12,
        quiet_window_ms=RTVI_QUIESCENCE_MIN_WINDOW_MS,
        sleep=advance_loop,
    )


@pytest.mark.asyncio
async def test_rtvi_quiescence_barrier_fails_frozen_sampled_clocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    barrier = await _runtime_quiescence_barrier_probe(
        monkeypatch,
        performance_ms=1000.0,
        wall_time_s=100.0,
    )

    assert barrier["window_end_performance_ms"] == 1000.0
    assert barrier["window_end_wall_time_s"] == 100.0
    assert barrier["passed"] is False


@pytest.mark.asyncio
async def test_rtvi_quiescence_barrier_fails_wall_sample_short_by_1_001_ms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sampled_wall = 100.5 - 0.001001
    barrier = await _runtime_quiescence_barrier_probe(
        monkeypatch,
        performance_ms=1500.0,
        wall_time_s=sampled_wall,
    )

    assert barrier["window_end_wall_time_s"] == sampled_wall
    assert barrier["passed"] is False


@pytest.mark.asyncio
async def test_rtvi_quiescence_barrier_accepts_wall_sample_short_by_exactly_1_ms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sampled_wall = 100.5 - RTVI_WALL_CLOCK_SKEW_TOLERANCE_S
    barrier = await _runtime_quiescence_barrier_probe(
        monkeypatch,
        performance_ms=1500.0,
        wall_time_s=sampled_wall,
    )

    assert barrier["window_end_wall_time_s"] == sampled_wall
    assert barrier["passed"] is True


def _rtvi_events(
    messages: list[dict[str, Any]], event_type: str, *, after_performance_ms: float = -1
) -> list[dict[str, Any]]:
    return [
        item
        for item in messages
        if item.get("receivedPerformanceMs", -1) >= after_performance_ms
        and item.get("message", {}).get("type") == event_type
    ]


def _rtvi_event_indices(
    messages: list[dict[str, Any]],
    event_type: str,
    *,
    after_performance_ms: float = -1,
    after_event_index: int | None = None,
) -> list[int]:
    return [
        index
        for index, item in enumerate(messages)
        if (after_event_index is None or index > after_event_index)
        and item.get("receivedPerformanceMs", -1) >= after_performance_ms
        and item.get("message", {}).get("type") == event_type
    ]


def _i1_completion_anchor_indices(
    messages: list[dict[str, Any]],
    *,
    original_start_index: int,
    interruption_marker_index: int,
) -> dict[str, int] | None:
    """Match only I1's ordered speaking anchors; ignore every other RTVI event."""

    if original_start_index >= interruption_marker_index:
        return None

    def first(event_type: str, *, at_or_after: int) -> int | None:
        return next(
            (
                index
                for index, item in enumerate(messages)
                if index >= at_or_after and item.get("message", {}).get("type") == event_type
            ),
            None,
        )

    original_stop_index = first("bot-stopped-speaking", at_or_after=interruption_marker_index)
    if original_stop_index is None:
        return None
    replacement_start_index = first("bot-started-speaking", at_or_after=original_stop_index + 1)
    if replacement_start_index is None:
        return None
    replacement_stop_index = first("bot-stopped-speaking", at_or_after=replacement_start_index + 1)
    if replacement_stop_index is None:
        return None
    return {
        "original_start": original_start_index,
        "interruption_marker": interruption_marker_index,
        "original_stop": original_stop_index,
        "replacement_start": replacement_start_index,
        "replacement_stop": replacement_stop_index,
    }


async def _retain_fixture_rtvi_diagnostics(
    page: Any,
    result: dict[str, Any],
    persist: Any,
) -> None:
    """Retain the raw ordered RTVI receipt projection for a failed fixture."""

    start_index = result.get("rtvi_window_start_index")
    if type(start_index) is not int or start_index < 0:
        start_index = 0
        result["rtvi_window_start_index"] = start_index
    try:
        observed = await _timed_rtvi_messages(page, start_index)
        result["observed_rtvi_messages"] = rtvi_message_diagnostics(
            observed, start_event_index=start_index
        )
        result["rtvi_observation_error"] = None
    except Exception as exc:  # noqa: BLE001 - preserve the diagnostic failure itself
        result["rtvi_observation_error"] = repr(exc)
    persist()


async def _run_step2_fixture_inner(
    page: Any,
    text_box: Any,
    sink: _BrowserPlayoutTraceSink,
    fixture: FixtureSpec,
    result: dict[str, Any],
    persist: Any,
    pipecat_trace_path: Path,
    *,
    trace_reader: Any = read_playout_trace,
    sleep: Any = asyncio.sleep,
    deadline_wait: Any = wait_for_interruption_deadline,
    attribution_cursor: dict[str, int | None] | None = None,
    quiescence_window_ms: float = RTVI_QUIESCENCE_WINDOW_MS,
) -> bool:
    """Require browser boundaries and one reconciled Pipecat response lifecycle."""

    result["completion_status"] = "running"
    result["send_attempt_wall_time_s"] = time.time()
    persist()
    baseline = await page.evaluate("() => (window.__voicechatRtviTimedMessages || []).length")
    result["rtvi_window_start_index"] = baseline
    persist()
    trace_before_send = trace_reader(pipecat_trace_path)
    if attribution_cursor is not None:
        prior_rtvi = attribution_cursor.get("rtvi_index")
        if prior_rtvi is not None and baseline > prior_rtvi:
            gap_messages = await _timed_rtvi_messages(page, prior_rtvi)
            gap_types = {
                item.get("message", {}).get("type")
                for item in gap_messages
                if isinstance(item.get("message"), dict)
            }
            if gap_types & {
                "bot-started-speaking",
                "bot-stopped-speaking",
                "bot-interrupted",
                "bot-transcription",
            }:
                raise AssertionError(
                    "unattributed RTVI response transition appeared between fixtures"
                )
        prior_playout = attribution_cursor.get("playout_index")
        if prior_playout is not None and any(
            record.get("type")
            in {
                "response.created",
                "response.output_audio.delta",
                "response.done",
                "input_audio_buffer.speech_started",
                "voicechat.playout.interruption",
                "voicechat.playout.release",
                "voicechat.playout.downstream_push",
            }
            for record in trace_before_send[prior_playout:]
        ):
            raise AssertionError(
                "unattributed Pipecat response transition appeared between fixtures"
            )
    trace_baseline = len(trace_before_send)
    await text_box.fill(fixture.typed_text)
    send_edge = await _emit_fixture_boundary_recorded(page, sink, fixture, result, "start", persist)
    await text_box.press("Enter")
    result["send_wall_time_s"] = time.time()
    persist()

    loop = asyncio.get_running_loop()
    deadline = loop.time() + fixture.wait.timeout_s
    first_audio: dict[str, Any] | None = None
    replacement_audio: dict[str, Any] | None = None
    interruption_performance_ms: float | None = None
    interruption_send_edge: dict[str, float | int] | None = None
    interruption_marker_event_index: int | None = None
    interruption_sent = False
    messages: list[dict[str, Any]] = []
    attribution: dict[str, Any] | None = None

    while loop.time() < deadline:
        messages = await _timed_rtvi_messages(page, baseline)
        validate_rtvi_message_clocks(messages)
        start_indices = _rtvi_event_indices(messages, "bot-started-speaking")
        starts = [messages[index] for index in start_indices]
        if fixture.interruption is None and len(starts) > 1:
            raise AssertionError("fixture observed extra bot-started-speaking transitions")
        if first_audio is None and starts:
            first_audio = starts[0]
            result["first_audio_wall_time_s"] = first_audio["receivedWallTimeS"]
            persist()

        interruption = fixture.interruption
        if first_audio is not None and interruption is not None and not interruption_sent:
            now_browser_s = (await page.evaluate("performance.now()")) / 1000.0
            target_browser_s = (
                first_audio["receivedPerformanceMs"] / 1000.0 + interruption.after_first_audio_s
            )
            target_loop_s = loop.time() + max(0.0, target_browser_s - now_browser_s)
            if target_loop_s >= deadline:
                break
            await deadline_wait(
                target_loop_s - interruption.after_first_audio_s,
                interruption.after_first_audio_s,
                monotonic=loop.time,
            )
            result["interruption"]["send_attempt_wall_time_s"] = time.time()
            persist()
            await text_box.fill(interruption.typed_text)
            interruption_clock = await _emit_fixture_boundary_recorded(
                page,
                sink,
                fixture,
                result,
                "interruption-send",
                persist,
            )
            interruption_performance_ms = interruption_clock["performance_ms"]
            interruption_send_edge = interruption_clock
            interruption_marker_event_index = int(interruption_clock["rtvi_message_index"])
            result["interruption"]["rtvi_send_marker_event_index"] = interruption_marker_event_index
            await text_box.press("Enter")
            result["interruption"]["send_wall_time_s"] = time.time()
            result["interruption"]["actual_offset_s"] = (
                interruption_performance_ms - first_audio["receivedPerformanceMs"]
            ) / 1000.0
            persist()
            interruption_sent = True
            continue

        completion_start_index = start_indices[0] if start_indices else None
        i1_anchors: dict[str, int] | None = None
        if interruption is not None:
            if not interruption_sent or interruption_performance_ms is None:
                await sleep(0.05)
                continue
            if interruption_marker_event_index is None:
                raise AssertionError("I1 lacks its ordered-channel interruption marker")
            marker_window_index = interruption_marker_event_index - baseline
            if start_indices:
                i1_anchors = _i1_completion_anchor_indices(
                    messages,
                    original_start_index=start_indices[0],
                    interruption_marker_index=marker_window_index,
                )
            if replacement_audio is None and i1_anchors is not None:
                replacement_audio = messages[i1_anchors["replacement_start"]]
            completion_start_index = (
                i1_anchors["replacement_start"] if i1_anchors is not None else None
            )

        if completion_start_index is not None:
            completion_start = messages[completion_start_index]
            completion_indices = (
                [i1_anchors["replacement_stop"]]
                if i1_anchors is not None
                else _rtvi_event_indices(
                    messages, fixture.wait.event, after_event_index=completion_start_index
                )
            )
            stop_indices = (
                _rtvi_event_indices(messages, fixture.wait.event) if interruption is None else []
            )
            completions = [messages[index] for index in completion_indices]
            stops = [messages[index] for index in stop_indices]
            original_terminal_index: int | None = None
            original_terminal: dict[str, Any] | None = None
            if fixture.interruption is None:
                browser_shape_met = (
                    len(starts) == 1 and len(stops) == 1 and start_indices[0] < stop_indices[0]
                )
            else:
                if interruption_marker_event_index is None:
                    raise AssertionError("I1 lacks its ordered-channel interruption marker")
                # The data-channel listener and marker snapshot share one browser
                # event index.  That ordering is exact; Date.now() is retained as
                # evidence but cannot be compared with Python's post-Enter clock,
                # because the caused stop can race that timestamp capture.
                if i1_anchors is not None:
                    original_terminal_index = i1_anchors["original_stop"]
                    original_terminal = messages[original_terminal_index]
                browser_shape_met = i1_anchors is not None
            playout_snapshot = trace_reader(pipecat_trace_path)
            attribution = correlate_response_lifecycles(
                playout_snapshot[trace_baseline:],
                interruption=fixture.interruption is not None,
            )
            transcriptions: list[str] = []
            transcript_bracket: dict[str, object] | None = None
            if completions and browser_shape_met and attribution:
                response_key = (
                    "replacement_response"
                    if fixture.interruption is not None
                    else "primary_response"
                )
                response = attribution[response_key]
                transcriptions, transcript_bracket = rtvi_transcript_evidence(
                    messages,
                    start_event_index=completion_start_index,
                    terminal_event_index=completion_indices[0],
                    start_event=completion_start,
                    terminal_event=completions[0],
                    response_id=response["response_id"],
                    expected_output_text=response["output_text"],
                    fixture_send_performance_ms=(
                        interruption_performance_ms
                        if fixture.interruption is not None
                        else send_edge["performance_ms"]
                    ),
                    fixture_send_wall_time_s=(
                        interruption_send_edge["wall_time_s"]
                        if interruption_send_edge is not None
                        else send_edge["wall_time_s"]
                    ),
                )
            if completions and browser_shape_met and attribution:
                completed = completions[0]
                response_ids = attribution["response_ids"]
                if not response_ids:
                    raise AssertionError("completed fixture has no attributed response ID")
                result["response_ids"] = response_ids
                result["transcript_evidence"] = transcriptions
                completed_response = attribution[
                    "replacement_response"
                    if fixture.interruption is not None
                    else "primary_response"
                ]
                result["transcript_gate_evidence"] = transcript_gate_evidence(
                    transcriptions, completed_response["output_text"]
                )
                if fixture.interruption is None:
                    attribution["primary_response"].update(
                        {
                            "speaking_start_wall_time_s": starts[0]["receivedWallTimeS"],
                            "speaking_stop_wall_time_s": stops[0]["receivedWallTimeS"],
                            "transcript_evidence": transcriptions,
                            "rtvi_event_bracket": transcript_bracket,
                        }
                    )
                else:
                    second = transcriptions
                    result["interruption"]["second_answer_transcript_evidence"] = second
                    result["interruption"]["second_answer_gate_evidence"] = (
                        second_answer_gate_evidence(second, completed_response["output_text"])
                    )
                    if original_terminal_index is None or original_terminal is None:
                        raise AssertionError("I1 lacks its interruption-caused terminal event")
                    original_bracket_messages = messages[
                        start_indices[0] : original_terminal_index + 1
                    ]
                    original_before_send = sum(
                        float(item["receivedPerformanceMs"]) <= float(send_edge["performance_ms"])
                        or float(item["receivedWallTimeS"]) <= float(send_edge["wall_time_s"])
                        for item in original_bracket_messages
                    )
                    if original_before_send:
                        raise ValueError(
                            "original RTVI bracket contains messages received before its send edge"
                        )
                    attribution["original_response"].update(
                        {
                            "speaking_start_wall_time_s": starts[0]["receivedWallTimeS"],
                            "interrupted_wall_time_s": original_terminal["receivedWallTimeS"],
                            "rtvi_event_bracket": {
                                "response_id": response_ids[0],
                                "transcript_binding": RTVI_TRANSCRIPT_IDENTITY,
                                "protocol_response_identity": None,
                                "fixture_send_performance_ms": send_edge["performance_ms"],
                                "fixture_send_wall_time_s": send_edge["wall_time_s"],
                                "messages_received_before_send_in_bracket": original_before_send,
                                "start_event_index": start_indices[0],
                                "terminal_event_index": original_terminal_index,
                                "start_event_type": "bot-started-speaking",
                                "terminal_event_type": original_terminal["message"]["type"],
                                "start_performance_ms": starts[0]["receivedPerformanceMs"],
                                "terminal_performance_ms": original_terminal[
                                    "receivedPerformanceMs"
                                ],
                                "start_wall_time_s": starts[0]["receivedWallTimeS"],
                                "terminal_wall_time_s": original_terminal["receivedWallTimeS"],
                                "transcript_event_indices": [],
                                "transcript_performance_ms": [],
                                "output_text_sha256": hashlib.sha256(
                                    attribution["original_response"]["output_text"].encode()
                                ).hexdigest(),
                            },
                        }
                    )
                    attribution["replacement_response"].update(
                        {
                            "speaking_start_wall_time_s": completion_start["receivedWallTimeS"],
                            "speaking_stop_wall_time_s": completed["receivedWallTimeS"],
                            "transcript_evidence": second,
                            "rtvi_event_bracket": transcript_bracket,
                        }
                    )
                result["response_attribution"] = attribution
                result["last_audio_wall_time_s"] = completed["receivedWallTimeS"]
                terminal_absolute_index = baseline + completion_indices[0] + 1
                result["quiescence_barrier"] = await _await_rtvi_quiescence_barrier(
                    page,
                    completed,
                    start_message_count=terminal_absolute_index,
                    quiet_window_ms=quiescence_window_ms,
                    sleep=sleep,
                )
                persist()
                if not result["quiescence_barrier"]["passed"]:
                    raise AssertionError(
                        "post-terminal RTVI quiescence barrier failed duration or activity checks"
                    )
                await _emit_fixture_boundary_recorded(
                    page, sink, fixture, result, "complete", persist
                )
                result["completion_wall_time_s"] = time.time()
                result["completion_status"] = "completed"
                if attribution_cursor is not None:
                    attribution_cursor["rtvi_index"] = result["quiescence_barrier"][
                        "end_message_count"
                    ]
                    attribution_cursor["playout_index"] = len(playout_snapshot)
                persist()
                return True
        await sleep(0.05)

    if attribution is not None:
        result["response_ids"] = attribution["response_ids"]
        result["response_attribution"] = attribution
    result["completion_wall_time_s"] = time.time()
    result["completion_status"] = "timeout"
    result["error"] = f"fixture did not complete within {fixture.wait.timeout_s:g}s"
    await _retain_fixture_rtvi_diagnostics(page, result, persist)
    try:
        await _emit_fixture_boundary_recorded(
            page, sink, fixture, result, "timeout", persist, detail=result["error"]
        )
    except Exception as exc:
        result["error"] += f"; timeout marker failed: {exc!r}"
    if attribution_cursor is not None:
        attribution_cursor["rtvi_index"] = baseline + len(messages)
        attribution_cursor["playout_index"] = len(trace_reader(pipecat_trace_path))
    persist()
    return False


async def _run_step2_fixture(*args: Any, **kwargs: Any) -> bool:
    """Run one fixture and retain its RTVI window before propagating any error."""

    page = args[0]
    result = args[4]
    persist = args[5]
    try:
        return await _run_step2_fixture_inner(*args, **kwargs)
    except Exception:
        await _retain_fixture_rtvi_diagnostics(page, result, persist)
        raise


def _assistant_pcm() -> bytes:
    rate = 22_050
    t = np.arange(rate, dtype=np.float64) / rate
    envelope = np.minimum(1.0, np.minimum(t * 20, (1 - t) * 20))
    samples = 0.12 * envelope * np.sin(2 * math.pi * 440 * t)
    return np.rint(samples * 32767).astype("<i2").tobytes()


async def _wait_http(url: str, timeout: float = 15.0) -> None:
    def probe() -> None:
        with urllib.request.urlopen(url, timeout=1):
            pass

    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        try:
            await asyncio.to_thread(probe)
            return
        except (OSError, urllib.error.URLError):
            await asyncio.sleep(0.1)
    raise TimeoutError(f"runner did not become ready: {url}")


def test_audio_worklet_capture_harness_schema_and_sink(tmp_path) -> None:
    script = _audio_worklet_playout_capture_script("__testPC")
    assert "registerProcessor('voicechat-playout-capture'" in script
    assert "browser.remote_audio.enqueue" in script
    assert "browser.audio_worklet.dequeue" in script
    assert "underrunCount" in script
    assert "browser.audio_worklet.onset" in script
    assert 'window["__testPC"]' in script
    assert "render_audio_context_time_s" in script
    assert "main_thread_received_performance_ms" in script
    assert "parallel_silent_audio_context_tap" in script
    assert "__voicechatEmitPlayoutTraceBatch" in script
    assert "EXPORT_CHUNK_RECORDS = 128" in script
    assert "this.batch.length >= 64" in script
    assert "kind: 'flush-ack'" in script
    assert "worklet_final_ordinals" in script
    assert "MAX_PENDING_BINDING_WRITES = 16" in script

    path = tmp_path / "browser-playout.jsonl"
    sink = _BrowserPlayoutTraceSink(path)
    with pytest.raises(ValueError, match="schema"):
        sink.emit_batch(
            None,
            [{"trace_schema": "nemotron_voicechat.browser_playout.v1", "ordinal": 1}],
        )
    sink.emit_batch(
        None,
        [
            {
                "trace_schema": "nemotron_voicechat.browser_playout.v1",
                "type": "browser.playout.capability",
                "audio_worklet": False,
            }
        ],
    )
    sink.close()
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["type"] == "browser.playout.capability"
    assert record["audio_worklet"] is False
    assert isinstance(record["playwright_batch_received_monotonic_s"], float)
    assert sink.overhead()["playwright_binding_chunks"] == 1
    assert sink.overhead()["playwright_binding_records"] == 1


def test_step2_child_writers_survive_deterministic_directory_swap_after_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    microphone = tmp_path / "mic.wav"
    microphone.write_bytes(b"RIFF")
    session = tmp_path / "session"
    displaced = tmp_path / "reserved-session"
    attacker = tmp_path / "attacker"
    attacker.mkdir()
    paths = resolve_step2_paths(
        microphone=microphone,
        capture_on=session / "browser-on.jsonl",
        capture_off=session / "browser-off.jsonl",
        summary=session / "summary.json",
        pipecat_trace=session / "playout-{connection}.jsonl",
        output_dir=session,
    )
    session.rename(displaced)
    session.symlink_to(attacker, target_is_directory=True)
    try:
        sink = _BrowserPlayoutTraceSink(
            Path(paths["browser_capture_on"]),
            capture_enabled=False,
            reservation=paths,
        )
        sink.close()
        monkeypatch.setattr(
            sys.modules[__name__], "validate_fixture_summary", lambda *_a, **_k: None
        )
        _write_fixture_summary(Path(paths["summary"]), {}, reservation=paths)

        assert (displaced / "browser-on.jsonl").is_file()
        assert (displaced / "summary.json").read_text(encoding="utf-8") == "{}\n"
        assert list(attacker.iterdir()) == []
        with pytest.raises(ValueError, match="no longer names|identity changed"):
            read_playout_trace(Path(paths["pipecat_playout_capture_on"]), reservation=paths)
    finally:
        paths.close()


def test_browser_playout_sink_rejects_unbounded_chunks(tmp_path) -> None:
    sink = _BrowserPlayoutTraceSink(tmp_path / "bounded.jsonl")
    with pytest.raises(TypeError, match="1..128"):
        sink.emit_batch(None, [{}] * 129)
    sink.close()


def test_browser_playout_sink_persists_capture_off_fixture_markers(tmp_path) -> None:
    plan = parse_fixture_plan(canonical_plan_json())
    path = tmp_path / "capture-off.jsonl"
    sink = _BrowserPlayoutTraceSink(path, capture_enabled=False)
    sink.emit_marker(
        fixture_marker(
            plan.fixtures[-1],
            "start",
            browser_performance_ms=10.0,
            browser_wall_time_s=20.0,
        )
    )
    sink.close()

    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["type"] == "browser.fixture_boundary"
    assert record["fixture_id"] == "capture-off-L1"
    assert record["script_id"] == "L1"
    assert record["capture_enabled"] is False
    assert isinstance(record["playwright_batch_received_wall_time_s"], float)
    assert sink.overhead()["fixture_marker_records"] == 1


def test_mixed_capture_on_file_passes_the_checked_browser_consumer(tmp_path) -> None:
    plan = parse_fixture_plan(canonical_plan_json())
    path = tmp_path / "mixed-browser.jsonl"
    sink = _BrowserPlayoutTraceSink(path)
    audio = {
        "trace_schema": "nemotron_voicechat.browser_playout.v1",
        "worklet_id": "worklet-1",
        "main_thread_received_performance_ms": 3.0,
        "render_audio_context_time_s": 0.1,
        "render_audio_context_frame": 128,
        "ordinal": 1,
        "sample_count": 128,
        "channel_count": 1,
        "rms": 0.1,
        "underrun": False,
        "underrun_count": 0,
    }
    sink.emit_batch(
        None,
        [
            {
                "trace_schema": "nemotron_voicechat.browser_playout.v1",
                "type": "browser.playout.capability",
                "audio_worklet": True,
            },
            {
                "trace_schema": "nemotron_voicechat.browser_playout.v1",
                "type": "browser.clock_mapping",
                "phase": "startup",
                "main_thread_performance_ms": 1.0,
                "audio_context_time_s": 0.0,
                "output_context_time_s": None,
                "output_performance_time_ms": None,
            },
            {
                "trace_schema": "nemotron_voicechat.browser_playout.v1",
                "type": "browser.audio_worklet.registered",
                "worklet_id": "worklet-1",
            },
            {**audio, "type": "browser.audio_worklet.dequeue"},
            {**audio, "type": "browser.audio_worklet.onset"},
        ],
    )
    marker_clock = 4.0
    for fixture in plan.sessions()[0][1]:
        boundaries = (
            ("start", "interruption-send", "complete")
            if fixture.interruption
            else ("start", "complete")
        )
        for boundary in boundaries:
            sink.emit_marker(
                fixture_marker(
                    fixture,
                    boundary,
                    browser_performance_ms=marker_clock,
                    browser_wall_time_s=100.0 + marker_clock,
                )
            )
            marker_clock += 1.0
    sink.emit_batch(
        None,
        [
            {
                "trace_schema": "nemotron_voicechat.browser_playout.v1",
                "type": "browser.audio_worklet.flush_ack",
                "worklet_id": "worklet-1",
                "final_ordinal": 1,
                "delivered_last_ordinal": 1,
            }
        ],
    )
    sink.close()

    validation = load_browser_artifact(path, capture_enabled=True)

    assert validation["valid"] is True
    assert validation["fixture_marker_records"] == 15
    assert validation["persisted_last_ordinals"] == {"worklet-1": 1}


def test_browser_playout_sink_rejects_missing_or_mismatched_final_ordinal(tmp_path) -> None:
    sink = _BrowserPlayoutTraceSink(tmp_path / "mismatch.jsonl")
    sink.emit_batch(
        None,
        [
            {
                "trace_schema": "nemotron_voicechat.browser_playout.v1",
                "type": "browser.audio_worklet.registered",
                "worklet_id": "worklet-1",
            },
            {
                "trace_schema": "nemotron_voicechat.browser_playout.v1",
                "type": "browser.audio_worklet.dequeue",
                "worklet_id": "worklet-1",
                "main_thread_received_performance_ms": 1.0,
                "render_audio_context_time_s": 0.1,
                "render_audio_context_frame": 128,
                "ordinal": 1,
                "sample_count": 128,
                "channel_count": 1,
                "rms": 0.1,
                "underrun": False,
                "underrun_count": 0,
            },
            {
                "trace_schema": "nemotron_voicechat.browser_playout.v1",
                "type": "browser.audio_worklet.flush_ack",
                "worklet_id": "worklet-1",
                "final_ordinal": 2,
                "delivered_last_ordinal": 2,
            },
        ],
    )
    with pytest.raises(AssertionError, match="acknowledged final ordinal"):
        sink.close()


@pytest.mark.asyncio
async def test_browser_finally_path_checks_persisted_ordinal_after_page_failure(
    tmp_path,
) -> None:
    sink = _BrowserPlayoutTraceSink(tmp_path / "finally.jsonl")
    sink.emit_batch(
        None,
        [
            {
                "trace_schema": "nemotron_voicechat.browser_playout.v1",
                "type": "browser.audio_worklet.registered",
                "worklet_id": "worklet-1",
            },
            {
                "trace_schema": "nemotron_voicechat.browser_playout.v1",
                "type": "browser.audio_worklet.dequeue",
                "worklet_id": "worklet-1",
                "main_thread_received_performance_ms": 1.0,
                "render_audio_context_time_s": 0.1,
                "render_audio_context_frame": 128,
                "ordinal": 1,
                "sample_count": 128,
                "channel_count": 1,
                "rms": 0.1,
                "underrun": False,
                "underrun_count": 0,
            },
            {
                "trace_schema": "nemotron_voicechat.browser_playout.v1",
                "type": "browser.audio_worklet.flush_ack",
                "worklet_id": "worklet-1",
                "final_ordinal": 1,
                "delivered_last_ordinal": 1,
            },
        ],
    )

    class _FailingPage:
        def is_closed(self) -> bool:
            return False

        async def evaluate(self, _script: str):
            raise RuntimeError("injected page shutdown failure")

    summary = await _shutdown_browser_playout_capture(_FailingPage(), sink)
    assert "injected page shutdown failure" in summary["shutdown_error"]
    assert summary["sink_close_error"] is None
    assert summary["persisted_last_ordinals"] == {"worklet-1": 1}
    assert summary["persisted_last_ordinals"] == summary["acknowledged_final_ordinals"]


def _fake_playout_event(event_type: str, clock: float, **fields: Any) -> dict[str, Any]:
    clock_field = {
        "response.created": "client_received_monotonic_s",
        "response.output_text.delta": "client_received_monotonic_s",
        "response.output_audio.delta": "client_received_monotonic_s",
        "response.done": "client_received_monotonic_s",
        "input_audio_buffer.speech_started": "client_received_monotonic_s",
        "voicechat.playout.release": "client_release_monotonic_s",
        "voicechat.playout.downstream_push": "client_downstream_push_monotonic_s",
    }[event_type]
    return {
        "trace_schema": "nemotron_voicechat.playout.v1",
        "type": event_type,
        clock_field: clock,
        **fields,
    }


def _fake_response_trace(
    response_id: str,
    start: float,
    *,
    cancelled: bool = False,
    include_typed_send: bool = True,
    output_text: str = "The red balloon rose.",
) -> list[dict[str, Any]]:
    records = [
        *(
            [
                _fake_playout_event(
                    "input_audio_buffer.speech_started",
                    start - 0.1,
                    response_id=None,
                    source="typed",
                )
            ]
            if include_typed_send
            else []
        ),
        _fake_playout_event("response.created", start, response_id=response_id, turn_id=None),
        _fake_playout_event(
            "response.output_text.delta",
            start + 0.05,
            response_id=response_id,
            ordinal=1,
            delta=output_text,
        ),
        _fake_playout_event(
            "response.output_audio.delta",
            start + 0.1,
            response_id=response_id,
            ordinal=1,
        ),
    ]
    if cancelled:
        records += [
            _fake_playout_event(
                "voicechat.playout.release",
                start + 0.11,
                response_id=response_id,
                reason="threshold",
                ordinals=[1],
                frames_released=1,
                frames_cleared=0,
                sample_count=128,
            ),
            _fake_playout_event(
                "voicechat.playout.downstream_push",
                start + 0.12,
                response_id=response_id,
                ordinal=1,
                sample_count=128,
            ),
            _fake_playout_event(
                "response.output_audio.delta",
                start + 0.15,
                response_id=response_id,
                ordinal=2,
            ),
            _fake_playout_event(
                "input_audio_buffer.speech_started",
                start + 0.2,
                response_id=response_id,
                source="typed",
            ),
            _fake_playout_event(
                "voicechat.playout.release",
                start + 0.21,
                response_id=response_id,
                reason="interruption-clear",
                ordinals=[2],
                frames_released=0,
                frames_cleared=1,
                sample_count=0,
            ),
            _fake_playout_event(
                "response.done",
                start + 0.3,
                response_id=response_id,
                status="cancelled",
            ),
        ]
    else:
        records += [
            _fake_playout_event(
                "response.done",
                start + 0.3,
                response_id=response_id,
                status="completed",
            ),
            _fake_playout_event(
                "voicechat.playout.release",
                start + 0.31,
                response_id=response_id,
                reason="done",
                ordinals=[1],
                frames_released=1,
                frames_cleared=0,
                sample_count=128,
            ),
            _fake_playout_event(
                "voicechat.playout.downstream_push",
                start + 0.32,
                response_id=response_id,
                ordinal=1,
                sample_count=128,
            ),
        ]
    return records


def _timed_message(event_type: str, performance_ms: float, **data: Any) -> dict[str, Any]:
    message: dict[str, Any] = {"type": event_type}
    if data:
        message["data"] = data
    return {
        "receivedPerformanceMs": performance_ms,
        "receivedWallTimeS": 100.0 + performance_ms / 1000.0,
        "message": message,
    }


class _FixtureFakePage:
    def __init__(
        self,
        batches: list[list[dict[str, Any]]],
        operations: list[str],
        *,
        fail_stage: str | None = None,
        baseline_length: int = 0,
        boundary_clocks: list[dict[str, float]] | None = None,
        performance_now_ms: float | None = None,
    ):
        self.batches = batches
        self.operations = operations
        self.fail_stage = fail_stage
        self.baseline_length = baseline_length
        self.boundary_clocks = boundary_clocks
        self.performance_now_ms = performance_now_ms
        self.clock_calls = 0
        self.poll_calls = 0

    async def evaluate(self, script: str, *_args: Any) -> Any:
        if "performance.now()" in script and "Date.now()" in script:
            self.clock_calls += 1
            stage = f"boundary-clock-{self.clock_calls}"
            default_clock = {
                "performance_ms": 5.0
                if self.clock_calls == 1
                else 4010.0
                if self.clock_calls == 2
                else 5000.0,
                "wall_time_s": 100.005
                if self.clock_calls == 1
                else 104.01
                if self.clock_calls == 2
                else 105.0,
            }
            clock = (
                self.boundary_clocks[min(self.clock_calls - 1, len(self.boundary_clocks) - 1)]
                if self.boundary_clocks
                else default_clock
            )
            value = {
                **clock,
                "rtvi_message_index": self.baseline_length
                + (
                    0
                    if self.clock_calls == 1
                    else len(self.batches[0])
                    if self.clock_calls == 2
                    else len(self.batches[-1])
                ),
            }
        elif ".length" in script:
            stage = "baseline"
            value: Any = self.baseline_length
        elif ".slice(after)" in script:
            self.poll_calls += 1
            stage = f"poll-{self.poll_calls}"
            batch = self.batches[min(self.poll_calls - 1, len(self.batches) - 1)]
            after = int(_args[0]) if _args else self.baseline_length
            value = batch[max(0, after - self.baseline_length) :]
        elif script == "performance.now()":
            stage = "performance-now"
            value = self.performance_now_ms if self.performance_now_ms is not None else 4010.0
        else:
            raise AssertionError(f"unexpected fake-page script: {script!r}")
        self.operations.append(stage)
        if self.fail_stage == stage:
            raise RuntimeError(f"injected {stage} failure")
        return value


class _FixtureFakeTextBox:
    def __init__(self, operations: list[str], *, fail_stage: str | None = None):
        self.operations = operations
        self.fail_stage = fail_stage
        self.fill_calls = 0
        self.press_calls = 0

    async def fill(self, _text: str) -> None:
        self.fill_calls += 1
        stage = f"fill-{self.fill_calls}"
        self.operations.append(stage)
        if self.fail_stage == stage:
            raise RuntimeError(f"injected {stage} failure")

    async def press(self, _key: str) -> None:
        self.press_calls += 1
        stage = f"press-{self.press_calls}"
        self.operations.append(stage)
        if self.fail_stage == stage:
            raise RuntimeError(f"injected {stage} failure")


class _FixtureFakeSink:
    capture_enabled = True

    def __init__(self, *, fail_boundary: str | None = None):
        self.fail_boundary = fail_boundary
        self.markers: list[dict[str, Any]] = []

    def emit_marker(self, marker: dict[str, Any]) -> None:
        if marker["boundary"] == self.fail_boundary:
            raise RuntimeError(f"injected {self.fail_boundary} marker failure")
        self.markers.append(marker)


class _FixtureFakeTrace:
    def __init__(self, records: list[dict[str, Any]], *, fail_after_baseline: bool = False):
        self.records = records
        self.calls = 0
        self.fail_after_baseline = fail_after_baseline

    def __call__(self, _path: Path) -> list[dict[str, Any]]:
        self.calls += 1
        if self.fail_after_baseline and self.calls > 1:
            raise RuntimeError("injected trace read failure")
        return [] if self.calls == 1 else self.records


@pytest.mark.asyncio
async def test_fixture_fake_joins_browser_boundaries_to_one_trace_response(tmp_path) -> None:
    fixture = parse_fixture_plan(canonical_plan_json()).fixtures[0]
    messages = [
        _timed_message("bot-started-speaking", 10.0),
        _timed_message("bot-transcription", 20.0, text="The red balloon rose."),
        _timed_message("bot-stopped-speaking", 30.0),
    ]
    operations: list[str] = []
    page = _FixtureFakePage([messages], operations)
    text_box = _FixtureFakeTextBox(operations)
    result = blank_fixture_result(fixture)
    snapshots: list[str] = []

    def persist() -> None:
        operations.append("persist")
        snapshots.append(result["completion_status"])

    completed = await _run_step2_fixture(
        page,
        text_box,
        _FixtureFakeSink(),
        fixture,
        result,
        persist,
        tmp_path / "trace.jsonl",
        trace_reader=_FixtureFakeTrace(_fake_response_trace("response-1", 1.0)),
        quiescence_window_ms=RTVI_QUIESCENCE_MIN_WINDOW_MS,
    )

    assert completed is True
    assert operations[:2] == ["persist", "baseline"]
    assert snapshots[0] == "running"
    assert result["response_ids"] == ["response-1"]
    assert result["response_attribution"]["primary_response"]["response_id"] == "response-1"
    assert result["transcript_gate_evidence"] == {
        "joined_browser_transcript": "The red balloon rose.",
        "response_owned_traced_text": "The red balloon rose.",
        "transcript_equal": True,
        "mismatch_reason": None,
    }
    assert result["send_attempt_wall_time_s"] <= result["send_wall_time_s"]


@pytest.mark.asyncio
async def test_fixture_fake_completes_truncated_l1_without_browser_transcript_flush(
    tmp_path: Path,
) -> None:
    fixture = parse_fixture_plan(canonical_plan_json()).fixtures[3]
    truncated_text = "The quick brown fox jumps over the lazy dog while seventeen green dragons"
    messages = [
        _timed_message("bot-started-speaking", 10.0),
        _timed_message("bot-stopped-speaking", 30.0),
    ]
    result = blank_fixture_result(fixture)

    completed = await _run_step2_fixture(
        _FixtureFakePage([messages], []),
        _FixtureFakeTextBox([]),
        _FixtureFakeSink(),
        fixture,
        result,
        lambda: None,
        tmp_path / "trace.jsonl",
        trace_reader=_FixtureFakeTrace(
            _fake_response_trace("response-L1-2", 1.0, output_text=truncated_text)
        ),
        quiescence_window_ms=RTVI_QUIESCENCE_MIN_WINDOW_MS,
    )

    assert completed is True
    assert result["completion_status"] == "completed"
    assert result["transcript_evidence"] == []
    assert result["response_attribution"]["primary_response"]["output_text"] == truncated_text
    assert result["response_attribution"]["primary_response"]["transcript_evidence"] == []
    assert result["transcript_gate_evidence"] == {
        "joined_browser_transcript": "",
        "response_owned_traced_text": truncated_text,
        "transcript_equal": False,
        "mismatch_reason": (
            "joined browser transcript is empty while response-owned traced text is nonempty"
        ),
    }
    assert result["quiescence_barrier"]["passed"] is True


@pytest.mark.asyncio
async def test_fixture_fake_excludes_stale_prior_transcript_and_records_gate_mismatch(
    tmp_path,
) -> None:
    fixture = parse_fixture_plan(canonical_plan_json()).fixtures[0]
    fixture = replace(fixture, wait=WaitCriteria(fixture.wait.event, 0.001, True))
    messages = [
        _timed_message("bot-transcription", 8.0, text="stale prior response"),
        _timed_message("bot-started-speaking", 10.0),
        _timed_message("bot-stopped-speaking", 30.0),
    ]
    result = blank_fixture_result(fixture)

    completed = await _run_step2_fixture(
        _FixtureFakePage([messages], []),
        _FixtureFakeTextBox([]),
        _FixtureFakeSink(),
        fixture,
        result,
        lambda: None,
        tmp_path / "trace.jsonl",
        trace_reader=_FixtureFakeTrace(_fake_response_trace("response-1", 1.0)),
    )

    assert completed is True
    assert result["transcript_evidence"] == []
    assert result["transcript_gate_evidence"]["transcript_equal"] is False
    assert result["transcript_gate_evidence"]["joined_browser_transcript"] == ""


@pytest.mark.asyncio
async def test_fixture_fake_rejects_repeated_s1_identical_bytes_late_transcript(
    tmp_path: Path,
) -> None:
    """A late S1-1 transcript makes its terminal drain fail before S1-2 can send."""

    fixture = parse_fixture_plan(canonical_plan_json()).fixtures[0]
    complete = [
        _timed_message("bot-started-speaking", 10.0),
        _timed_message("bot-transcription", 20.0, text=fixture.typed_text),
        _timed_message("bot-stopped-speaking", 30.0),
    ]
    late_identical = complete + [_timed_message("bot-transcription", 31.0, text=fixture.typed_text)]
    result = blank_fixture_result(fixture)

    with pytest.raises(AssertionError, match="quiescence barrier"):
        await _run_step2_fixture(
            _FixtureFakePage([complete, late_identical], []),
            _FixtureFakeTextBox([]),
            _FixtureFakeSink(),
            fixture,
            result,
            lambda: None,
            tmp_path / "trace.jsonl",
            trace_reader=_FixtureFakeTrace(
                _fake_response_trace("response-S1-1", 1.0, output_text=fixture.typed_text)
            ),
            quiescence_window_ms=RTVI_QUIESCENCE_MIN_WINDOW_MS,
        )

    assert result["quiescence_barrier"]["bot_message_count"] == 1
    assert result["quiescence_barrier"]["bot_message_types"] == ["bot-transcription"]
    assert result["quiescence_barrier"]["passed"] is False


@pytest.mark.asyncio
async def test_fixture_fake_rejects_bracket_received_before_own_send_edge(
    tmp_path: Path,
) -> None:
    fixture = parse_fixture_plan(canonical_plan_json()).fixtures[0]
    messages = [
        _timed_message("bot-started-speaking", 4.0),
        _timed_message("bot-transcription", 20.0, text="answer"),
        _timed_message("bot-stopped-speaking", 30.0),
    ]
    with pytest.raises(ValueError, match="before its send edge"):
        await _run_step2_fixture(
            _FixtureFakePage([messages], []),
            _FixtureFakeTextBox([]),
            _FixtureFakeSink(),
            fixture,
            blank_fixture_result(fixture),
            lambda: None,
            tmp_path / "trace.jsonl",
            trace_reader=_FixtureFakeTrace(
                _fake_response_trace("response-1", 1.0, output_text="answer")
            ),
            quiescence_window_ms=RTVI_QUIESCENCE_MIN_WINDOW_MS,
        )


@pytest.mark.parametrize(
    ("probe", "messages", "output_text", "outcome"),
    [
        (
            "before-start-equal-clock",
            [
                _timed_message("bot-transcription", 10.0, text="current response"),
                _timed_message("bot-started-speaking", 10.0),
                _timed_message("bot-stopped-speaking", 30.0),
            ],
            "current response",
            "completed-mismatch",
        ),
        (
            "before-start-inside-clock-window",
            [
                _timed_message("bot-transcription", 20.0, text="current response"),
                _timed_message("bot-started-speaking", 10.0),
                _timed_message("bot-stopped-speaking", 30.0),
            ],
            "current response",
            "clock-error",
        ),
        (
            "after-stop-equal-clock",
            [
                _timed_message("bot-started-speaking", 10.0),
                _timed_message("bot-stopped-speaking", 30.0),
                _timed_message("bot-transcription", 30.0, text="current response"),
            ],
            "current response",
            "quiescence-error",
        ),
        (
            "late-prior-inside-current-window",
            [
                _timed_message("bot-started-speaking", 10.0),
                _timed_message("bot-transcription", 20.0, text="late prior response"),
                _timed_message("bot-stopped-speaking", 30.0),
            ],
            "current response",
            "completed-mismatch",
        ),
    ],
    ids=lambda value: value if isinstance(value, str) else None,
)
@pytest.mark.asyncio
async def test_fixture_fake_preserves_round3_noncausal_transcript_boundaries(
    tmp_path: Path,
    probe: str,
    messages: list[dict[str, Any]],
    output_text: str,
    outcome: str,
) -> None:
    fixture = parse_fixture_plan(canonical_plan_json()).fixtures[0]
    fixture = replace(fixture, wait=WaitCriteria(fixture.wait.event, 0.001, True))
    result = blank_fixture_result(fixture)

    async def run() -> bool:
        return await _run_step2_fixture(
            _FixtureFakePage([messages], []),
            _FixtureFakeTextBox([]),
            _FixtureFakeSink(),
            fixture,
            result,
            lambda: None,
            tmp_path / f"{probe}.jsonl",
            trace_reader=_FixtureFakeTrace(
                _fake_response_trace("response-1", 1.0, output_text=output_text)
            ),
        )

    if outcome == "clock-error":
        with pytest.raises(ValueError, match="clock"):
            await run()
    elif outcome == "quiescence-error":
        with pytest.raises(AssertionError, match="quiescence barrier"):
            await run()
    else:
        assert await run() is True
        assert result["transcript_gate_evidence"]["transcript_equal"] is False


@pytest.mark.parametrize(
    ("browser_text", "traced_text", "expected_adjudication"),
    [
        ("The answer is four.", "The answer is four.", True),
        (None, " Two plus two is four.", True),
        (None, "The answer is five.", False),
    ],
)
@pytest.mark.asyncio
async def test_fixture_fake_i1_completes_cancel_then_replace_with_recorded_answer_evidence(
    tmp_path: Path,
    browser_text: str | None,
    traced_text: str,
    expected_adjudication: bool,
) -> None:
    fixture = parse_fixture_plan(canonical_plan_json()).fixtures[6]
    first = [_timed_message("bot-started-speaking", 10.0)]
    complete = first + [
        _timed_message("bot-interrupted", 4015.0),
        _timed_message("bot-stopped-speaking", 4017.0),
        _timed_message("bot-started-speaking", 4020.0),
    ]
    if browser_text is not None:
        complete.append(_timed_message("bot-transcription", 4030.0, text=browser_text))
    complete.append(_timed_message("bot-stopped-speaking", 4040.0))
    records = _fake_response_trace("original", 1.0, cancelled=True)
    records += _fake_response_trace(
        "replacement",
        2.0,
        output_text=traced_text,
    )
    operations: list[str] = []
    result = blank_fixture_result(fixture)

    completed = await _run_step2_fixture(
        _FixtureFakePage([first, complete], operations),
        _FixtureFakeTextBox(operations),
        _FixtureFakeSink(),
        fixture,
        result,
        lambda: operations.append("persist"),
        tmp_path / "trace.jsonl",
        trace_reader=_FixtureFakeTrace(records),
        quiescence_window_ms=RTVI_QUIESCENCE_MIN_WINDOW_MS,
    )

    assert completed is True
    assert result["response_ids"] == ["original", "replacement"]
    assert result["response_attribution"]["original_response"]["terminal"]["status"] == "cancelled"
    assert (
        result["response_attribution"]["replacement_response"]["terminal"]["status"] == "completed"
    )
    assert result["interruption"]["actual_offset_s"] == 4.0
    expected_transcripts = [] if browser_text is None else [browser_text]
    assert result["interruption"]["second_answer_transcript_evidence"] == expected_transcripts
    assert result["interruption"]["second_answer_gate_evidence"] == {
        "joined_browser_transcript": browser_text or "",
        "response_owned_traced_text": traced_text,
        "second_answer_adjudication": expected_adjudication,
    }


@pytest.mark.asyncio
async def test_attempt_5_retained_i1_trace_correlates_and_completes_without_browser_text(
    tmp_path: Path,
) -> None:
    artifact = (
        Path(__file__).resolve().parents[2]
        / "reports/step2-live/session-20260813T071818Z/driver/pipecat-playout-0001.jsonl"
    )
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == (
        "93996e3a788d0cb7832ce16a9646eeb7487f4631644263716cc12d1ff625a341"
    )
    retained_records = read_playout_trace(artifact)
    typed_sends = [
        index
        for index, record in enumerate(retained_records)
        if record.get("type") == "input_audio_buffer.speech_started"
        and record.get("source") == "typed"
        and record.get("response_id") is None
    ]
    i1_records = retained_records[typed_sends[-2] :]
    retained_attribution = correlate_response_lifecycles(i1_records, interruption=True)
    assert retained_attribution is not None

    fixture = parse_fixture_plan(canonical_plan_json()).fixtures[6]
    first = [_timed_message("bot-started-speaking", 10.0)]
    complete = first + [
        _timed_message("bot-interrupted", 4015.0),
        _timed_message("bot-stopped-speaking", 4017.0),
        _timed_message("bot-started-speaking", 4020.0),
        _timed_message("bot-stopped-speaking", 4040.0),
    ]
    result = blank_fixture_result(fixture)
    completed = await _run_step2_fixture(
        _FixtureFakePage([first, complete], []),
        _FixtureFakeTextBox([]),
        _FixtureFakeSink(),
        fixture,
        result,
        lambda: None,
        artifact,
        trace_reader=_FixtureFakeTrace(i1_records),
        quiescence_window_ms=RTVI_QUIESCENCE_MIN_WINDOW_MS,
    )

    assert completed is True
    assert result["completion_status"] == "completed"
    assert result["response_attribution"]["original_response"]["terminal"]["status"] == (
        "cancelled"
    )
    assert result["response_attribution"]["replacement_response"]["terminal"]["status"] == (
        "completed"
    )
    assert result["interruption"]["second_answer_transcript_evidence"] == []
    assert result["interruption"]["second_answer_gate_evidence"] == {
        "joined_browser_transcript": "",
        "response_owned_traced_text": " Two plus two is four.",
        "second_answer_adjudication": True,
    }
    assert result["quiescence_barrier"]["passed"] is True


@pytest.mark.asyncio
async def test_attempt_7_retained_i1_observation_matches_completion_anchors(
    tmp_path: Path,
) -> None:
    summary_path = (
        Path(__file__).resolve().parents[2]
        / "reports/step2-live/session-20260813T075531Z/driver/fixture-summary.json"
    )
    assert hashlib.sha256(summary_path.read_bytes()).hexdigest() == (
        "4799d62e040158f1cfea74c1ce99f5976ff1950b09ca0d5eddcd1993f3966133"
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    retained = next(item for item in summary["fixtures"] if item["fixture_id"] == "I1")
    observed = retained["observed_rtvi_messages"]
    messages = [
        {
            "receivedPerformanceMs": item["received_performance_ms"],
            "receivedWallTimeS": item["received_wall_time_s"],
            "message": {"type": item["type"]},
        }
        for item in observed
    ]
    baseline = retained["rtvi_window_start_index"]
    assert rtvi_message_diagnostics(messages, start_event_index=baseline) == observed
    marker = retained["interruption"]["rtvi_send_marker_event_index"]
    before_interruption_send = messages[: marker - baseline]
    first_start = next(
        item for item in messages if item["message"]["type"] == "bot-started-speaking"
    )
    marker_performance_ms = first_start["receivedPerformanceMs"] + (
        retained["interruption"]["actual_offset_s"] * 1000.0
    )
    final_stop = next(
        item for item in reversed(messages) if item["message"]["type"] == "bot-stopped-speaking"
    )
    fixture = parse_fixture_plan(canonical_plan_json()).fixtures[6]
    fixture = replace(fixture, wait=WaitCriteria(fixture.wait.event, 0.02, True))
    result = blank_fixture_result(fixture)
    records = _fake_response_trace("original", 1.0, cancelled=True)
    records += _fake_response_trace("replacement", 2.0, output_text=" Two plus two is four.")

    completed = await _run_step2_fixture(
        _FixtureFakePage(
            [before_interruption_send, messages],
            [],
            baseline_length=baseline,
            performance_now_ms=marker_performance_ms,
            boundary_clocks=[
                {
                    "performance_ms": observed[0]["received_performance_ms"] - 1.0,
                    "wall_time_s": observed[0]["received_wall_time_s"] - 0.001,
                },
                {
                    "performance_ms": marker_performance_ms,
                    "wall_time_s": retained["interruption"]["send_attempt_wall_time_s"],
                },
                {
                    "performance_ms": final_stop["receivedPerformanceMs"]
                    + RTVI_QUIESCENCE_MIN_WINDOW_MS,
                    "wall_time_s": final_stop["receivedWallTimeS"]
                    + RTVI_QUIESCENCE_MIN_WINDOW_MS / 1000.0,
                },
            ],
        ),
        _FixtureFakeTextBox([]),
        _FixtureFakeSink(),
        fixture,
        result,
        lambda: None,
        tmp_path / "trace.jsonl",
        trace_reader=_FixtureFakeTrace(records),
        quiescence_window_ms=RTVI_QUIESCENCE_MIN_WINDOW_MS,
    )

    assert completed is True
    assert result["response_attribution"]["original_response"]["terminal"]["status"] == (
        "cancelled"
    )
    assert result["response_attribution"]["replacement_response"]["terminal"]["status"] == (
        "completed"
    )
    assert result["interruption"]["rtvi_send_marker_event_index"] == 300
    original_bracket = result["response_attribution"]["original_response"]["rtvi_event_bracket"]
    replacement_bracket = result["response_attribution"]["replacement_response"][
        "rtvi_event_bracket"
    ]
    assert (
        original_bracket["start_event_index"],
        original_bracket["terminal_event_index"],
        original_bracket["terminal_event_type"],
    ) == (9, 27, "bot-stopped-speaking")
    assert (
        replacement_bracket["start_event_index"],
        replacement_bracket["terminal_event_index"],
    ) == (37, 50)
    assert result["quiescence_barrier"]["passed"] is True


@pytest.mark.parametrize("bot_interrupted_count", [0, 1, 3])
@pytest.mark.parametrize("include_user_speech_echoes", [False, True])
@pytest.mark.asyncio
async def test_i1_completion_anchors_ignore_interruption_count_and_user_speech_echoes(
    tmp_path: Path,
    bot_interrupted_count: int,
    include_user_speech_echoes: bool,
) -> None:
    fixture = parse_fixture_plan(canonical_plan_json()).fixtures[6]
    first = [_timed_message("bot-started-speaking", 10.0)]
    ignored = [
        _timed_message("bot-interrupted", 4011.0 + index) for index in range(bot_interrupted_count)
    ]
    ignored.extend(
        [
            _timed_message("bot-llm-text", 4014.0),
            _timed_message("metrics", 4015.0),
            _timed_message("bot-tts-stopped", 4016.0),
        ]
    )
    if include_user_speech_echoes:
        ignored.extend(
            [
                _timed_message("user-started-speaking", 4017.0),
                _timed_message("user-stopped-speaking", 4018.0),
            ]
        )
    complete = (
        first
        + ignored
        + [
            _timed_message("bot-stopped-speaking", 4020.0),
            _timed_message("bot-started-speaking", 4030.0),
            _timed_message("bot-transcription", 4040.0, text="Two plus two is four."),
            _timed_message("bot-stopped-speaking", 4050.0),
        ]
    )
    records = _fake_response_trace("original", 1.0, cancelled=True)
    records += _fake_response_trace("replacement", 2.0, output_text=" Two plus two is four.")
    result = blank_fixture_result(fixture)

    completed = await _run_step2_fixture(
        _FixtureFakePage([first, complete], []),
        _FixtureFakeTextBox([]),
        _FixtureFakeSink(),
        fixture,
        result,
        lambda: None,
        tmp_path / "trace.jsonl",
        trace_reader=_FixtureFakeTrace(records),
        quiescence_window_ms=RTVI_QUIESCENCE_MIN_WINDOW_MS,
    )

    assert completed is True
    assert result["completion_status"] == "completed"
    assert (
        result["response_attribution"]["original_response"]["rtvi_event_bracket"][
            "terminal_event_type"
        ]
        == "bot-stopped-speaking"
    )
    assert result["quiescence_barrier"]["passed"] is True


@pytest.mark.asyncio
async def test_i1_uses_event_index_when_original_stop_precedes_send_clock_by_56ms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = parse_fixture_plan(canonical_plan_json()).fixtures[6]
    first = [_timed_message("bot-started-speaking", 10.0)]
    complete = first + [
        _timed_message("bot-stopped-speaking", 4066.0),
        _timed_message("bot-started-speaking", 4070.0),
        _timed_message("bot-stopped-speaking", 4090.0),
    ]
    records = _fake_response_trace("original", 1.0, cancelled=True)
    records += _fake_response_trace("replacement", 2.0, output_text=" Two plus two is four.")
    wall_times = iter((99.0, 99.1, 104.0, 104.122, 110.0))
    monkeypatch.setattr(time, "time", lambda: next(wall_times))
    result = blank_fixture_result(fixture)

    completed = await _run_step2_fixture(
        _FixtureFakePage([first, complete], []),
        _FixtureFakeTextBox([]),
        _FixtureFakeSink(),
        fixture,
        result,
        lambda: None,
        tmp_path / "trace.jsonl",
        trace_reader=_FixtureFakeTrace(records),
        quiescence_window_ms=RTVI_QUIESCENCE_MIN_WINDOW_MS,
    )

    original_terminal = result["response_attribution"]["original_response"]
    assert completed is True
    assert original_terminal["rtvi_event_bracket"]["terminal_event_type"] == (
        "bot-stopped-speaking"
    )
    assert result["interruption"]["rtvi_send_marker_event_index"] == 1
    assert result["interruption"]["send_wall_time_s"] - original_terminal[
        "interrupted_wall_time_s"
    ] == pytest.approx(0.056)


@pytest.mark.parametrize(
    "stage",
    [
        "baseline",
        "fill-1",
        "boundary-clock-1",
        "press-1",
        "poll-1",
        "boundary-clock-2",
        "trace-reader",
        "poll-sleep",
    ],
)
@pytest.mark.asyncio
async def test_fixture_fake_injected_failure_at_each_ordinary_await_boundary(
    tmp_path, stage: str
) -> None:
    fixture = parse_fixture_plan(canonical_plan_json()).fixtures[0]
    messages = [
        _timed_message("bot-started-speaking", 10.0),
        _timed_message("bot-transcription", 20.0, text="answer"),
        _timed_message("bot-stopped-speaking", 30.0),
    ]
    operations: list[str] = []
    page_stage = (
        stage
        if stage in {"baseline", "boundary-clock-1", "press-1", "poll-1", "boundary-clock-2"}
        else None
    )
    text_stage = stage if stage in {"fill-1", "press-1"} else None
    if stage == "press-1":
        page_stage = None
    batches = [[]] if stage == "poll-sleep" else [messages]

    async def failing_sleep(_seconds: float) -> None:
        raise RuntimeError("injected poll-sleep failure")

    with pytest.raises(RuntimeError, match="injected"):
        await _run_step2_fixture(
            _FixtureFakePage(batches, operations, fail_stage=page_stage),
            _FixtureFakeTextBox(operations, fail_stage=text_stage),
            _FixtureFakeSink(),
            fixture,
            blank_fixture_result(fixture),
            lambda: operations.append("persist"),
            tmp_path / "trace.jsonl",
            trace_reader=_FixtureFakeTrace(
                _fake_response_trace("response-1", 1.0, output_text="answer"),
                fail_after_baseline=stage == "trace-reader",
            ),
            sleep=failing_sleep if stage == "poll-sleep" else asyncio.sleep,
        )


@pytest.mark.asyncio
async def test_fixture_fake_records_marker_emission_failure(tmp_path) -> None:
    fixture = parse_fixture_plan(canonical_plan_json()).fixtures[0]
    result = blank_fixture_result(fixture)

    with pytest.raises(RuntimeError, match="start marker"):
        await _run_step2_fixture(
            _FixtureFakePage([[]], []),
            _FixtureFakeTextBox([]),
            _FixtureFakeSink(fail_boundary="start"),
            fixture,
            result,
            lambda: None,
            tmp_path / "trace.jsonl",
            trace_reader=_FixtureFakeTrace([]),
        )
    assert result["marker_emission"]["start"] == {
        "status": "failed",
        "error": "RuntimeError('injected start marker failure')",
    }
    assert result["rtvi_window_start_index"] == 0
    assert result["observed_rtvi_messages"] == []
    assert result["rtvi_observation_error"] is None


@pytest.mark.parametrize(
    "stage",
    [
        "performance-now",
        "deadline-wait",
        "fill-2",
        "boundary-clock-2",
        "press-2",
        "poll-2",
    ],
)
@pytest.mark.asyncio
async def test_fixture_fake_injected_failure_at_each_i1_await_boundary(
    tmp_path, stage: str
) -> None:
    fixture = parse_fixture_plan(canonical_plan_json()).fixtures[6]
    first = [_timed_message("bot-started-speaking", 10.0)]
    complete = first + [
        _timed_message("bot-interrupted", 4015.0),
        _timed_message("bot-started-speaking", 4020.0),
        _timed_message("bot-transcription", 4030.0, text="four"),
        _timed_message("bot-stopped-speaking", 4040.0),
    ]
    operations: list[str] = []
    page_stage = stage if stage in {"performance-now", "boundary-clock-2", "poll-2"} else None
    text_stage = stage if stage in {"fill-2", "press-2"} else None

    async def failing_deadline(*_args: Any, **_kwargs: Any) -> float:
        raise RuntimeError("injected deadline-wait failure")

    with pytest.raises(RuntimeError, match="injected"):
        await _run_step2_fixture(
            _FixtureFakePage([first, complete], operations, fail_stage=page_stage),
            _FixtureFakeTextBox(operations, fail_stage=text_stage),
            _FixtureFakeSink(),
            fixture,
            blank_fixture_result(fixture),
            lambda: operations.append("persist"),
            tmp_path / "trace.jsonl",
            trace_reader=_FixtureFakeTrace(
                _fake_response_trace("original", 1.0, cancelled=True)
                + _fake_response_trace("replacement", 2.0)
            ),
            deadline_wait=(
                failing_deadline if stage == "deadline-wait" else wait_for_interruption_deadline
            ),
        )


@pytest.mark.parametrize("probe", ["stale-trace", "extra-browser-start"])
@pytest.mark.asyncio
async def test_fixture_fake_rejects_stale_cross_fixture_attribution(tmp_path, probe: str) -> None:
    fixture = parse_fixture_plan(canonical_plan_json()).fixtures[0]
    messages = [
        _timed_message("bot-started-speaking", 10.0),
        _timed_message("bot-transcription", 20.0, text="answer"),
        _timed_message("bot-stopped-speaking", 30.0),
    ]
    records = _fake_response_trace("response-1", 1.0)
    if probe == "stale-trace":
        records.insert(
            0,
            _fake_playout_event("response.done", 0.9, response_id="stale", status="completed"),
        )
    else:
        messages.insert(0, _timed_message("bot-started-speaking", 9.0))

    with pytest.raises((ValueError, AssertionError), match="stale|foreign|extra"):
        await _run_step2_fixture(
            _FixtureFakePage([messages], []),
            _FixtureFakeTextBox([]),
            _FixtureFakeSink(),
            fixture,
            blank_fixture_result(fixture),
            lambda: None,
            tmp_path / "trace.jsonl",
            trace_reader=_FixtureFakeTrace(records),
        )


@pytest.mark.parametrize("source", ["rtvi", "playout"])
@pytest.mark.asyncio
async def test_fixture_cursor_rejects_transitions_arriving_between_fixture_windows(
    tmp_path, source: str
) -> None:
    fixture = parse_fixture_plan(canonical_plan_json()).fixtures[1]
    gap_messages = [_timed_message("bot-started-speaking", 9.0)]
    gap_trace = _fake_response_trace("late-response", 0.5)
    page = _FixtureFakePage(
        [gap_messages],
        [],
        baseline_length=1 if source == "rtvi" else 0,
    )

    with pytest.raises(AssertionError, match="between fixtures"):
        await _run_step2_fixture(
            page,
            _FixtureFakeTextBox([]),
            _FixtureFakeSink(),
            fixture,
            blank_fixture_result(fixture),
            lambda: None,
            tmp_path / "trace.jsonl",
            trace_reader=(lambda _path: gap_trace if source == "playout" else []),
            attribution_cursor={"rtvi_index": 0, "playout_index": 0},
        )


@pytest.mark.parametrize("marker_fails", [False, True])
@pytest.mark.asyncio
async def test_fixture_fake_timeout_persists_terminal_and_marker_status(
    tmp_path, marker_fails: bool
) -> None:
    fixture = parse_fixture_plan(canonical_plan_json()).fixtures[0]
    fixture = replace(fixture, wait=WaitCriteria(fixture.wait.event, 0.001, True))
    result = blank_fixture_result(fixture)
    snapshots: list[dict[str, Any]] = []

    def persist() -> None:
        snapshots.append(copy.deepcopy(result))

    completed = await _run_step2_fixture(
        _FixtureFakePage([[]], []),
        _FixtureFakeTextBox([]),
        _FixtureFakeSink(fail_boundary="timeout" if marker_fails else None),
        fixture,
        result,
        persist,
        tmp_path / "trace.jsonl",
        trace_reader=_FixtureFakeTrace([]),
        sleep=asyncio.sleep,
    )

    assert completed is False
    assert snapshots[-1]["completion_status"] == "timeout"
    assert snapshots[-1]["rtvi_window_start_index"] == 0
    assert snapshots[-1]["observed_rtvi_messages"] == []
    assert snapshots[-1]["rtvi_observation_error"] is None
    expected = "failed" if marker_fails else "succeeded"
    assert snapshots[-1]["marker_emission"]["timeout"]["status"] == expected


@pytest.mark.asyncio
async def test_fixture_timeout_retains_raw_ordered_rtvi_window(tmp_path: Path) -> None:
    fixture = parse_fixture_plan(canonical_plan_json()).fixtures[0]
    fixture = replace(fixture, wait=WaitCriteria(fixture.wait.event, 0.001, True))
    observed = [_timed_message("bot-started-speaking", 10.0)]
    result = blank_fixture_result(fixture)

    completed = await _run_step2_fixture(
        _FixtureFakePage([observed], []),
        _FixtureFakeTextBox([]),
        _FixtureFakeSink(),
        fixture,
        result,
        lambda: None,
        tmp_path / "trace.jsonl",
        trace_reader=_FixtureFakeTrace([]),
    )

    assert completed is False
    assert result["observed_rtvi_messages"] == [
        {
            "event_index": 0,
            "type": "bot-started-speaking",
            "received_performance_ms": 10.0,
            "received_wall_time_s": 100.01,
        }
    ]


class _SetupFakeBase:
    def __init__(self, operations: list[str], fail_stage: str | None):
        self.operations = operations
        self.fail_stage = fail_stage

    async def step(self, stage: str, value: Any = None) -> Any:
        self.operations.append(stage)
        if self.fail_stage == stage:
            raise RuntimeError(f"injected {stage} failure")
        return value


class _SetupFakeControl(_SetupFakeBase):
    def __init__(self, operations: list[str], fail_stage: str | None, prefix: str):
        super().__init__(operations, fail_stage)
        self.prefix = prefix

    async def click(self) -> None:
        await self.step(f"{self.prefix}-click")

    async def wait_for(self, **_kwargs: Any) -> None:
        if self.prefix == "disconnect" and self.fail_stage == "retry-wait":
            raise PlaywrightTimeoutError("injected disconnect timeout")
        await self.step(f"{self.prefix}-wait")

    async def is_enabled(self) -> bool:
        return await self.step("is-enabled", True)

    async def count(self) -> int:
        return await self.step(f"{self.prefix}-count", 1)

    async def is_visible(self) -> bool:
        return await self.step(f"{self.prefix}-visible", True)


class _SetupFakePage(_SetupFakeBase):
    def on(self, *_args: Any) -> None:
        return None

    async def goto(self, *_args: Any, **_kwargs: Any) -> None:
        await self.step("goto")

    async def evaluate(self, script: str) -> dict[str, Any]:
        if "peer_connection_present" in script:
            return await self.step(
                "peer-state",
                {
                    "peer_connection_present": True,
                    "connection_state": "closed",
                    "signaling_state": "closed",
                },
            )
        return await self.step("capture-evaluate", {})

    def get_by_role(self, _role: str, *, name: str, exact: bool) -> _SetupFakeControl:
        del exact
        return _SetupFakeControl(self.operations, self.fail_stage, name.lower())

    def locator(self, _selector: str) -> _SetupFakeControl:
        return _SetupFakeControl(self.operations, self.fail_stage, "text")

    async def wait_for_timeout(self, _milliseconds: float) -> None:
        await self.step("retry-wait")

    def is_closed(self) -> bool:
        return False


class _SetupFakeContext(_SetupFakeBase):
    def __init__(self, operations: list[str], fail_stage: str | None):
        super().__init__(operations, fail_stage)
        self.page = _SetupFakePage(operations, fail_stage)

    async def add_init_script(self, _script: str) -> None:
        await self.step("init-script")

    async def new_page(self) -> _SetupFakePage:
        return await self.step("new-page", self.page)

    async def expose_binding(self, *_args: Any) -> None:
        await self.step("expose-binding")

    async def close(self) -> None:
        await self.step("context-close")


class _SetupFakeBrowser(_SetupFakeBase):
    def __init__(self, operations: list[str], fail_stage: str | None):
        super().__init__(operations, fail_stage)
        self.context = _SetupFakeContext(operations, fail_stage)

    async def new_context(self, **_kwargs: Any) -> _SetupFakeContext:
        return await self.step("new-context", self.context)

    async def close(self) -> None:
        await self.step("browser-close")


class _SetupFakeChromium(_SetupFakeBase):
    def __init__(self, operations: list[str], fail_stage: str | None):
        super().__init__(operations, fail_stage)
        self.browsers: list[_SetupFakeBrowser] = []

    async def launch(self, **_kwargs: Any) -> _SetupFakeBrowser:
        browser = _SetupFakeBrowser(self.operations, self.fail_stage)
        self.browsers.append(browser)
        return await self.step("launch", browser)


class _SetupFakePlaywright:
    def __init__(self, operations: list[str], fail_stage: str | None):
        self.chromium = _SetupFakeChromium(operations, fail_stage)


@pytest.mark.parametrize(
    "stage",
    [
        "launch",
        "new-context",
        "init-script",
        "new-page",
        "goto",
        "expose-binding",
        "capture-evaluate",
        "connect-click",
        "disconnect-wait",
        "retry-wait",
        "text-wait",
        "is-enabled",
    ],
)
@pytest.mark.asyncio
async def test_setup_fake_injected_failure_at_every_awaited_boundary(tmp_path, stage: str) -> None:
    operations: list[str] = []
    state: dict[str, Any] = {}
    artifact = tmp_path / "browser.jsonl"

    with pytest.raises(RuntimeError, match=f"injected {stage}"):
        await _setup_step2_session(
            state=state,
            playwright=_SetupFakePlaywright(operations, stage),
            url="http://example.invalid/client/",
            microphone_wav=tmp_path / "mic.wav",
            artifact=artifact,
            capture_mode="on",
            browser_log=[],
        )

    _shutdown, _validation, _errors = await _teardown_step2_session(
        page=state.get("page"),
        sink=state.get("sink"),
        context=state.get("context"),
        browser=state.get("browser"),
        artifact=artifact,
        capture_enabled=True,
    )
    if state.get("context") is not None:
        assert "context-close" in operations
    if state.get("browser") is not None:
        assert "browser-close" in operations


class _TeardownFakeSink:
    capture_enabled = True

    def __init__(self, operations: list[str], *, close_fails: bool = False):
        self.operations = operations
        self.close_fails = close_fails

    def close(self) -> None:
        self.operations.append("sink-close")
        if self.close_fails:
            raise RuntimeError("injected sink-close failure")

    def ordinal_summary(self) -> dict[str, Any]:
        return {}

    def overhead(self) -> dict[str, Any]:
        return {"fixture_marker_records": 0}


@pytest.mark.parametrize(
    "stage",
    ["shutdown-evaluate", "sink-close", "artifact-load", "context-close", "browser-close"],
)
@pytest.mark.asyncio
async def test_teardown_fake_attempts_all_boundaries_and_retains_each_error(
    tmp_path, stage: str
) -> None:
    operations: list[str] = []
    artifact = tmp_path / "browser.jsonl"
    artifact.write_text("{}\n", encoding="utf-8")
    page = _SetupFakePage(operations, "capture-evaluate" if stage == "shutdown-evaluate" else None)
    context = _SetupFakeContext(operations, "context-close" if stage == "context-close" else None)
    browser = _SetupFakeBrowser(operations, "browser-close" if stage == "browser-close" else None)
    sink = _TeardownFakeSink(operations, close_fails=stage == "sink-close")

    def load(_path: Path, *, capture_enabled: bool) -> dict[str, Any]:
        del capture_enabled
        operations.append("artifact-load")
        if stage == "artifact-load":
            raise RuntimeError("injected artifact-load failure")
        return {"valid": True}

    _shutdown, _validation, errors = await _teardown_step2_session(
        page=page,
        sink=sink,
        context=context,
        browser=browser,
        artifact=artifact,
        capture_enabled=True,
        artifact_loader=load,
    )

    expected_error = "capture-evaluate" if stage == "shutdown-evaluate" else stage
    assert any(expected_error in error for error in errors)
    assert "sink-close" in operations
    assert "artifact-load" in operations
    assert "context-close" in operations
    assert "browser-close" in operations


@pytest.mark.asyncio
async def test_teardown_cleanly_disconnects_webrtc_before_context_close(tmp_path) -> None:
    operations: list[str] = []
    artifact = tmp_path / "browser.jsonl"
    artifact.write_text("{}\n", encoding="utf-8")
    page = _SetupFakePage(operations, None)
    context = _SetupFakeContext(operations, None)
    browser = _SetupFakeBrowser(operations, None)
    sink = _TeardownFakeSink(operations)

    def load(_path: Path, *, capture_enabled: bool) -> dict[str, Any]:
        del capture_enabled
        operations.append("artifact-load")
        return {"valid": True}

    shutdown, _validation, errors = await _teardown_step2_session(
        page=page,
        sink=sink,
        context=context,
        browser=browser,
        artifact=artifact,
        capture_enabled=True,
        artifact_loader=load,
    )

    assert errors == []
    assert shutdown is not None
    assert shutdown["webrtc_disconnect"]["completed"] is True
    assert operations.index("sink-close") < operations.index("disconnect-click")
    assert operations.index("disconnect-click") < operations.index("peer-state")
    assert operations.index("peer-state") < operations.index("context-close")


@pytest.mark.asyncio
async def test_playwright_exit_failure_is_explicit_terminalization_error() -> None:
    errors: list[str] = []

    class FailingPlaywright:
        async def stop(self) -> None:
            raise RuntimeError("injected exit failure")

    error = await _stop_playwright_recorded(FailingPlaywright(), errors)

    assert error == "Playwright exit failed: RuntimeError('injected exit failure')"
    assert errors == [error]


@pytest.mark.asyncio
async def test_playwright_entry_failure_still_produces_failed_terminal_summary() -> None:
    summary = {"status": "running", "completed_wall_time_s": None}
    persisted: list[dict[str, Any]] = []

    class FailingManager:
        async def start(self) -> None:
            raise RuntimeError("injected Playwright entry failure")

    failure: str | None = None
    try:
        await _start_playwright(FailingManager())
    except Exception as exc:  # noqa: BLE001 - mirrors the outer campaign boundary
        failure = repr(exc)
    finally:
        _terminalize_step2_summary(
            summary, failure, lambda: persisted.append(copy.deepcopy(summary))
        )

    assert failure == "RuntimeError('injected Playwright entry failure')"
    assert persisted[-1]["status"] == "failed"
    assert persisted[-1]["completed_wall_time_s"] is not None


@pytest.mark.asyncio
async def test_setup_fake_capture_sessions_have_isolated_state(tmp_path) -> None:
    operations: list[str] = []
    playwright = _SetupFakePlaywright(operations, None)
    states = [{}, {}]

    await _setup_step2_session(
        state=states[0],
        playwright=playwright,
        url="http://example.invalid/client/",
        microphone_wav=tmp_path / "mic.wav",
        artifact=tmp_path / "on.jsonl",
        capture_mode="on",
        browser_log=[],
    )
    await _setup_step2_session(
        state=states[1],
        playwright=playwright,
        url="http://example.invalid/client/",
        microphone_wav=tmp_path / "mic.wav",
        artifact=tmp_path / "off.jsonl",
        capture_mode="off",
        browser_log=[],
    )

    for key in ("browser", "context", "page", "sink"):
        assert states[0][key] is not states[1][key]
    assert states[0]["sink"].capture_enabled is True
    assert states[1]["sink"].capture_enabled is False
    for index, state in enumerate(states):
        await _teardown_step2_session(
            page=state["page"],
            sink=state["sink"],
            context=state["context"],
            browser=state["browser"],
            artifact=tmp_path / ("on.jsonl" if index == 0 else "off.jsonl"),
            capture_enabled=index == 0,
        )


@pytest.mark.asyncio
async def test_real_chromium_smallwebrtc_round_trip(tmp_path):
    if CHROMIUM and not Path(CHROMIUM).exists():
        pytest.skip(f"configured Chromium is not installed at {CHROMIUM}")

    microphone_wav = tmp_path / "fake-microphone.wav"
    _write_fake_microphone(microphone_wav)
    observed: dict[str, object] = {"audio_appends": 0}
    observed_rtvi: list[dict[str, object]] = []
    input_seen = asyncio.Event()

    async def model_handler(websocket):
        await websocket.send(
            json.dumps(
                {
                    "type": "session.created",
                    "event_id": "evt_created",
                    "protocol": {"name": "voicechat.realtime", "version": 3},
                    "capabilities": {"input_turn_detection": "client_smart_turn_v1"},
                    "session": {"id": "browser_e2e"},
                }
            )
        )
        responded = False
        session_configured = False
        async for raw in websocket:
            message = json.loads(raw)
            if message["type"] == "session.update":
                assert int(observed["audio_appends"]) == 0
                assert message["session"]["protocol_version"] == 3
                observed["session_update"] = message
                session_configured = True
                await websocket.send(
                    json.dumps(
                        {
                            "type": "session.updated",
                            "event_id": "evt_updated",
                            "session": {
                                "id": "browser_e2e",
                                "protocol_version": 3,
                            },
                        }
                    )
                )
            elif message["type"] == "input_audio_buffer.turn_start":
                await websocket.send(
                    json.dumps(
                        {
                            "type": "input_audio_buffer.turn_started",
                            "client_event_id": message["event_id"],
                            "client_turn_id": message["client_turn_id"],
                            "turn_id": f"turn_browser_{message['client_turn_id']}",
                        }
                    )
                )
            elif message["type"] == "input_audio_buffer.commit":
                await websocket.send(
                    json.dumps(
                        {
                            "type": "input_audio_buffer.committed",
                            "client_event_id": message["event_id"],
                            "client_turn_id": message["client_turn_id"],
                            "turn_id": f"turn_browser_{message['client_turn_id']}",
                        }
                    )
                )
            elif message["type"] == "input_audio_buffer.append":
                assert session_configured
                assert message["encoding"] == "pcm16"
                assert message["sample_rate"] == 16_000
                assert message["channels"] == 1
                observed["audio_appends"] = int(observed["audio_appends"]) + 1
                input_seen.set()
                if int(observed["audio_appends"]) >= 10 and not responded:
                    responded = True
                    turn_id = "turn_browser_1"
                    response_id = "response_browser_1"
                    server_events = [
                        {
                            "type": "input_audio_buffer.speech_started",
                            "turn_id": turn_id,
                        },
                        {
                            "type": "conversation.item.input_audio_transcription.delta",
                            "turn_id": turn_id,
                            "transcript": "browser microphone reached the bot",
                        },
                        {
                            "type": "input_audio_buffer.speech_stopped",
                            "turn_id": turn_id,
                        },
                        {
                            "type": "conversation.item.input_audio_transcription.completed",
                            "turn_id": turn_id,
                            "transcript": "browser microphone reached the bot",
                        },
                        {"type": "response.created", "response_id": response_id},
                        {
                            "type": "response.output_text.delta",
                            "response_id": response_id,
                            "delta": "The deterministic test passed.",
                        },
                    ]
                    for event in server_events:
                        await websocket.send(json.dumps(event))
                    pcm = _assistant_pcm()
                    chunk_bytes = 1764 * 2
                    for offset in range(0, len(pcm), chunk_bytes):
                        chunk = pcm[offset : offset + chunk_bytes]
                        await websocket.send(
                            json.dumps(
                                {
                                    "type": "response.output_audio.delta",
                                    "response_id": response_id,
                                    "encoding": "pcm16",
                                    "sample_rate": 22_050,
                                    "channels": 1,
                                    "delta": base64.b64encode(chunk).decode(),
                                }
                            )
                        )
                        await asyncio.sleep(0.02)
                    await websocket.send(
                        json.dumps({"type": "response.done", "response_id": response_id})
                    )
            elif message["type"] == "session.stop":
                await websocket.send(
                    json.dumps(
                        {
                            "type": "session.closed",
                            "status": "completed",
                            "reason": "client_requested",
                        }
                    )
                )
                return

    runner_port = _free_port()
    async with serve(model_handler, "127.0.0.1", 0) as model_server:
        model_port = model_server.sockets[0].getsockname()[1]
        repo_root = Path(__file__).resolve().parents[2]
        env = os.environ.copy()
        env["NEMOTRON_VOICECHAT_WS_URL"] = f"ws://127.0.0.1:{model_port}"
        # This voice-only mock gate must not load the large optional typed-
        # input model. Production keeps prewarm enabled and has a separate
        # real Pocket TTS typed-input qualification.
        env["VOICECHAT_TYPED_INPUT_PREWARM"] = "0"
        # The in-process unit suite imports NLTK before this subprocess and its
        # CWD-hardening can propagate a safe-path environment. Make the
        # application root explicit so the E2E launch is order-independent.
        env["PYTHONPATH"] = os.pathsep.join(
            filter(None, (str(repo_root / "src"), env.get("PYTHONPATH")))
        )
        runner = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "nemotron_voicechat_pipecat.demo",
            "-t",
            "webrtc",
            "--host",
            "127.0.0.1",
            "--port",
            str(runner_port),
            cwd=repo_root,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        runner_lines: list[str] = []

        async def drain_runner_output():
            while line := await runner.stdout.readline():
                runner_lines.append(line.decode(errors="replace").rstrip())

        runner_log_task = asyncio.create_task(drain_runner_output())
        browser_trace_page = None
        browser_trace_sink = None
        browser_capture_started = time.perf_counter()
        try:
            await _wait_http(f"http://127.0.0.1:{runner_port}/client/")
            async with async_playwright() as playwright:
                browser = await playwright.chromium.launch(
                    executable_path=CHROMIUM,
                    headless=True,
                    args=[
                        "--no-sandbox",
                        "--use-fake-device-for-media-stream",
                        "--use-fake-ui-for-media-stream",
                        f"--use-file-for-fake-audio-capture={microphone_wav}%noloop",
                        "--autoplay-policy=no-user-gesture-required",
                    ],
                )
                context = await browser.new_context(
                    # Pipecat's generic debug UI initializes both device classes
                    # before connecting even though this bot publishes audio only.
                    permissions=["microphone", "camera"],
                    base_url=f"http://127.0.0.1:{runner_port}",
                )
                await context.add_init_script(_rtc_probe_script("__voicechatTestPeerConnection"))
                page = await context.new_page()
                browser_trace_page = page
                browser_log: list[str] = []
                page.on("console", lambda message: browser_log.append(message.text))
                page.on("pageerror", lambda error: browser_log.append(f"pageerror: {error}"))
                await page.goto("/client/")
                browser_trace_path = tmp_path / "browser-playout.jsonl"
                browser_trace_sink = await _install_browser_playout_capture(
                    context,
                    page,
                    browser_trace_path,
                    peer_connection_slot="__voicechatTestPeerConnection",
                )
                connect = page.get_by_role("button", name="Connect", exact=True)
                disconnect = page.get_by_role("button", name="Disconnect", exact=True)
                # The generic UI initializes media devices asynchronously and
                # currently leaves Connect clickable during that transition.
                # A click in that narrow interval is intentionally a no-op, so
                # retry until the UI confirms that signaling actually began.
                for _ in range(5):
                    await connect.click()
                    try:
                        await disconnect.wait_for(timeout=3_000)
                        break
                    except PlaywrightTimeoutError:
                        await page.wait_for_timeout(250)
                else:
                    pytest.fail(
                        "Pipecat debug UI never entered connected state; "
                        f"browser_log={browser_log} runner_log={runner_lines[-80:]}"
                    )
                try:
                    await asyncio.wait_for(input_seen.wait(), timeout=20)
                except TimeoutError:
                    buttons = await page.get_by_role("button").all_inner_texts()
                    pytest.fail(
                        "browser sent no model audio; "
                        f"buttons={buttons} browser_log={browser_log} "
                        f"runner_log={runner_lines[-80:]}"
                    )
                await page.get_by_text(
                    "The deterministic test passed.", exact=False
                ).first.wait_for(timeout=20_000)
                observed_rtvi = await page.evaluate("() => window.__voicechatRtviMessages || []")

                async def playback_stats():
                    return await page.evaluate(
                        """async () => {
                          const pc = window.__voicechatTestPeerConnection;
                          let inbound = 0, outbound = 0;
                          if (pc) for (const report of (await pc.getStats()).values()) {
                            if (report.type === 'inbound-rtp' && report.kind === 'audio')
                              inbound += report.bytesReceived || 0;
                            if (report.type === 'outbound-rtp' && report.kind === 'audio')
                              outbound += report.bytesSent || 0;
                          }
                          const audio = document.querySelector('audio');
                          return { inbound, outbound,
                            tracks: audio?.srcObject?.getAudioTracks()?.length || 0,
                            currentTime: audio?.currentTime || 0 };
                        }"""
                    )

                stats = {"inbound": 0, "outbound": 0, "tracks": 0, "currentTime": 0}
                for _ in range(100):
                    stats = await playback_stats()
                    if stats["inbound"] > 0 and stats["outbound"] > 0:
                        break
                    await asyncio.sleep(0.05)
                await page.screenshot(path=str(tmp_path / "browser-e2e.png"))
                capture_overhead = await _shutdown_browser_playout_capture(page, browser_trace_sink)
                capture_overhead["test_elapsed_ms"] = (
                    time.perf_counter() - browser_capture_started
                ) * 1000.0
                print(
                    "BROWSER_PLAYOUT_CAPTURE_OVERHEAD "
                    + json.dumps(capture_overhead, sort_keys=True)
                )
                assert capture_overhead["shutdown_error"] is None
                assert capture_overhead["sink_close_error"] is None
                assert capture_overhead["persisted_last_ordinals"]
                assert (
                    capture_overhead["persisted_last_ordinals"]
                    == capture_overhead["acknowledged_final_ordinals"]
                    == capture_overhead["worklet_final_ordinals"]
                )
                browser_trace_sink = None
                # Exercise the browser-owned teardown path and give the
                # server-side peer event time to cancel its pipeline worker.
                await disconnect.click()
                await page.get_by_role("button", name="Connect", exact=True).wait_for(timeout=5_000)
                await asyncio.sleep(0.25)
                await context.close()
                await browser.close()

            update = observed["session_update"]
            assert update["session"]["protocol_version"] == 3
            assert update["session"]["instructions"].startswith("You are a helpful voice assistant")
            assert update["session"]["tools"][0]["name"] == "get_current_utc_time"
            assert int(observed["audio_appends"]) >= 10
            assert stats["tracks"] >= 1
            assert stats["outbound"] > 0
            assert stats["inbound"] > 0
            browser_trace = [
                json.loads(line)
                for line in browser_trace_path.read_text(encoding="utf-8").splitlines()
            ]
            assert any(
                record["type"] == "browser.playout.capability" and record["audio_worklet"] is True
                for record in browser_trace
            )
            assert any(
                record["type"] == "browser.audio_worklet.dequeue" for record in browser_trace
            )
            acknowledged = {
                record["worklet_id"]: record["final_ordinal"]
                for record in browser_trace
                if record["type"] == "browser.audio_worklet.flush_ack"
            }
            persisted_last = {
                worklet_id: max(
                    record["ordinal"]
                    for record in browser_trace
                    if record["type"] == "browser.audio_worklet.dequeue"
                    and record["worklet_id"] == worklet_id
                )
                for worklet_id in acknowledged
            }
            assert persisted_last == acknowledged
            assert any(
                message.get("type") == "bot-transcription"
                and "deterministic test passed"
                in str(message.get("data", {}).get("text", "")).lower()
                for message in observed_rtvi
            )
        finally:
            if browser_trace_page is not None:
                await _shutdown_browser_playout_capture(browser_trace_page, browser_trace_sink)
            if runner.returncode is None:
                runner.terminate()
                try:
                    await asyncio.wait_for(runner.wait(), timeout=10)
                except TimeoutError:
                    runner.kill()
                    await runner.wait()
            await runner_log_task
            if runner.returncode not in (0, -15):
                pytest.fail(
                    f"Pipecat runner failed ({runner.returncode}):\n" + "\n".join(runner_lines)
                )


@pytest.mark.browser_e2e
@pytest.mark.asyncio
async def test_live_playground_typed_turn_and_audio(tmp_path):
    """Real Playground/SmallWebRTC/strict-v3/Pocket/model integration gate."""

    url = os.environ.get("VOICECHAT_LIVE_PLAYGROUND_URL")
    if not url:
        pytest.skip("set VOICECHAT_LIVE_PLAYGROUND_URL for the live DGX gate")
    configured_microphone = os.environ.get("VOICECHAT_LIVE_MIC_WAV")
    if configured_microphone:
        microphone_wav = Path(configured_microphone).expanduser()
        if not microphone_wav.is_file():
            pytest.fail(f"VOICECHAT_LIVE_MIC_WAV does not exist: {microphone_wav}")
    else:
        microphone_wav = tmp_path / "silent-microphone.wav"
        _write_silent_microphone(microphone_wav)
    prompt = "What is two plus three? Answer in one short sentence."
    measure_second_voice_turn = os.environ.get("VOICECHAT_LIVE_SECOND_VOICE") == "1"
    tool_first = os.environ.get("VOICECHAT_LIVE_TOOL_FIRST") == "1"

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            executable_path=CHROMIUM,
            headless=True,
            args=[
                "--no-sandbox",
                "--use-fake-device-for-media-stream",
                "--use-fake-ui-for-media-stream",
                f"--use-file-for-fake-audio-capture={microphone_wav}%noloop",
                "--autoplay-policy=no-user-gesture-required",
            ],
        )
        context = await browser.new_context(permissions=["microphone", "camera"])
        await context.add_init_script(_rtc_probe_script("__voicechatLivePC"))
        page = await context.new_page()
        browser_log: list[str] = []
        browser_trace_sink = None
        capture_started = time.perf_counter()
        page.on("console", lambda message: browser_log.append(message.text))
        page.on("pageerror", lambda error: browser_log.append(f"pageerror: {error}"))
        try:
            await page.goto(url.rstrip("/") + "/", wait_until="load", timeout=30_000)
            # Install the media probe after the secure document exists but
            # before Connect requests the microphone.
            await page.evaluate(_voice_latency_probe_script())
            configured_playout_trace = os.environ.get("VOICECHAT_LIVE_PLAYOUT_TRACE")
            if configured_playout_trace:
                browser_trace_path = Path(configured_playout_trace).expanduser().resolve()
                browser_trace_sink = await _install_browser_playout_capture(
                    context,
                    page,
                    browser_trace_path,
                    peer_connection_slot="__voicechatLivePC",
                )
            connect = page.get_by_role("button", name="Connect", exact=True)
            disconnect = page.get_by_role("button", name="Disconnect", exact=True)
            for _ in range(6):
                await connect.click()
                try:
                    await disconnect.wait_for(timeout=4_000)
                    break
                except PlaywrightTimeoutError:
                    await page.wait_for_timeout(300)
            else:
                pytest.fail(f"live Playground did not connect: {browser_log[-50:]}")

            # The Playground keeps inactive Radix tab panels mounted, so the
            # same input can exist twice in the DOM. Target the active panel;
            # Playwright strict mode will still fail if more than one is visible.
            text_box = page.locator('[placeholder="Type message..."]:visible')
            await text_box.wait_for(state="visible", timeout=10_000)
            if not await text_box.is_enabled():
                pytest.fail(f"Playground text input never became ready: {browser_log[-50:]}")

            async def rtc_stats():
                return await page.evaluate(
                    """async () => {
                      const pc = window.__voicechatLivePC; let inbound=0, outbound=0;
                      if (pc) for (const report of (await pc.getStats()).values()) {
                        if (report.type === 'inbound-rtp' && report.kind === 'audio')
                          inbound += report.bytesReceived || 0;
                        if (report.type === 'outbound-rtp' && report.kind === 'audio')
                          outbound += report.bytesSent || 0;
                      }
                      const audio = document.querySelector('audio');
                      return {inbound, outbound,
                        tracks: audio?.srcObject?.getAudioTracks()?.length || 0,
                        currentTime: audio?.currentTime || 0};
                    }"""
                )

            async def rtvi_transcriptions(event_type: str) -> list[str]:
                return await page.evaluate(
                    """(eventType) => (window.__voicechatRtviMessages || [])
                      .filter((message) => message?.type === eventType)
                      .map((message) => String(message?.data?.text || ""))""",
                    event_type,
                )

            async def rtvi_event_count(event_type: str) -> int:
                return await page.evaluate(
                    """(eventType) => (window.__voicechatRtviMessages || [])
                      .filter((message) => message?.type === eventType).length""",
                    event_type,
                )

            async def wait_for_rtvi_event(
                event_type: str, *, after: int, timeout: float = 90
            ) -> int:
                deadline = asyncio.get_running_loop().time() + timeout
                count = 0
                while asyncio.get_running_loop().time() < deadline:
                    count = await rtvi_event_count(event_type)
                    if count > after:
                        return count
                    await asyncio.sleep(0.25)
                pytest.fail(
                    f"no new {event_type} after index {after}; count={count} "
                    f"browser_log={browser_log[-50:]}"
                )

            async def wait_for_rtvi_text(
                event_type: str,
                pattern: re.Pattern[str],
                *,
                after: int = 0,
                timeout: float = 90,
            ) -> list[str]:
                deadline = asyncio.get_running_loop().time() + timeout
                texts: list[str] = []
                while asyncio.get_running_loop().time() < deadline:
                    texts = await rtvi_transcriptions(event_type)
                    if any(pattern.search(text) for text in texts[after:]):
                        return texts
                    await asyncio.sleep(0.25)
                pytest.fail(
                    f"no {event_type} matched {pattern.pattern!r} after index {after}; "
                    f"texts={texts!r} browser_log={browser_log[-50:]}"
                )

            if tool_first:
                # Start a deliberately long function epoch while Chrome is
                # already delivering the fake microphone's leading silence.
                # The later real voice turn proves that held idle PCM did not
                # become seconds of ordered input debt.
                tool_started_ms = await page.evaluate("performance.now()")
                bot_before_tool = len(await rtvi_transcriptions("bot-transcription"))
                stopped_before_tool = await rtvi_event_count("bot-stopped-speaking")
                await text_box.fill("Use the available clock tool and tell me the current time.")
                await text_box.press("Enter")
                tool_texts = await wait_for_rtvi_text(
                    "bot-transcription",
                    re.compile(r"\S"),
                    after=bot_before_tool,
                    timeout=120,
                )
                stopped_after_tool = await wait_for_rtvi_event(
                    "bot-stopped-speaking", after=stopped_before_tool, timeout=120
                )
                print(
                    "VOICECHAT_TOOL_FIRST "
                    + json.dumps(
                        {
                            "elapsedMs": await page.evaluate("performance.now()") - tool_started_ms,
                            "transcripts": tool_texts[bot_before_tool:],
                        },
                        sort_keys=True,
                    )
                )

            baseline = await rtc_stats()

            # The generated microphone carries a real spoken "remember
            # sapphire" turn after its leading silence. Complete that voice
            # turn before sending typed input: Chrome advances fake-capture
            # audio from browser startup, so attempting both concurrently
            # would correctly make the server's half-duplex gate discard the
            # microphone turn.
            sapphire_pattern = re.compile(r"\bsapphire\b", re.I)
            stopped_before_voice = (
                stopped_after_tool if tool_first else await rtvi_event_count("bot-stopped-speaking")
            )
            await wait_for_rtvi_text("user-transcription", sapphire_pattern, timeout=120)
            # The acknowledgement need not repeat the secret. Memory is tested
            # only by the later recall prompt, which deliberately omits it.
            await wait_for_rtvi_event(
                "bot-stopped-speaking", after=stopped_before_voice, timeout=120
            )
            voice_to_voice = await page.evaluate(
                """(stopProbe) => {
                  const state = window.__voicechatV2V;
                  state.stopped = stopProbe;
                  const {stopped, ...report} = state;
                  return report;
                }""",
                not measure_second_voice_turn,
            )
            assert voice_to_voice["outboundSpeechFirstMs"] is not None
            assert voice_to_voice["outboundSpeechLastMs"] is not None
            assert voice_to_voice["inboundSpeechFirstMs"] is not None
            assert voice_to_voice["latencyMs"] >= 0
            print("VOICE_TO_VOICE_LATENCY " + json.dumps(voice_to_voice, sort_keys=True))
            rtvi_timing = await page.evaluate(
                """() => (window.__voicechatRtviTimedMessages || [])
                  .filter(({message}) =>
                    String(message?.type || '').includes('speaking') ||
                    String(message?.type || '').includes('metrics'))"""
            )
            print("VOICE_TO_VOICE_RTVI_TIMING " + json.dumps(rtvi_timing, sort_keys=True))

            if measure_second_voice_turn:
                user_before_second = len(await rtvi_transcriptions("user-transcription"))
                stopped_before_second = await rtvi_event_count("bot-stopped-speaking")
                await page.evaluate("() => window.__voicechatResetV2V()")
                await wait_for_rtvi_text(
                    "user-transcription",
                    sapphire_pattern,
                    after=user_before_second,
                    timeout=120,
                )
                await wait_for_rtvi_event(
                    "bot-stopped-speaking", after=stopped_before_second, timeout=120
                )
                second_voice_to_voice = await page.evaluate(
                    """() => {
                      const state = window.__voicechatV2V;
                      state.stopped = true;
                      const {stopped, ...report} = state;
                      return report;
                    }"""
                )
                assert second_voice_to_voice["outboundSpeechFirstMs"] is not None
                assert second_voice_to_voice["outboundSpeechLastMs"] is not None
                assert second_voice_to_voice["inboundSpeechFirstMs"] is not None
                assert second_voice_to_voice["latencyMs"] >= 0
                print(
                    "VOICE_TO_VOICE_LATENCY_WARM "
                    + json.dumps(second_voice_to_voice, sort_keys=True)
                )
                second_rtvi_timing = await page.evaluate(
                    """() => (window.__voicechatRtviTimedMessages || [])
                      .filter(({message}) =>
                        String(message?.type || '').includes('speaking') ||
                        String(message?.type || '').includes('metrics'))"""
                )
                print(
                    "VOICE_TO_VOICE_RTVI_TIMING_WARM "
                    + json.dumps(second_rtvi_timing, sort_keys=True)
                )

            bot_before_math = len(await rtvi_transcriptions("bot-transcription"))
            stopped_before_math = await rtvi_event_count("bot-stopped-speaking")
            await text_box.fill(prompt)
            await text_box.press("Enter")
            await wait_for_rtvi_text(
                "bot-transcription",
                re.compile(r"\bfive\b", re.I),
                after=bot_before_math,
            )
            await wait_for_rtvi_event("bot-stopped-speaking", after=stopped_before_math)

            # This prompt deliberately omits the codeword. A new final
            # bot-transcription event proves that typed input can recall
            # context supplied only through the real microphone/WebRTC path.
            bot_before_recall = len(await rtvi_transcriptions("bot-transcription"))
            await text_box.fill("What codeword did I ask you to remember?")
            await text_box.press("Enter")
            await wait_for_rtvi_text("bot-transcription", sapphire_pattern, after=bot_before_recall)

            stats = {"inbound": 0, "outbound": 0, "tracks": 0, "currentTime": 0}
            deadline = asyncio.get_running_loop().time() + 90
            while asyncio.get_running_loop().time() < deadline:
                stats = await rtc_stats()
                if stats["inbound"] - baseline["inbound"] > 2_000:
                    break
                await asyncio.sleep(0.5)
            await page.screenshot(path=str(tmp_path / "live-playground.png"), full_page=True)
            assert stats["outbound"] > 0
            assert stats["inbound"] - baseline["inbound"] > 2_000
            assert stats["tracks"] >= 1
            assert stats["currentTime"] > 0
            await disconnect.click()
            await connect.wait_for(timeout=8_000)
        finally:
            capture_overhead = await _shutdown_browser_playout_capture(page, browser_trace_sink)
            capture_overhead["test_elapsed_ms"] = (time.perf_counter() - capture_started) * 1000.0
            print(
                "BROWSER_PLAYOUT_CAPTURE_OVERHEAD " + json.dumps(capture_overhead, sort_keys=True)
            )
            await context.close()
            await browser.close()


@pytest.mark.browser_e2e
@pytest.mark.asyncio
async def test_step4c_live_fixture_plan(tmp_path):
    """Run one exact L1 incidence block under the isolated Step 4c schema."""

    raw_plan = os.environ.get("VOICECHAT_STEP4C_FIXTURE_PLAN")
    if not raw_plan:
        pytest.skip("set VOICECHAT_STEP4C_FIXTURE_PLAN for the Step 4c live fixture")
    plan = parse_step4c_fixture_plan(raw_plan)
    url = os.environ.get("VOICECHAT_LIVE_PLAYGROUND_URL")
    if not url:
        pytest.fail("VOICECHAT_LIVE_PLAYGROUND_URL is required for Step 4c")
    if CHROMIUM and not Path(CHROMIUM).exists():
        pytest.fail(f"configured Chromium is not installed at {CHROMIUM}")
    configured_trace = os.environ.get("NEMOTRON_VOICECHAT_PLAYOUT_TRACE")
    configured_output = os.environ.get("VOICECHAT_STEP4C_OUTPUT_DIR")
    if not configured_trace or not configured_output:
        pytest.fail("Step 4c requires explicit playout trace and output directory paths")
    output_dir = Path(configured_output).expanduser()
    microphone_wav = Path(
        os.environ.get("VOICECHAT_LIVE_MIC_WAV", str(tmp_path / "step4c-silent-mic.wav"))
    ).expanduser()
    if not microphone_wav.exists():
        _write_silent_microphone(
            microphone_wav, duration=plan.duration_seconds + plan.drain_seconds + 30
        )
    capture_artifact = output_dir / "browser-playout.jsonl"
    unused_capture_off = output_dir / "browser-playout-unused-capture-off.jsonl"
    summary_path = output_dir / "step4c-browser-summary.json"
    try:
        resolved = resolve_step2_paths(
            microphone=microphone_wav,
            capture_on=capture_artifact,
            capture_off=unused_capture_off,
            summary=summary_path,
            pipecat_trace=Path(configured_trace).expanduser(),
            output_dir=output_dir,
        )
    except (ValueError, FileExistsError) as exc:
        pytest.fail(f"unsafe Step 4c path configuration: {exc}")
    capture_artifact = Path(resolved["browser_capture_on"])
    summary_path = Path(resolved["summary"])
    pipecat_trace = connection_playout_trace_path(Path(resolved["pipecat_playout_trace"]), 1)
    health_url = os.environ.get("VOICECHAT_STEP4C_HEALTH_URL", "http://127.0.0.1:8786/health")
    try:
        with urllib.request.urlopen(health_url, timeout=10) as response:  # noqa: S310
            health = json.load(response)
    except Exception as exc:  # noqa: BLE001 - retained pre-connection failure
        resolved.close()
        pytest.fail(f"Step 4c immediate health check failed: {exc!r}")
    if not isinstance(health, dict) or health.get("active_client") is not False:
        resolved.close()
        pytest.fail("Step 4c immediate health check did not prove active_client:false")
    health_path = output_dir / "health-immediately-before.json"
    health_descriptor = resolved.open_child(
        health_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
    with os.fdopen(health_descriptor, "w", encoding="utf-8") as stream:
        json.dump(health, stream, indent=2, sort_keys=True)
        stream.write("\n")

    fixture = plan.fixtures[0]
    result = blank_fixture_result(fixture)
    summary: dict[str, Any] = {
        "schema": STEP4C_SUMMARY_SCHEMA,
        "status": "running",
        "started_wall_time_s": time.time(),
        "completed_wall_time_s": None,
        "plan": plan_document(plan),
        "health_immediately_before": health,
        "session": {
            "status": "running",
            "session_id": None,
            "browser_artifact_validation": None,
            "observed_playout_trace": None,
            "capture_shutdown": None,
            "teardown_errors": [],
            "connected_window_seconds": None,
        },
        "fixture": result,
        "error": None,
    }

    def persist() -> None:
        _write_step4c_summary(summary_path, summary, resolved)

    persist()
    manager = async_playwright()
    playwright = None
    state: dict[str, Any] = {"browser": None, "context": None, "page": None, "sink": None}
    browser_log: list[str] = []
    failure: str | None = None
    connected_started: float | None = None
    try:
        playwright = await _start_playwright(manager)
        text_box = await _setup_step2_session(
            state=state,
            playwright=playwright,
            url=url,
            microphone_wav=Path(resolved["microphone"]),
            artifact=capture_artifact,
            capture_mode="on",
            browser_log=browser_log,
            reservation=resolved,
        )
        connected_started = time.monotonic()
        await asyncio.sleep(plan.warmup_seconds)
        attribution_cursor: dict[str, int | None] = {
            "rtvi_index": 0,
            "playout_index": len(read_playout_trace(pipecat_trace, reservation=resolved)),
        }
        fixture_ok = await _run_step2_fixture(
            state["page"],
            text_box,
            state["sink"],
            fixture,
            result,
            persist,
            pipecat_trace,
            trace_reader=lambda path: read_playout_trace(path, reservation=resolved),
            attribution_cursor=attribution_cursor,
        )
        if not fixture_ok:
            failure = f"Step 4c L1 failed: {result['error']}"
        remaining = connected_started + plan.duration_seconds - time.monotonic()
        if remaining > 0:
            await asyncio.sleep(remaining)
        await asyncio.sleep(plan.drain_seconds)
        summary["session"]["connected_window_seconds"] = time.monotonic() - connected_started
    except Exception as exc:  # noqa: BLE001 - retain live partial evidence
        failure = failure or repr(exc)
    finally:
        shutdown, artifact_validation, teardown_errors = await _teardown_step2_session(
            page=state.get("page"),
            sink=state.get("sink"),
            context=state.get("context"),
            browser=state.get("browser"),
            artifact=capture_artifact,
            capture_enabled=True,
            expected_fixtures=plan.fixtures,
            reservation=resolved,
        )
        summary["session"]["capture_shutdown"] = shutdown
        summary["session"]["browser_artifact_validation"] = artifact_validation
        summary["session"]["teardown_errors"] = teardown_errors
        if teardown_errors:
            failure = failure or teardown_errors[0]
        if playwright is not None:
            terminal_errors: list[str] = []
            stop_error = await _stop_playwright_recorded(playwright, terminal_errors)
            failure = failure or stop_error
        try:
            observed = await wait_for_playout_trace_publication(pipecat_trace, reservation=resolved)
            summary["session"]["observed_playout_trace"] = observed
            records = read_playout_trace(pipecat_trace, reservation=resolved)
            session_ids = {
                record.get("session_id")
                for record in records
                if isinstance(record.get("session_id"), str)
            }
            if len(session_ids) != 1:
                failure = failure or f"ambiguous Step 4c session IDs: {sorted(session_ids)}"
            else:
                summary["session"]["session_id"] = session_ids.pop()
        except Exception as exc:  # noqa: BLE001 - terminal evidence is mandatory
            failure = failure or f"playout publication failed: {exc!r}"
        summary["completed_wall_time_s"] = time.time()
        summary["status"] = "failed" if failure else "completed"
        summary["session"]["status"] = "failed" if failure else "completed"
        summary["error"] = failure
        persist()
        resolved.close()
    if failure:
        pytest.fail(f"Step 4c browser fixture failed closed: {failure}; summary={summary_path}")


@pytest.mark.browser_e2e
@pytest.mark.asyncio
async def test_step2_live_fixture_plan(tmp_path):
    """Run the preregistered plan in one session per capture mode."""

    raw_plan = os.environ.get("VOICECHAT_LIVE_FIXTURE_PLAN")
    if not raw_plan:
        pytest.skip("set VOICECHAT_LIVE_FIXTURE_PLAN for the Step 2 live fixture driver")
    plan = parse_fixture_plan(raw_plan)
    try:
        quiescence_window_ms = parse_rtvi_quiescence_window_ms(
            os.environ.get("VOICECHAT_LIVE_RTVI_QUIET_WINDOW_MS")
        )
    except ValueError as exc:
        pytest.fail(f"invalid VOICECHAT_LIVE_RTVI_QUIET_WINDOW_MS: {exc}")
    url = os.environ.get("VOICECHAT_LIVE_PLAYGROUND_URL")
    if not url:
        pytest.fail("VOICECHAT_LIVE_PLAYGROUND_URL is required with the Step 2 fixture plan")
    if CHROMIUM and not Path(CHROMIUM).exists():
        pytest.fail(f"configured Chromium is not installed at {CHROMIUM}")

    configured_microphone = os.environ.get("VOICECHAT_LIVE_MIC_WAV")
    if configured_microphone:
        microphone_wav = Path(configured_microphone).expanduser().resolve()
        if not microphone_wav.is_file():
            pytest.fail(f"VOICECHAT_LIVE_MIC_WAV does not exist: {microphone_wav}")
    else:
        microphone_wav = tmp_path / "step2-silent-microphone.wav"
        _write_silent_microphone(microphone_wav)

    configured_trace = os.environ.get("VOICECHAT_LIVE_PLAYOUT_TRACE")
    configured_pipecat_trace = os.environ.get("NEMOTRON_VOICECHAT_PLAYOUT_TRACE")
    if not configured_pipecat_trace:
        pytest.fail("NEMOTRON_VOICECHAT_PLAYOUT_TRACE is required for response attribution")
    configured_output = os.environ.get("VOICECHAT_LIVE_FIXTURE_OUTPUT_DIR")
    if configured_output:
        output_dir = Path(configured_output).expanduser()
    elif configured_trace:
        output_dir = Path(configured_trace).expanduser().parent
    else:
        output_dir = tmp_path / "step2-live-session"
    if configured_trace:
        capture_on_artifact = Path(configured_trace).expanduser()
    else:
        capture_on_artifact = output_dir / "browser-playout-capture-on.jsonl"
    capture_off_artifact = capture_on_artifact.with_name(
        capture_on_artifact.stem + "-capture-off" + capture_on_artifact.suffix
    )
    artifacts = {"on": capture_on_artifact, "off": capture_off_artifact}
    configured_summary = os.environ.get("VOICECHAT_LIVE_FIXTURE_SUMMARY")
    summary_path = (
        Path(configured_summary).expanduser()
        if configured_summary
        else output_dir / "fixture-summary.json"
    )
    try:
        resolved_paths = resolve_step2_paths(
            microphone=microphone_wav,
            capture_on=capture_on_artifact,
            capture_off=capture_off_artifact,
            summary=summary_path,
            pipecat_trace=Path(configured_pipecat_trace),
            output_dir=output_dir,
        )
    except (ValueError, FileExistsError) as exc:
        pytest.fail(f"unsafe Step 2 path configuration: {exc}")
    microphone_wav = Path(resolved_paths["microphone"])
    capture_on_artifact = Path(resolved_paths["browser_capture_on"])
    capture_off_artifact = Path(resolved_paths["browser_capture_off"])
    pipecat_trace_path = Path(resolved_paths["pipecat_playout_trace"])
    summary_path = Path(resolved_paths["summary"])
    artifacts = {"on": capture_on_artifact, "off": capture_off_artifact}

    summary: dict[str, Any] = {
        "schema": SUMMARY_SCHEMA,
        "status": "running",
        "started_wall_time_s": time.time(),
        "completed_wall_time_s": None,
        "plan": plan_document(plan),
        "resolved_paths": resolved_paths,
        "terminalization_errors": [],
        "sessions": [
            {
                "capture": mode,
                "artifact": str(artifacts[mode]),
                "status": "not_run",
                "capture_shutdown": None,
                "error": None,
                "teardown_errors": [],
                "browser_artifact_validation": None,
                "pipecat_artifact": resolved_paths[f"pipecat_playout_capture_{mode}"],
                "pipecat_publication_wait_s": None,
                "observed_playout_trace": None,
            }
            for mode, _fixtures in plan.sessions()
        ],
        "fixtures": [blank_fixture_result(fixture) for fixture in plan.fixtures],
    }

    def persist() -> None:
        _write_fixture_summary(summary_path, summary, reservation=resolved_paths)

    persist()
    failure: str | None = None
    playwright_manager = async_playwright()
    playwright = None
    try:
        playwright = await _start_playwright(playwright_manager)
        for session_index, (capture_mode, fixtures) in enumerate(plan.sessions()):
            session = summary["sessions"][session_index]
            session_pipecat_trace = connection_playout_trace_path(
                pipecat_trace_path, session_index + 1
            )
            session["status"] = "running"
            persist()
            browser = None
            context = None
            page = None
            sink = None
            current_fixture: FixtureSpec | None = None
            capture_started = time.perf_counter()
            browser_log: list[str] = []
            setup_state = {
                "browser": None,
                "context": None,
                "page": None,
                "sink": None,
            }
            try:
                text_box = await _setup_step2_session(
                    state=setup_state,
                    playwright=playwright,
                    url=url,
                    microphone_wav=microphone_wav,
                    artifact=artifacts[capture_mode],
                    capture_mode=capture_mode,
                    browser_log=browser_log,
                    reservation=resolved_paths,
                )
                browser = setup_state["browser"]
                context = setup_state["context"]
                page = setup_state["page"]
                sink = setup_state["sink"]
                attribution_cursor: dict[str, int | None] = {
                    "rtvi_index": 0,
                    "playout_index": len(
                        read_playout_trace(session_pipecat_trace, reservation=resolved_paths)
                    ),
                }

                for current_fixture in fixtures:
                    result = summary["fixtures"][current_fixture.plan_index]
                    if not await _run_step2_fixture(
                        page,
                        text_box,
                        sink,
                        current_fixture,
                        result,
                        persist,
                        session_pipecat_trace,
                        trace_reader=lambda path: read_playout_trace(
                            path, reservation=resolved_paths
                        ),
                        attribution_cursor=attribution_cursor,
                        quiescence_window_ms=quiescence_window_ms,
                    ):
                        failure = f"{current_fixture.fixture_id}: {result['error']}"
                        break
                if failure is None:
                    session["status"] = "completed"
                else:
                    session["status"] = "failed"
                    session["error"] = failure
            except Exception as exc:  # noqa: BLE001 - retain live partial evidence
                failure = failure or f"capture-{capture_mode} session failed: {exc!r}"
                session["error"] = repr(exc)
                if current_fixture is not None:
                    result = summary["fixtures"][current_fixture.plan_index]
                    if result["completion_status"] == "running":
                        result["completion_status"] = "error"
                        result["completion_wall_time_s"] = time.time()
                        result["error"] = repr(exc)
                        marker_available = False
                        try:
                            marker_available = (
                                page is not None and sink is not None and not page.is_closed()
                            )
                        except Exception as availability_exc:  # noqa: BLE001
                            result["marker_emission"]["error"] = {
                                "status": "failed",
                                "error": (
                                    f"error marker availability check failed: {availability_exc!r}"
                                ),
                            }
                        if marker_available:
                            try:
                                await _emit_fixture_boundary_recorded(
                                    page,
                                    sink,
                                    current_fixture,
                                    result,
                                    "error",
                                    persist,
                                    detail=repr(exc),
                                )
                            except Exception as marker_exc:  # noqa: BLE001
                                result["error"] += f"; error marker failed: {marker_exc!r}"
                        elif result["marker_emission"]["error"]["status"] == "not_attempted":
                            result["marker_emission"]["error"] = {
                                "status": "failed",
                                "error": "error marker unavailable: page or sink closed/missing",
                            }
                session["status"] = "failed"
                persist()
            finally:
                browser = setup_state["browser"]
                context = setup_state["context"]
                page = setup_state["page"]
                sink = setup_state["sink"]
                (
                    capture_shutdown,
                    artifact_validation,
                    teardown_errors,
                ) = await _teardown_step2_session(
                    page=page,
                    sink=sink,
                    context=context,
                    browser=browser,
                    artifact=artifacts[capture_mode],
                    capture_enabled=capture_mode == "on",
                    reservation=resolved_paths,
                )
                sink = None
                if capture_shutdown is not None:
                    capture_shutdown["test_elapsed_ms"] = (
                        time.perf_counter() - capture_started
                    ) * 1000.0
                    session["capture_shutdown"] = capture_shutdown
                    print(
                        "BROWSER_PLAYOUT_CAPTURE_OVERHEAD "
                        + json.dumps(
                            {"capture_mode": capture_mode, **capture_shutdown},
                            sort_keys=True,
                        )
                    )
                session["browser_artifact_validation"] = artifact_validation
                session["teardown_errors"].extend(teardown_errors)
                publication_wait_started = time.monotonic()
                try:
                    session["observed_playout_trace"] = await wait_for_playout_trace_publication(
                        session_pipecat_trace, reservation=resolved_paths
                    )
                except Exception as exc:  # noqa: BLE001 - terminal evidence is mandatory
                    session["teardown_errors"].append(
                        f"Pipecat artifact publication/validation failed: {exc!r}"
                    )
                finally:
                    session["pipecat_publication_wait_s"] = (
                        time.monotonic() - publication_wait_started
                    )
                if teardown_errors:
                    failure = failure or f"capture-{capture_mode} {teardown_errors[0]}"
                    session["status"] = "failed"
                if session["teardown_errors"]:
                    failure = failure or (f"capture-{capture_mode} {session['teardown_errors'][0]}")
                    session["status"] = "failed"
                persist()
            if failure is not None:
                break
    except Exception as exc:  # noqa: BLE001 - outer setup/runner boundary
        failure = failure or f"Step 2 campaign failed: {exc!r}"
        summary["terminalization_errors"].append(repr(exc))
    finally:
        if playwright is not None:
            error = await _stop_playwright_recorded(playwright, summary["terminalization_errors"])
            if error is not None:
                failure = failure or error
        try:
            _terminalize_step2_summary(summary, failure, persist)
        finally:
            resolved_paths.close()
    if failure is not None:
        pytest.fail(f"Step 2 fixture driver failed closed: {failure}; summary={summary_path}")
    assert all(result["completion_status"] == "completed" for result in summary["fixtures"])
