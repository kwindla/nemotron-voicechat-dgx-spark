"""Full browser -> SmallWebRTC -> bot -> Voicechat WebSocket qualification."""

from __future__ import annotations

import asyncio
import base64
import json
import math
import os
import re
import socket
import sys
import urllib.error
import urllib.request
import wave
from pathlib import Path

import numpy as np
import pytest
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from playwright.async_api import async_playwright
from websockets.asyncio.server import serve

CHROMIUM = os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE")


def _rtc_probe_script(peer_connection_slot: str) -> str:
    """Observe RTC stats and browser-received RTVI messages without altering them."""

    return f"""(() => {{
      const NativePeerConnection = window.RTCPeerConnection;
      const observedChannels = new WeakSet();
      window.__voicechatRtviMessages = [];
      const observeChannel = (channel) => {{
        if (observedChannels.has(channel)) return;
        observedChannels.add(channel);
        channel.addEventListener("message", (event) => {{
          if (typeof event.data !== "string") return;
          try {{ window.__voicechatRtviMessages.push(JSON.parse(event.data)); }}
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
                    "capabilities": {},
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
                await context.add_init_script(
                    _rtc_probe_script("__voicechatTestPeerConnection")
                )
                page = await context.new_page()
                browser_log: list[str] = []
                page.on("console", lambda message: browser_log.append(message.text))
                page.on("pageerror", lambda error: browser_log.append(f"pageerror: {error}"))
                await page.goto("/client/")
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
                observed_rtvi = await page.evaluate(
                    "() => window.__voicechatRtviMessages || []"
                )

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
            assert any(
                message.get("type") == "bot-transcription"
                and "deterministic test passed"
                in str(message.get("data", {}).get("text", "")).lower()
                for message in observed_rtvi
            )
        finally:
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
        microphone_wav = Path(configured_microphone).expanduser().resolve()
        if not microphone_wav.is_file():
            pytest.fail(f"VOICECHAT_LIVE_MIC_WAV does not exist: {microphone_wav}")
    else:
        microphone_wav = tmp_path / "silent-microphone.wav"
        _write_silent_microphone(microphone_wav)
    prompt = "What is two plus three? Answer in one short sentence."

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
        page.on("console", lambda message: browser_log.append(message.text))
        page.on("pageerror", lambda error: browser_log.append(f"pageerror: {error}"))
        try:
            await page.goto(url.rstrip("/") + "/", wait_until="load", timeout=30_000)
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

            baseline = await rtc_stats()

            # The generated microphone carries a real spoken "remember
            # sapphire" turn after its leading silence. Complete that voice
            # turn before sending typed input: Chrome advances fake-capture
            # audio from browser startup, so attempting both concurrently
            # would correctly make the server's half-duplex gate discard the
            # microphone turn.
            sapphire_pattern = re.compile(r"\bsapphire\b", re.I)
            await wait_for_rtvi_text(
                "user-transcription", sapphire_pattern, timeout=120
            )
            await wait_for_rtvi_text(
                "bot-transcription", sapphire_pattern, timeout=120
            )

            bot_before_math = len(await rtvi_transcriptions("bot-transcription"))
            await text_box.fill(prompt)
            await text_box.press("Enter")
            await wait_for_rtvi_text(
                "bot-transcription",
                re.compile(r"\bfive\b", re.I),
                after=bot_before_math,
            )

            # This prompt deliberately omits the codeword. A new final
            # bot-transcription event proves that typed input can recall
            # context supplied only through the real microphone/WebRTC path.
            bot_before_recall = len(await rtvi_transcriptions("bot-transcription"))
            await text_box.fill("What codeword did I ask you to remember?")
            await text_box.press("Enter")
            await wait_for_rtvi_text(
                "bot-transcription", sapphire_pattern, after=bot_before_recall
            )

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
            await context.close()
            await browser.close()
