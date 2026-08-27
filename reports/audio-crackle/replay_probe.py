"""Drive one browser session with a recorded microphone, then read the counter.

Deliberately free of page instrumentation: an earlier probe wrapped
RTCPeerConnection and subclassed AudioContext, and that perturbation broke the
very session it was meant to observe. The only evidence taken here is the
bot's own summary log after disconnect.
"""

import asyncio
import sys

from playwright.async_api import async_playwright

CHROMIUM = "/home/khkramer/.cache/ms-playwright/chromium-1234/chrome-linux/chrome"
URL = "http://127.0.0.1:7860/"
WAV = sys.argv[1]
SECONDS = float(sys.argv[2]) if len(sys.argv) > 2 else 95.0


async def main():
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            executable_path=CHROMIUM,
            headless=True,
            args=[
                "--no-sandbox",
                "--use-fake-device-for-media-stream",
                "--use-fake-ui-for-media-stream",
                f"--use-file-for-fake-audio-capture={WAV}%noloop",
                "--autoplay-policy=no-user-gesture-required",
            ],
        )
        context = await browser.new_context(permissions=["microphone", "camera"])
        page = await context.new_page()
        errors = []
        page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
        await page.goto(URL, wait_until="load", timeout=30_000)
        connect = page.get_by_role("button", name="Connect", exact=True)
        disconnect = page.get_by_role("button", name="Disconnect", exact=True)
        for _ in range(6):
            await connect.click()
            try:
                await disconnect.wait_for(timeout=5_000)
                break
            except Exception:
                await page.wait_for_timeout(300)
        else:
            raise SystemExit(f"never connected: {errors[-10:]}")
        print("connected; replaying", SECONDS, "s", flush=True)
        await page.wait_for_timeout(int(SECONDS * 1000))
        # The fixture is finite and plays with %noloop, so if the replay runs
        # past the end of the WAV the microphone goes silent and the pipeline
        # idle timeout ends the session on its own. That is a normal end to the
        # run, not a failure: the button is simply gone by the time we click.
        try:
            await disconnect.click(timeout=5_000)
            print("disconnected", flush=True)
        except Exception:
            print("session already ended (idle timeout past end of fixture)", flush=True)
        await page.wait_for_timeout(1500)
        await browser.close()


asyncio.run(main())
