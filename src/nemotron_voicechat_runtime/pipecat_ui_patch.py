"""Pinned repair of the vendored Pipecat prebuilt browser UI.

``pipecat-ai-prebuilt==1.0.5`` ships ``@pipecat-ai/small-webrtc-transport``
1.10.6, whose ``DailyMediaManager.connect()`` never settles on a second call.

The manager is constructed with recording disabled, so
``_mediaStreamRecorder`` is undefined.  Its ``connect()`` takes an
uninitialized-path branch that stores the promise resolver in
``_connectResolve``, awaits ``initialize()``, and then falls off the end of the
async closure without invoking it.  The sole call to ``_connectResolve()``
lives in ``handleTrackStarted()`` behind a ``_mediaStreamRecorder.begin(track)``
guard, which is unreachable when recording is disabled, so the promise stays
pending forever.

The first browser connection is unaffected because the UI initializes devices
on mount, leaving ``_initialized`` true and taking the fast path.  Disconnect
sets ``_initialized`` false while the client's own device state stays
``granted``, so ``needsInit()`` is false and ``initDevices()`` is not called
again.  The next same-page Connect therefore takes the broken branch: media
acquisition succeeds, ``startNewPeerConnection()`` is never reached, no SDP
offer is ever sent, and the UI sits on "Connecting..." until the page reloads.

No server-side change can repair this: the browser never asks the server to
create a connection.  This module applies the upstream correction -- settle the
promise once ``initialize()`` returns when no recorder exists -- directly to the
installed asset.

The repair is pinned by SHA-256 and fails closed.  If the vendored bundle is
neither the known-unpatched nor the known-patched revision, the pin is stale
and must be re-verified against the new upstream bytes rather than applied
blindly to unknown code.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

#: Asset carrying ``DailyMediaManager`` in pipecat-ai-prebuilt 1.0.5.  Vite
#: content-hashes the filename, so it is stable for a given release.
ASSET_GLOB = "pipecat_ai_prebuilt/client/dist/assets/index.module-*.js"

UNPATCHED_SHA256 = "4fc40154f5262add6c4926b49daa52f576eb0fac9fb6a9ee561878f04533cae9"
PATCHED_SHA256 = "7438a4c0e7f6a98c40d27f950f1abe2db480223d67784a66c28bd7e1ae70239e"

#: The uninitialized-path closure that returns without settling its promise.
_ANCHOR = "case 0:return this._connectResolve=t,[4,this.initialize()];case 1:return e.sent(),[2]}"

#: Same closure, settling the stored resolver exactly once.
_REPAIRED = (
    "case 0:return this._connectResolve=t,[4,this.initialize()];"
    "case 1:return e.sent(),"
    "this._connectResolve&&(this._connectResolve(),this._connectResolve=null),[2]}"
)


class PipecatUIPatchError(RuntimeError):
    """The vendored UI does not match the revision this repair is pinned to."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def candidates(venv: Path) -> list[Path]:
    """Return every prebuilt UI module chunk installed in ``venv``.

    The release ships several ``index.module-*.js`` chunks, so the chunk
    carrying ``DailyMediaManager`` is identified by content hash rather than by
    filename.
    """
    return sorted(venv.glob(f"lib/python3.*/site-packages/{ASSET_GLOB}"))


def find_asset(venv: Path) -> Path | None:
    """Return the unpatched chunk this repair targets, if it is present."""
    for asset in candidates(venv):
        if _sha256(asset) == UNPATCHED_SHA256:
            return asset
    return None


def apply(venv: Path) -> str:
    """Ensure the prebuilt UI in ``venv`` can reconnect without a page reload.

    Returns a short status string.  Raises :class:`PipecatUIPatchError` when the
    installed bundle is an unrecognised revision, so an upstream change surfaces
    as a hard failure instead of a silently unpatched browser regression.
    """
    installed = candidates(venv)
    if not installed:
        return "pipecat prebuilt UI not installed; nothing to repair"

    digests = {asset: _sha256(asset) for asset in installed}
    if PATCHED_SHA256 in digests.values():
        return "pipecat prebuilt UI already repaired"

    asset = find_asset(venv)
    if asset is None:
        raise PipecatUIPatchError(
            "no installed Pipecat prebuilt UI chunk matches the revision this "
            f"repair is pinned to (unpatched {UNPATCHED_SHA256}); found "
            + ", ".join(f"{path.name}={digest}" for path, digest in digests.items())
            + ". Re-verify the DailyMediaManager.connect() defect against the new "
            "bundle and update src/nemotron_voicechat_runtime/pipecat_ui_patch.py "
            "before starting the bot."
        )

    source = asset.read_text(encoding="utf-8")
    if source.count(_ANCHOR) != 1:
        raise PipecatUIPatchError(
            f"{asset} matches the pinned SHA-256 but does not contain exactly one "
            "DailyMediaManager.connect() anchor; refusing to patch"
        )

    asset.write_text(source.replace(_ANCHOR, _REPAIRED), encoding="utf-8")

    written = _sha256(asset)
    if written != PATCHED_SHA256:
        raise PipecatUIPatchError(
            f"repaired {asset} to unexpected SHA-256 {written}, expected {PATCHED_SHA256}"
        )
    return "repaired pipecat prebuilt UI same-page reconnect"
