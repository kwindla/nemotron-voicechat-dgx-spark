# Same-page reconnect root-cause analysis — round 1

Date: 2026-08-18 UTC

## Verdict

The reconnect hang is client-side in the static UI shipped by
`pipecat-ai-prebuilt==1.0.5`. It is not caused by a retained server peer
connection, a reused `pc_id`, or a dead Pipecat worker.

The exact non-settling await is:

```text
SmallWebRTCTransport._connect()
  -> await this.mediaManager.connect()
```

`mediaManager.connect()` is `DailyMediaManager.connect()`. On the second
connection it creates and returns a Promise whose resolver is stored in
`_connectResolve`. Its inner `await this.initialize()` completes successfully,
including `Daily.startCamera()`/`getUserMedia`, but version 1.10.6 never invokes
the stored resolver when the manager was constructed without a recorder. The
Promise therefore remains pending forever. `SmallWebRTCTransport._connect()`
never reaches `await this.startNewPeerConnection()`, so no
`RTCPeerConnection`, SDP offer, or signaling request can exist.

No server-side change can repair this execution path. A fix is possible without
changing the frozen Python/Pipecat runtime, but it requires changing the
frontend that `/client/` serves: either serve a locally rebuilt/patched prebuilt
UI or change that UI to call `client.initDevices()` before every connect. If the
prebuilt frontend bytes are also part of the frozen contract, there is no fix
within the frozen environment; only reload remains as a workaround.

## Pinned component and source evidence

The installed Python artifact reports `pipecat-ai-prebuilt==1.0.5`. The v1.0.5
prebuilt lock resolves these exact browser packages:

| package | version |
|---|---:|
| `@pipecat-ai/client-js` | 1.13.0 |
| `@pipecat-ai/client-react` | 1.8.1 |
| `@pipecat-ai/small-webrtc-transport` | 1.10.6 |

The installed SmallWebRTC browser bundle is
`.venv/lib/python3.12/site-packages/pipecat_ai_prebuilt/client/dist/assets/index.module-BotpBYFo.js`.
Its minified `DailyMediaManager` and `SmallWebRTCTransport` implementations
match the 1.10.6 source at transport repository commit
`71165519a60ae6b7c2e234dbc519f758b44748fa`.

Two upstream records independently confirm the diagnosis:

- [Issue 154](https://github.com/pipecat-ai/pipecat-client-web-transports/issues/154)
  describes the same second-connect hang before an offer and identifies the
  unreachable `_connectResolve` path.
- [Commit dc935d3](https://github.com/pipecat-ai/pipecat-client-web-transports/commit/dc935d3ad1ce76c401b24fc35ae338e3175fef07),
  dated 2026-07-24 and titled “Fix issue where the
  DailyMediaManager.connect() would not resolve on second call,” adds the
  missing resolver call. That commit postdates the 1.10.6 package commit and is
  not in the installed bundle.

As of this analysis, the npm registry still reports 1.10.6 as the latest
released `@pipecat-ai/small-webrtc-transport`, so the upstream source fix is not
available through a released version bump.

## Exact lifecycle

### Construction

`SmallWebRTCTransport` constructs its default manager as:

```ts
new DailyMediaManager(false, false, onTrackStarted, onTrackStopped)
```

The second `false` disables recording, leaving
`DailyMediaManager._mediaStreamRecorder` undefined. This is intentional for the
SmallWebRTC transport because the WebRTC peer connection, not the WAV recorder,
consumes the media track.

### Why the first connection works

The prebuilt UI initializes devices on mount. That calls
`DailyMediaManager.initialize()` before the first Connect and leaves
`_initialized=true`.

The first `DailyMediaManager.connect()` therefore takes its initialized fast
path:

1. `_connected` changes from false to true.
2. The `if (!this._initialized)` branch is skipped.
3. `_startRecording()` is called when the microphone is enabled, but returns
   immediately because no recorder exists.
4. The async method returns normally, settling
   `await this.mediaManager.connect()`.
5. `SmallWebRTCTransport._connect()` enters
   `startNewPeerConnection()`, creates the peer connection, and signals the
   offer.

### What disconnect changes

`SmallWebRTCTransport.stop()` closes and nulls its peer connection, then awaits
`mediaManager.disconnect()`. `DailyMediaManager.disconnect()` clears its media
state and sets both `_initialized=false` and `_connected=false`.

The Pipecat client's higher-level device state remains `granted`, however.
Consequently the next `PipecatClient.connect()` sees `needsInit()==false` and
does not call `initDevices()` again. The media manager reaches its second
`connect()` uninitialized.

### Why the second connection hangs

The relevant 1.10.6 logic is equivalent to:

```ts
async connect(): Promise<void> {
  if (this._connected) return;
  this._connected = true;
  if (!this._initialized) {
    return new Promise((resolve) => {
      (async () => {
        this._connectResolve = resolve;
        await this.initialize();
      })();
    });
  }
  if (this._micEnabled) this._startRecording();
}
```

On reconnect:

1. `_connected` becomes true.
2. `_initialized` is false, so `connect()` returns the new Promise.
3. The Promise executor stores `resolve` in `_connectResolve` and starts its
   async closure.
4. `await this.initialize()` completes. In the observed browser run,
   `getUserMedia` returns a live audio track; the later initialization steps
   also return and `_initialized` becomes true.
5. The closure falls off its end without invoking `resolve` or wiring a
   rejection to the outer Promise.
6. The outer Promise remains pending. There is no thrown error, unhandled
   rejection, or page error.

The only 1.10.6 call to `_connectResolve()` is in
`DailyMediaManager.handleTrackStarted()`, inside the following chain of guards:

```text
local track -> audio track -> _mediaStreamRecorder exists
  -> recorder status is "ended" -> recorder.begin(track) succeeds
  -> _connected -> _connectResolve exists
```

The SmallWebRTC construction explicitly disables `_mediaStreamRecorder`, so
that resolver path is unreachable. This explains every instrumented symptom:
the UI state has already become `connecting`, media acquisition completes, but
`startNewPeerConnection()` is never entered and signaling is silent.

The upstream correction adds this immediately after `await initialize()`:

```ts
if (this._connectResolve && !this._mediaStreamRecorder) {
  this._connectResolve();
  this._connectResolve = null;
}
```

That is the missing settlement for the recorder-disabled manager.

## Fixability with the frozen environment

### Not possible on the server

There is no server-side interception point. The second `/start` succeeds, but
the browser never constructs a peer connection and never posts an offer. The
server cannot close, reject, replace, or map a connection that it is never
asked to create. The proposed `demo.py` terminal-close logic therefore does not
address this bug and would add unsupported coupling to Pipecat internals.

The retained `Reusing existing connection for pc_id` lines are also not evidence
of this reconnect failure. In the inspected logs they occur during the initial
connection's normal offer/ICE renegotiation, before the worker is ready. On
actual disconnect, `request_handler.handle_disconnected` discards the map entry
before the application callback cancels the worker.

### Possible without upgrading Python Pipecat

There are two technically valid frontend-only paths:

1. Rebuild and serve a repository-owned copy of the v1.0.5 prebuilt UI with the
   six-line upstream `DailyMediaManager.connect()` correction applied to its
   pinned SmallWebRTC transport dependency. Mount that artifact instead of the
   installed `PipecatPrebuiltUI`. This preserves the Python model/runtime
   dependency versions while changing the defective browser artifact.
2. Change the prebuilt UI's Connect handler to await `client.initDevices()`
   before every `client.connect()`. Upstream issue 154 identifies this as a
   working workaround because it restores `_initialized=true`, keeping
   `DailyMediaManager.connect()` on its fast path.

The first option matches the actual upstream fix and is the narrower semantic
change. It should be preferred if a repository-local frontend patch is
authorized. The build must be pinned by source commit, lockfile, and artifact
hash; editing the installed one-line minified bundle in place would be brittle
and unreproducible.

Neither path is expressible through the current Python runner or `/start`
response. The prebuilt app does not expose a Python-side hook around its
internal Connect handler. If “environment frozen” includes the served prebuilt
asset itself and forbids a repository-owned replacement, the answer is no: the
same-page reconnect cannot be repaired under that contract.

## Scope and repository state

The earlier experimental `demo.py` lifecycle change and browser/unit-test
changes were removed. This analysis made no change under `src/` or `tests/`.
No fix, dependency update, browser asset patch, or commit was made.
