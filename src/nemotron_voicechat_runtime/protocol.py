"""Strict Voicechat realtime WebSocket protocol state.

The inference engine exposes cumulative text and per-frame control state. This
module turns those signals into one-shot, response-scoped protocol events. It is
pure Python so lifecycle behavior can be tested without loading NeMo or CUDA.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import Any

PROTOCOL_NAME = "voicechat.realtime"
PROTOCOL_VERSION = 3
FUNCTION_OUTPUT_MAX_BYTES = 16 * 1024
FUNCTION_OUTPUT_MAX_TOKENS = 128
FUNCTION_OUTPUT_RECOVERY_MAX_FRAMES = 256
FUNCTION_OUTPUT_RECOVERY_FRAME_SECONDS = 0.08
FUNCTION_OUTPUT_ACK_TIMEOUT_SECONDS = 30.0
MAX_FUNCTION_CALLS_PER_RESPONSE = 8


def protocol_capabilities(
    *,
    function_call_timeout_seconds: float,
    max_session_model_frames: int,
    typed_input: bool = False,
    client_turn_detection: bool = False,
) -> dict[str, Any]:
    return {
        "server_turn_detection": None if client_turn_detection else "rnnt",
        "input_turn_detection": (
            "client_smart_turn_v1" if client_turn_detection else "server_rnnt"
        ),
        "input_transcription": True,
        "function_calling": True,
        "typed_input": typed_input,
        "settings_mutable_until": "first_model_input_frame",
        "single_client": True,
        "function_call_timeout_seconds": function_call_timeout_seconds,
        "function_output_max_bytes": FUNCTION_OUTPUT_MAX_BYTES,
        "function_output_max_tokens": FUNCTION_OUTPUT_MAX_TOKENS,
        "function_output_recovery_max_frames": FUNCTION_OUTPUT_RECOVERY_MAX_FRAMES,
        "function_output_ack_timeout_seconds": FUNCTION_OUTPUT_ACK_TIMEOUT_SECONDS,
        "max_function_calls_per_response": MAX_FUNCTION_CALLS_PER_RESPONSE,
        "function_event_emit_timeout_seconds": 5.0,
        "max_session_model_frames": max_session_model_frames,
        "binary_audio": False,
    }


@dataclass
class RealtimeProtocolSession:
    """Generate stable IDs and exact turn/response lifecycle events."""

    session_id: str
    capabilities: dict[str, Any]
    _event_index: int = 0
    _turn_index: int = 0
    _response_index: int = 0
    _turn_id: str | None = None
    _response_id: str | None = None
    _response_turn_id: str | None = None
    _user_open: bool = False
    _response_open: bool = False
    _response_audio_started: bool = False
    _response_text: str = ""
    _response_cancel_reason: str | None = None
    _completed_user_text: str = ""
    _turn_transcript: str = ""
    _turn_transcript_emitted: int = 0
    _input_source: str = "microphone"
    _input_job_id: str | None = None
    _turn_source: str = "microphone"
    _turn_job_id: str | None = None
    _turn_correlation: str | int | None = None
    _pending_turn_transcript: str = ""
    _pending_decoded_token_count: int = 0
    _function_call_scopes: dict[str, tuple[str | None, str | None]] = field(default_factory=dict)

    def _event(self, event_type: str, **fields: Any) -> dict[str, Any]:
        self._event_index += 1
        event = {
            "type": event_type,
            "event_id": f"evt_{self.session_id}_{self._event_index}",
            "session_id": self.session_id,
        }
        event.update(fields)
        return event

    def session_created(self, session: dict[str, Any]) -> dict[str, Any]:
        return self._event(
            "session.created",
            protocol={"name": PROTOCOL_NAME, "version": PROTOCOL_VERSION},
            capabilities=self.capabilities,
            session={"id": self.session_id, **session},
        )

    def session_updated(self, session: dict[str, Any]) -> dict[str, Any]:
        return self._event(
            "session.updated",
            protocol={"name": PROTOCOL_NAME, "version": PROTOCOL_VERSION},
            capabilities=self.capabilities,
            session={
                "id": self.session_id,
                "protocol_version": PROTOCOL_VERSION,
                **session,
            },
        )

    def error(
        self,
        code: str,
        message: str,
        *,
        fatal: bool = False,
        client_event_id: str | None = None,
        **details: Any,
    ) -> dict[str, Any]:
        error = {"code": code, "message": message, "fatal": fatal, **details}
        if client_event_id:
            error["client_event_id"] = client_event_id
        return self._event("error", error=error)

    def session_closed(self, reason: str, *, status: str = "completed") -> dict[str, Any]:
        return self._event("session.closed", status=status, reason=reason)

    def close_events(
        self,
        reason: str,
        *,
        cumulative_user_text: str = "",
        status: str = "completed",
    ) -> list[dict[str, Any]]:
        """Close any open brackets, then close the permanent session."""
        events = self._finish_turn(cumulative_user_text)
        if self._response_open:
            events.extend(self._finish_response({"boundary_reason": reason, "eos_reason": reason}))
        events.append(self.session_closed(reason, status=status))
        return events

    def diagnostic_event(self, event_type: str, **fields: Any) -> dict[str, Any]:
        """Create a correlated additive diagnostic event."""
        return self._event(event_type, **fields)

    @property
    def response_open(self) -> bool:
        return self._response_open

    def cancel_active_response(self, reason: str) -> None:
        if self._response_open:
            self._response_cancel_reason = reason

    def set_input_source(self, source: str, job_id: str | None = None) -> None:
        """Set provenance for the next RNNT-owned user turn."""

        if source not in {"microphone", "typed"}:
            raise ValueError(f"unsupported input source: {source}")
        if source == "typed" and not job_id:
            raise ValueError("typed input source requires job_id")
        self._input_source = source
        self._input_job_id = job_id if source == "typed" else None

    def typed_input_accepted(self, job_id: str) -> dict[str, Any]:
        return self._event("input_text.accepted", job_id=job_id)

    def typed_input_rejected(self, job_id: str | None, reason: str) -> dict[str, Any]:
        return self._event("input_text.rejected", job_id=job_id, reason=reason)

    def typed_input_started(self, job_id: str) -> dict[str, Any]:
        return self._event("input_text.injection_started", job_id=job_id)

    def typed_input_finished(self, job_id: str, disposition: str) -> dict[str, Any]:
        if disposition not in {"completed", "cancelled", "replaced", "error"}:
            raise ValueError(f"invalid typed-input disposition: {disposition}")
        return self._event(
            "input_text.injection_finished",
            job_id=job_id,
            disposition=disposition,
        )

    def begin_input_turn(
        self,
        source: str,
        correlation: str | int,
        *,
        job_id: str | None = None,
    ) -> dict[str, Any]:
        """Open one explicitly controlled input turn."""
        if self._user_open:
            raise ValueError("an input turn is already open")
        self.set_input_source(source, job_id)
        self._turn_correlation = correlation
        return self._new_turn()

    def input_turn_started_ack(
        self,
        client_event_id: str,
        client_turn_id: int,
        *,
        turn_id: str | None = None,
    ) -> dict[str, Any]:
        return self._event(
            "input_audio_buffer.turn_started",
            client_event_id=client_event_id,
            client_turn_id=client_turn_id,
            turn_id=self._turn_id if turn_id is None else turn_id,
        )

    def input_turn_committed_ack(
        self,
        client_event_id: str,
        client_turn_id: int,
        *,
        turn_id: str | None = None,
    ) -> dict[str, Any]:
        return self._event(
            "input_audio_buffer.committed",
            client_event_id=client_event_id,
            client_turn_id=client_turn_id,
            turn_id=self._turn_id if turn_id is None else turn_id,
        )

    def _new_turn(self) -> dict[str, Any]:
        self._turn_index += 1
        self._turn_id = f"turn_{self.session_id}_{self._turn_index}"
        self._user_open = True
        self._turn_source = self._input_source
        self._turn_job_id = self._input_job_id
        self._turn_transcript = self._pending_turn_transcript
        self._turn_transcript_emitted = 0
        self._pending_turn_transcript = ""
        self._pending_decoded_token_count = 0
        return self._event(
            "input_audio_buffer.speech_started",
            turn_id=self._turn_id,
            source=self._turn_source,
            job_id=self._turn_job_id,
            correlation=self._turn_correlation,
        )

    def _update_turn_transcript(
        self, current_text: str, *, decoded_token_count: int = 0
    ) -> list[dict[str, Any]]:
        if not self._user_open:
            return []
        # The public RNNT display is turn-scoped but retains the last finalized
        # string until the next turn emits text. A changed string is therefore
        # the new turn's complete partial. decoded_token_count disambiguates an
        # exact repeated utterance ("yes" followed by "yes").
        if current_text != self._completed_user_text or decoded_token_count > 0:
            self._turn_transcript = current_text
        delta = self._turn_transcript[self._turn_transcript_emitted :]
        if not delta:
            return []
        self._turn_transcript_emitted = len(self._turn_transcript)
        return [
            self._event(
                "conversation.item.input_audio_transcription.delta",
                turn_id=self._turn_id,
                delta=delta,
                transcript=self._turn_transcript,
                source=self._turn_source,
                job_id=self._turn_job_id,
            )
        ]

    def _finish_turn(
        self, current_text: str, *, decoded_token_count: int = 0
    ) -> list[dict[str, Any]]:
        if not self._user_open:
            return []
        events = self._update_turn_transcript(current_text, decoded_token_count=decoded_token_count)
        transcript = self._turn_transcript
        events.extend(
            [
                self._event(
                    "input_audio_buffer.speech_stopped",
                    turn_id=self._turn_id,
                    source=self._turn_source,
                    job_id=self._turn_job_id,
                ),
                self._event(
                    "conversation.item.input_audio_transcription.completed",
                    turn_id=self._turn_id,
                    transcript=transcript,
                    empty=not bool(transcript.strip()),
                    source=self._turn_source,
                    job_id=self._turn_job_id,
                ),
            ]
        )
        self._completed_user_text = transcript
        self._user_open = False
        self._turn_source = "microphone"
        self._turn_job_id = None
        self._turn_correlation = None
        return events

    def _start_response(self) -> dict[str, Any] | None:
        if self._response_open:
            return None
        self._response_index += 1
        self._response_id = f"response_{self.session_id}_{self._response_index}"
        self._response_turn_id = self._turn_id
        self._response_open = True
        self._response_audio_started = False
        self._response_text = ""
        self._response_cancel_reason = None
        return self._event(
            "response.created",
            turn_id=self._response_turn_id,
            response_id=self._response_id,
        )

    def _finish_response(
        self,
        boundary: dict[str, Any],
        *,
        status_override: str | None = None,
    ) -> list[dict[str, Any]]:
        if not self._response_open:
            return []
        common = {
            "turn_id": self._response_turn_id,
            "response_id": self._response_id,
        }
        reason = (
            boundary.get("eos_reason") or boundary.get("boundary_reason") or "model_or_turn_taking"
        )
        status = status_override or ("cancelled" if self._response_cancel_reason else "completed")
        if self._response_cancel_reason:
            reason = self._response_cancel_reason
        events = [
            self._event(
                "response.output_text.done",
                **common,
                text=self._response_text,
                empty=not bool(self._response_text),
            ),
            self._event(
                "response.output_audio.done",
                **common,
                empty=not self._response_audio_started,
            ),
            self._event(
                "response.done",
                **common,
                status=status,
                reason=reason,
            ),
        ]
        self._response_open = False
        self._response_id = None
        self._response_turn_id = None
        self._response_audio_started = False
        self._response_text = ""
        self._response_cancel_reason = None
        return events

    def fail_active_response(self, reason: str) -> list[dict[str, Any]]:
        """Close an open model response as failed without fabricating output."""
        if not self._response_open:
            return []
        return self._finish_response(
            {"boundary_reason": reason, "eos_reason": reason},
            status_override="failed",
        )

    def step_events(
        self,
        result: Any,
        *,
        output_pcm: bytes,
        audio_delivered: bool,
        output_sample_rate: int,
        synthetic_control: bool = False,
        suppress_response_output: bool = False,
    ) -> list[dict[str, Any]]:
        """Translate one model step into correctly ordered protocol events."""

        events: list[dict[str, Any]] = []
        turn_state = result.turn_state or {}
        rnnt = turn_state.get("rnnt") or {}
        decoded_token_count = int(rnnt.get("decoded_token_count") or 0)
        control = turn_state.get("agent_control")
        boundary = turn_state.get("response_boundary") or {}
        client_turn_detection = (
            self.capabilities.get("input_turn_detection") == "client_smart_turn_v1"
        )

        if client_turn_detection and not self._user_open and not synthetic_control:
            if result.user_text != self._completed_user_text or decoded_token_count > 0:
                self._pending_turn_transcript = result.user_text
                self._pending_decoded_token_count = decoded_token_count

        # RNNT speech-confirmed is a per-turn state. It is cleared on the same
        # step that injects agent BOS, so start the turn before handling BOS.
        if not client_turn_detection and bool(rnnt.get("speech_confirmed")) and not self._user_open:
            events.append(self._new_turn())
        events.extend(
            self._update_turn_transcript(result.user_text, decoded_token_count=decoded_token_count)
        )

        response_start = boundary.get("event") == "start" or control == "agent_bos"
        if response_start:
            # The accepted agent-BOS is the observable EOU edge in this wrapper.
            events.extend(
                self._finish_turn(result.user_text, decoded_token_count=decoded_token_count)
            )
            created = self._start_response()
            if created:
                events.append(created)

        common = {
            "turn_id": self._response_turn_id,
            "response_id": self._response_id,
        }
        if result.assistant_delta and not suppress_response_output:
            if not self._response_open:
                created = self._start_response()
                if created:
                    events.append(created)
                common["response_id"] = self._response_id
            self._response_text += result.assistant_delta
            events.append(
                self._event(
                    "response.output_text.delta",
                    **common,
                    delta=result.assistant_delta,
                    text=self._response_text,
                )
            )
        if output_pcm and audio_delivered and not suppress_response_output:
            if not self._response_open:
                created = self._start_response()
                if created:
                    events.append(created)
                common["response_id"] = self._response_id
            self._response_audio_started = True
            events.append(
                self._event(
                    "response.output_audio.delta",
                    **common,
                    delta=base64.b64encode(output_pcm).decode("ascii"),
                    encoding="pcm16",
                    sample_rate=output_sample_rate,
                    channels=1,
                )
            )

        if boundary.get("event") == "end":
            events.extend(self._finish_response(boundary))
        return events

    def function_call_events(
        self,
        *,
        call_id: str,
        name: str,
        arguments: str,
    ) -> list[dict[str, Any]]:
        created = self._start_response()
        events = [created] if created is not None else []
        event = self._event(
            "response.function_call_arguments.done",
            turn_id=self._response_turn_id,
            response_id=self._response_id,
            call_id=call_id,
            name=name,
            arguments=arguments,
        )
        self._function_call_scopes[call_id] = (
            self._response_turn_id,
            self._response_id,
        )
        events.append(event)
        return events

    def function_call_failed_event(self, *, call_id: str, reason: str) -> dict[str, Any]:
        turn_id, response_id = self._function_call_scopes.pop(
            call_id, (self._response_turn_id, self._response_id)
        )
        return self._event(
            "response.function_call.failed",
            turn_id=turn_id,
            response_id=response_id,
            call_id=call_id,
            reason=reason,
        )

    def function_call_output_applied(self, call_id: str) -> dict[str, Any]:
        return self._event(
            "conversation.item.function_call_output.applied",
            call_id=call_id,
        )

    def function_call_completed(self, call_id: str) -> None:
        self._function_call_scopes.pop(call_id, None)

    @property
    def response_id(self) -> str | None:
        return self._response_id

    @property
    def turn_id(self) -> str | None:
        return self._turn_id
