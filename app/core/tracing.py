"""Request ids, carried as W3C Trace Context.

One id follows a call across every service that touches it: it is read from the
inbound ``traceparent`` header when the caller sent one, and generated when they
did not. It is reported on every response body as ``request_id`` and echoed in
the ``traceparent`` and ``X-Request-Id`` headers. Calls this service makes to
other services should forward ``trace_headers()`` so the trace continues.

Reference: https://www.w3.org/TR/trace-context/
"""

from __future__ import annotations

import re
import uuid
from contextvars import ContextVar
from dataclasses import dataclass

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

TRACEPARENT_HEADER = "traceparent"
REQUEST_ID_HEADER = "x-request-id"

# Key the context is published under on the ASGI scope state. Exception handlers
# read it from there rather than from the context variable: the handler for
# unhandled errors runs outside this middleware, where the variable has already
# been reset.
SCOPE_KEY = "trace_context"

_VERSION = "00"
_SAMPLED_FLAG = 0b1
_HEX = re.compile(r"[0-9a-f]+")
_ZERO_TRACE_ID = "0" * 32
_ZERO_SPAN_ID = "0" * 16


def new_trace_id() -> str:
    """A fresh 32 hex character trace id."""
    return uuid.uuid4().hex


def new_span_id() -> str:
    """A fresh 16 hex character span id, identifying one hop of a trace."""
    return uuid.uuid4().hex[:16]


def _is_hex(value: str, length: int) -> bool:
    return len(value) == length and _HEX.fullmatch(value) is not None


def parse_traceparent(value: str | None) -> tuple[str, str, bool] | None:
    """Read a ``traceparent`` header into ``(trace_id, parent_span_id, sampled)``.

    Returns None for anything malformed, so a caller sending a broken header
    simply gets a new trace instead of an error.
    """
    if not value:
        return None

    fields = value.strip().lower().split("-")
    if len(fields) < 4:
        return None

    version, trace_id, span_id, flags = fields[:4]
    # Version 00 is exactly four fields. Later versions may append more, and the
    # spec says to read the first four and ignore the rest; "ff" is forbidden.
    if not _is_hex(version, 2) or version == "ff":
        return None
    if version == _VERSION and len(fields) != 4:
        return None
    if not _is_hex(trace_id, 32) or trace_id == _ZERO_TRACE_ID:
        return None
    if not _is_hex(span_id, 16) or span_id == _ZERO_SPAN_ID:
        return None
    if not _is_hex(flags, 2):
        return None

    return trace_id, span_id, bool(int(flags, 16) & _SAMPLED_FLAG)


@dataclass(frozen=True, slots=True)
class TraceContext:
    """The trace one request belongs to."""

    trace_id: str
    """32 hex characters, shared by every service handling this call."""

    span_id: str
    """16 hex characters, this service's own id for its part of the call."""

    parent_span_id: str | None = None
    """The caller's span, when the caller sent a usable ``traceparent``."""

    sampled: bool = True

    @classmethod
    def start(cls, traceparent: str | None = None) -> TraceContext:
        """Continue the caller's trace, or begin a new one."""
        parsed = parse_traceparent(traceparent)
        if parsed is None:
            return cls(trace_id=new_trace_id(), span_id=new_span_id())

        trace_id, parent_span_id, sampled = parsed
        return cls(
            trace_id=trace_id,
            span_id=new_span_id(),
            parent_span_id=parent_span_id,
            sampled=sampled,
        )

    @property
    def request_id(self) -> str:
        """The id reported to the client and written to the logs."""
        return self.trace_id

    @property
    def traceparent(self) -> str:
        """This context as a ``traceparent`` header value."""
        return f"{_VERSION}-{self.trace_id}-{self.span_id}-{'01' if self.sampled else '00'}"


_current: ContextVar[TraceContext | None] = ContextVar("trace_context", default=None)


def current_trace_context() -> TraceContext:
    """The context of the request being handled.

    Outside a request — a script, a worker, a test — there is no inbound header,
    so a trace is started on first use and reused for the rest of that context.
    """
    context = _current.get()
    if context is None:
        context = TraceContext.start()
        _current.set(context)
    return context


def current_request_id() -> str:
    """The id of the request being handled. See ``current_trace_context``."""
    return current_trace_context().request_id


def trace_context_from_scope(scope: Scope) -> TraceContext | None:
    """The context ``RequestIdMiddleware`` published on an ASGI scope, if any."""
    context = scope.get("state", {}).get(SCOPE_KEY)
    return context if isinstance(context, TraceContext) else None


def trace_headers() -> dict[str, str]:
    """Headers to send with outgoing calls so they join the current trace."""
    context = current_trace_context()
    return {
        TRACEPARENT_HEADER: context.traceparent,
        REQUEST_ID_HEADER: context.request_id,
    }


class RequestIdMiddleware:
    """Give every request a trace, and report it back on the response.

    Written against the raw ASGI interface rather than ``BaseHTTPMiddleware`` so
    the context variable is set in the same task that runs the endpoint.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        inbound = Headers(scope=scope).get(TRACEPARENT_HEADER)
        context = TraceContext.start(inbound)

        scope.setdefault("state", {})[SCOPE_KEY] = context
        token = _current.set(context)

        async def send_with_trace(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers[REQUEST_ID_HEADER] = context.request_id
                headers[TRACEPARENT_HEADER] = context.traceparent
            await send(message)

        try:
            await self.app(scope, receive, send_with_trace)
        finally:
            _current.reset(token)
