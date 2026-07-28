"""The envelope every endpoint answers with, and the helpers that build it.

Modelled on Rust's ``Result<T, E>`` and Go's ``(value, err)``: a response either
carries ``data`` or an ``error``, never both, and one field says which::

    {"ok": true,  "data": {…}, "error": null,                          "request_id": "…"}
    {"ok": false, "data": null, "error": {"code": "not_found", "message": "…"}, "request_id": "…"}

An error describes the failure with a stable ``code`` clients may branch on and
a message written for the person who will read it. Nothing about how the failure
happened — exception types, stack frames, SQL — ever reaches the client;
``request_id`` is what ties their copy of the failure to the server logs.

Note ``Response`` here is the response *body*, not ``fastapi.Response``.
"""

from __future__ import annotations

from enum import StrEnum
from http import HTTPStatus
from typing import Any, Generic, Literal, TypeVar, cast

from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.core.tracing import REQUEST_ID_HEADER, current_request_id

T = TypeVar("T")


class ErrorCode(StrEnum):
    """What went wrong, in a form code can branch on.

    Values are part of the API: once released a code keeps its meaning, while
    the message next to it may be reworded at any time.
    """

    # Generic, derived from the status code.
    BAD_REQUEST = "bad_request"
    VALIDATION_ERROR = "validation_error"
    UNAUTHORIZED = "unauthorized"
    FORBIDDEN = "forbidden"
    NOT_FOUND = "not_found"
    METHOD_NOT_ALLOWED = "method_not_allowed"
    CONFLICT = "conflict"
    TOO_MANY_REQUESTS = "too_many_requests"
    INTERNAL_ERROR = "internal_error"
    SERVICE_UNAVAILABLE = "service_unavailable"

    # Domain errors, raised by the repositories.
    UNKNOWN_LABELS = "unknown_labels"
    DUPLICATE_LABEL_NAME = "duplicate_label_name"
    INVALID_DATE_RANGE = "invalid_date_range"


_CODES_BY_STATUS: dict[int, ErrorCode] = {
    400: ErrorCode.BAD_REQUEST,
    401: ErrorCode.UNAUTHORIZED,
    403: ErrorCode.FORBIDDEN,
    404: ErrorCode.NOT_FOUND,
    405: ErrorCode.METHOD_NOT_ALLOWED,
    409: ErrorCode.CONFLICT,
    422: ErrorCode.VALIDATION_ERROR,
    429: ErrorCode.TOO_MANY_REQUESTS,
    500: ErrorCode.INTERNAL_ERROR,
    503: ErrorCode.SERVICE_UNAVAILABLE,
}


def code_for_status(status_code: int) -> ErrorCode:
    """The error code an HTTP status maps to."""
    code = _CODES_BY_STATUS.get(status_code)
    if code is not None:
        return code
    return ErrorCode.INTERNAL_ERROR if status_code >= 500 else ErrorCode.BAD_REQUEST


def message_for_status(status_code: int) -> str:
    """A safe stand-in message for a status with nothing better to say."""
    try:
        return HTTPStatus(status_code).phrase
    except ValueError:
        return "Request failed"


class ApiError(BaseModel):
    """The error half of a response. Safe to show to a user as it stands."""

    code: ErrorCode
    message: str = Field(description="What went wrong, phrased for the reader")
    details: Any = Field(
        default=None,
        description="Extra context, shaped however this error code sees fit. "
        "The code says how to read it; clients that do not know the code "
        "should show the message and ignore this.",
    )


class UnwrapError(RuntimeError):
    """Raised by ``Response.unwrap`` when the response carries an error."""


class Response(BaseModel, Generic[T]):
    """Every endpoint answers with this, success or failure."""

    ok: bool = Field(description="True when data is present, false when error is")
    data: T | None = Field(default=None, description="The payload, on success")
    error: ApiError | None = Field(default=None, description="The failure, on error")
    request_id: str = Field(
        description="Trace id for this call, shared across services. "
        "Quote it when reporting a problem."
    )

    # Result-style accessors, for the services and tests that consume this.
    def is_ok(self) -> bool:
        return self.ok

    def is_err(self) -> bool:
        return not self.ok

    def unwrap(self) -> T:
        """The payload, or ``UnwrapError`` if this response is an error."""
        if not self.ok:
            error = self.error
            raise UnwrapError(
                f"[{error.code}] {error.message}"
                if error is not None
                else "response is an error"
            )
        return cast(T, self.data)

    def unwrap_or(self, default: T) -> T:
        """The payload, or ``default`` if this response is an error."""
        return cast(T, self.data) if self.ok else default


class ErrorResponse(BaseModel):
    """``Response`` narrowed to the failure case: ``data`` out, ``error`` in.

    The same four fields, spelled out separately rather than inherited so the
    narrowing survives type checking. This is what ``err`` builds and what every
    failing status is documented as, so the docs describe an error the way it
    actually arrives instead of falling back to FastAPI's validation schema.
    """

    ok: Literal[False] = False
    data: None = Field(default=None, description="Always null on an error")
    error: ApiError
    request_id: str = Field(
        description="Trace id for this call. Quote it when reporting the problem."
    )


# Builders. The only supported way to make an envelope.
def ok(data: T, *, request_id: str | None = None) -> Response[T]:
    """A success response around ``data``.

    Pass ``None`` for endpoints with nothing to return::

        return ok(TaskResponse.model_validate(task))
        return ok(None)

    ``request_id`` is taken from the current request unless given explicitly.
    """
    return cast(
        Response[T],
        Response[Any](
            ok=True,
            data=data,
            error=None,
            request_id=request_id or current_request_id(),
        ),
    )


def err(
    code: ErrorCode,
    message: str,
    *,
    details: Any = None,
    request_id: str | None = None,
) -> ErrorResponse:
    """A failure response.

    ``message`` is shown to the user as written, so it must describe the problem
    and not the machinery: no exception text, no identifiers they cannot act on.
    ``details`` is free-form and travels with the code; keep it JSON-serialisable
    and keep anything the user should not see out of it.
    """
    return ErrorResponse(
        error=ApiError(code=code, message=message, details=details),
        request_id=request_id or current_request_id(),
    )


def error_response(
    status_code: int,
    message: str,
    *,
    code: ErrorCode | None = None,
    details: Any = None,
    request_id: str | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    """``err`` as a ready-to-send HTTP response, for the exception handlers.

    ``code`` defaults to whatever the status maps to.
    """
    request_id = request_id or current_request_id()
    body = err(
        code or code_for_status(status_code),
        message,
        details=details,
        request_id=request_id,
    )
    # Set here as well as in the middleware: the handler for unhandled errors
    # answers from outside it, where the response headers are no longer touched.
    return JSONResponse(
        status_code=status_code,
        content=body.model_dump(mode="json"),
        headers={REQUEST_ID_HEADER: request_id, **(headers or {})},
    )


_ERROR_DESCRIPTIONS: dict[int, str] = {
    400: "The request was refused; `error.code` says why.",
    401: "Credentials are missing, expired or invalid.",
    403: "Authenticated, but this resource belongs to someone else.",
    404: "No such resource — or none this caller is allowed to see.",
    409: "Conflicts with something that already exists.",
    422: "The request failed validation; `error.details` lists the fields.",
    429: "Too many requests.",
    500: "Something went wrong on our side; quote `request_id` when reporting it.",
    503: "Temporarily unable to serve the request.",
}


def error_responses(*status_codes: int) -> dict[int | str, dict[str, Any]]:
    """OpenAPI ``responses`` entries documenting failures as this envelope.

    Attach to a router or a route so the docs describe an error the way it is
    actually returned. Declaring 422 also stops FastAPI documenting its own
    ``HTTPValidationError`` shape, which this app never sends::

        router = APIRouter(responses=error_responses(401, 404))
    """
    return {
        status_code: {
            "model": ErrorResponse,
            "description": _ERROR_DESCRIPTIONS.get(
                status_code, message_for_status(status_code)
            ),
        }
        for status_code in status_codes
    }
