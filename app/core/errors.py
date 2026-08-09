"""Every failure leaves the app through here, shaped like every success.

Routes raise (``HTTPException``) and repositories raise domain errors; both are
turned into the same envelope, with a message written for whoever reads it. An
unhandled exception is the one case where the app has nothing safe to say, so it
says so, and the traceback goes to the log under the same request id.
"""

from __future__ import annotations

import logging
from typing import Any, cast

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.response import (
    ErrorCode,
    code_for_status,
    error_response,
    message_for_status,
)
from app.core.tracing import current_request_id, trace_context_from_scope
from app.repo.base import (
    DuplicateAgentNameError,
    DuplicateGitRepoNameError,
    DuplicateLabelNameError,
    ExecutionHistoryExistsError,
    InterventionAlreadyResolvedError,
    InterventionKindMismatchError,
    InvalidCodeChangeError,
    InvalidDateRangeError,
    InvalidTransitionError,
    RepoAlreadyAttachedError,
    RepoError,
    RepoNotAttachedError,
    ResourceInUseError,
    UnknownLabelsError,
    UnknownParentEventError,
)

logger = logging.getLogger(__name__)

UNEXPECTED_ERROR_MESSAGE = (
    "Something went wrong on our side. Try again, and quote the request id "
    "below if it keeps happening."
)

# Domain errors carry a message meant for the user, so only the status and the
# code have to be decided here. Anything not listed is a client mistake we have
# not named yet.
REPO_ERRORS: dict[type[RepoError], tuple[int, ErrorCode]] = {
    UnknownLabelsError: (status.HTTP_400_BAD_REQUEST, ErrorCode.UNKNOWN_LABELS),
    DuplicateLabelNameError: (status.HTTP_409_CONFLICT, ErrorCode.DUPLICATE_LABEL_NAME),
    InvalidDateRangeError: (
        status.HTTP_422_UNPROCESSABLE_CONTENT,
        ErrorCode.INVALID_DATE_RANGE,
    ),
    DuplicateAgentNameError: (
        status.HTTP_409_CONFLICT,
        ErrorCode.DUPLICATE_AGENT_NAME,
    ),
    DuplicateGitRepoNameError: (
        status.HTTP_409_CONFLICT,
        ErrorCode.DUPLICATE_REPO_NAME,
    ),
    ResourceInUseError: (status.HTTP_409_CONFLICT, ErrorCode.RESOURCE_IN_USE),
    InvalidTransitionError: (
        status.HTTP_409_CONFLICT,
        ErrorCode.INVALID_TRANSITION,
    ),
    ExecutionHistoryExistsError: (
        status.HTTP_409_CONFLICT,
        ErrorCode.HISTORY_EXISTS,
    ),
    RepoAlreadyAttachedError: (
        status.HTTP_409_CONFLICT,
        ErrorCode.REPO_ALREADY_ATTACHED,
    ),
    RepoNotAttachedError: (
        status.HTTP_409_CONFLICT,
        ErrorCode.REPO_NOT_ATTACHED,
    ),
    UnknownParentEventError: (
        status.HTTP_400_BAD_REQUEST,
        ErrorCode.UNKNOWN_PARENT_EVENT,
    ),
    InvalidCodeChangeError: (
        status.HTTP_422_UNPROCESSABLE_CONTENT,
        ErrorCode.INVALID_CODE_CHANGE,
    ),
    InterventionAlreadyResolvedError: (
        status.HTTP_409_CONFLICT,
        ErrorCode.INTERVENTION_ALREADY_RESOLVED,
    ),
    InterventionKindMismatchError: (
        status.HTTP_409_CONFLICT,
        ErrorCode.INTERVENTION_KIND_MISMATCH,
    ),
}


def request_id_of(request: Request) -> str:
    """The trace id of a request.

    Read off the scope rather than the context variable so it still resolves in
    the handler for unhandled errors, which runs outside the middleware.
    """
    context = trace_context_from_scope(request.scope)
    return context.request_id if context is not None else current_request_id()


async def http_exception_handler(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    """Anything raised as an ``HTTPException``, plus Starlette's own 404/405."""
    # A non-string detail is a dict some route chose to send; the status phrase
    # is the only thing safe to put in front of a user in that case.
    message = exc.detail if isinstance(exc.detail, str) else None
    return error_response(
        exc.status_code,
        message or message_for_status(exc.status_code),
        code=code_for_status(exc.status_code),
        details=None if message else exc.detail,
        request_id=request_id_of(request),
        # Kept so 401s still tell the client how to authenticate.
        headers=dict(exc.headers) if exc.headers else None,
    )


async def validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """A malformed request. The details say which parts, field by field."""
    details: list[dict[str, Any]] = [
        {
            "field": ".".join(str(part) for part in error.get("loc", ())),
            "message": error.get("msg", "Invalid value"),
            "type": error.get("type"),
        }
        # Only these three keys: pydantic also reports "input" and "ctx", which
        # echo back what the client sent — a submitted password, for instance.
        for error in exc.errors()
    ]
    return error_response(
        status.HTTP_422_UNPROCESSABLE_CONTENT,
        "The request is not valid. See details for the fields to fix.",
        code=ErrorCode.VALIDATION_ERROR,
        details=details,
        request_id=request_id_of(request),
    )


async def repo_error_handler(request: Request, exc: RepoError) -> JSONResponse:
    """Domain errors from the repositories, e.g. a label name already in use."""
    status_code, code = REPO_ERRORS.get(
        type(exc), (status.HTTP_400_BAD_REQUEST, ErrorCode.BAD_REQUEST)
    )
    return error_response(
        status_code, str(exc), code=code, request_id=request_id_of(request)
    )


async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """A bug. The client gets a request id, the log gets the traceback."""
    request_id = request_id_of(request)
    logger.exception(
        "Unhandled error on %s %s (request_id=%s)",
        request.method,
        request.url.path,
        request_id,
        exc_info=exc,
    )
    return error_response(
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        UNEXPECTED_ERROR_MESSAGE,
        code=ErrorCode.INTERNAL_ERROR,
        request_id=request_id,
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Wire the handlers above onto the app."""
    handlers: tuple[tuple[type[Exception], Any], ...] = (
        (StarletteHTTPException, http_exception_handler),
        (RequestValidationError, validation_error_handler),
        # Registered on the base class: Starlette walks the exception's MRO, so
        # every current and future subclass lands here.
        (RepoError, repo_error_handler),
        (Exception, unhandled_error_handler),
    )
    for exception_type, handler in handlers:
        # Starlette types a handler as taking a bare Exception; each of ours is
        # narrowed to the type it is registered for.
        app.add_exception_handler(exception_type, cast(Any, handler))
