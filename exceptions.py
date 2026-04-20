"""
exceptions.py — Typed Exception Hierarchy for Verity
=====================================================
Follows FR-2.1: Define typed exception hierarchy.

All exceptions should use this module. Never raise generic Exception.
"""
from __future__ import annotations

import logging
import traceback
import uuid
from abc import ABC
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from typing import Callable

logger = logging.getLogger(__name__)


# ─── Error Context ────────────────────────────────────────────────────────────


@dataclass
class ErrorContext:
    """Structured error context for debugging and logging (FR-2.1)."""
    correlation_id: str | None = None
    stage: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def with_correlation(self, correlation_id: str | None = None) -> ErrorContext:
        """Create new context with correlation ID."""
        if correlation_id is None:
            correlation_id = str(uuid.uuid4())[:8]
        return ErrorContext(
            correlation_id=correlation_id,
            stage=self.stage,
            details=self.details,
            timestamp=self.timestamp,
        )


# ─── Result Type ──────────────────────────────────────────────────────────────


@dataclass
class Result[T]:
    """Explicit Result type for fallible operations (FR-2.7)."""
    value: T | None = None
    error: VerityError | None = None

    @classmethod
    def ok(cls, value: T) -> Result[T]:
        """Create success result."""
        return cls(value=value, error=None)

    @classmethod
    def fail(cls, error: VerityError) -> Result[T]:
        """Create failure result."""
        return cls(value=None, error=error)

    @property
    def is_ok(self) -> bool:
        """Check if result is success."""
        return self.error is None

    @property
    def is_fail(self) -> bool:
        """Check if result is failure."""
        return self.error is not None

    def unwrap(self) -> T:
        """Unwrap value, raise if error."""
        if self.error is not None:
            raise self.error
        if self.value is None:
            raise VeritySystemError("Result has no value")
        return self.value

    def unwrap_or(self, default: T) -> T:
        """Unwrap value or return default."""
        return self.value if self.error is None else default


# ─── Base Exceptions ──────────────────────────────────────────────────────────


class VerityBaseException(ABC, Exception):
    """Base exception for all Verity errors (FR-2.1)."""

    def __init__(
        self,
        message: str,
        context: ErrorContext | None = None,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message, cause)
        self.message = message
        self.context = context or ErrorContext()
        self.cause = cause

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "type": self.__class__.__name__,
            "message": self.message,
            "correlation_id": self.context.correlation_id,
            "stage": self.context.stage,
            "details": self.context.details,
            "timestamp": self.context.timestamp.isoformat(),
            "cause": str(self.cause) if self.cause else None,
        }

    def log(self, logger: logging.Logger | None = None) -> None:
        """Log exception with structured data (FR-2.6)."""
        target = logger or logging.getLogger(__name__)
        target.error(
            f"[{self.__class__.__name__}] {self.message}",
            extra={"error_data": self.to_dict()},
            exc_info=self.cause,
        )


# ─── User-Facing Errors ───────────────────────────────────────────────────────


class VerityError(VerityBaseException):
    """Base class for user-facing errors."""

    def get_safe_message(self) -> str:
        """Return safe error message for user display (FR-3.3)."""
        return "An error occurred. Please try again."


class InputValidationError(VerityError):
    """Raised when input fails validation (FR-1.1)."""

    def get_safe_message(self) -> str:
        return "Invalid input format. Please check your input."


class ModelError(VerityError):
    """Raised when ML model inference fails (FR-1.5)."""

    def get_safe_message(self) -> str:
        return "Model prediction failed. Please try again."


class APIError(VerityError):
    """Raised when external API call fails (FR-2.4)."""

    def __init__(
        self,
        message: str,
        provider: str | None = None,
        status_code: int | None = None,
        is_rate_limit: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.provider = provider
        self.status_code = status_code
        self.is_rate_limit = is_rate_limit

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data["provider"] = self.provider
        data["status_code"] = self.status_code
        data["is_rate_limit"] = self.is_rate_limit
        return data

    def get_safe_message(self) -> str:
        if self.is_rate_limit:
            return "Service temporarily unavailable due to rate limits. Please wait and try again."
        return "External service error. Please try again later."


class DataError(VerityError):
    """Raised when data loading or processing fails (FR-4.3)."""

    def get_safe_message(self) -> str:
        return "Failed to process data. Please try again."


class PipelineError(VerityError):
    """Raised when pipeline execution fails (FR-4.1)."""

    def get_safe_message(self) -> str:
        return "Analysis pipeline failed. Please try again."


# ─── System Errors ────────────────────────────────────────────────────────────


class VeritySystemError(VerityBaseException):
    """Base class for internal system errors."""

    def get_safe_message(self) -> str:
        """System errors never expose internal details (FR-3.3)."""
        return "An internal error occurred. Please contact support."


class ConfigurationError(VeritySystemError):
    """Raised when configuration is invalid or missing (FR-4.6)."""

    def get_safe_message(self) -> str:
        return "Configuration error. Please check environment setup."


class DatabaseError(VeritySystemError):
    """Raised when database operations fail (FR-4.5)."""

    def get_safe_message(self) -> str:
        return "Database error. Please try again later."


class TimeoutError(VeritySystemError):
    """Raised when operations exceed timeout (NFR-8.4)."""

    def get_safe_message(self) -> str:
        return "Request timed out. Please try again."


# ─── Error Helpers ────────────────────────────────────────────────────────────


def generate_correlation_id() -> str:
    """Generate unique correlation ID for request tracing."""
    return str(uuid.uuid4())[:8]


def safe_execute[T](
    func: Callable[..., T],
    *args: Any,
    default: T | None = None,
    context: ErrorContext | None = None,
    error_type: type[VerityError] = VerityError,
    **kwargs: Any,
) -> Result[T]:
    """
    Execute function with structured error handling (FR-2.5).

    Returns Result[T] instead of raising or returning None.
    """
    correlation_id = generate_correlation_id()

    try:
        result = func(*args, **kwargs)
        return Result.ok(result)
    except VerityError as e:
        e.context.correlation_id = correlation_id
        e.log()
        return Result.fail(e)
    except Exception as e:
        error = error_type(
            message=f"Unexpected error in {func.__name__}: {str(e)}",
            context=context or ErrorContext(
                correlation_id=correlation_id,
                stage=context.stage if context else None,
            ),
            cause=e,
        )
        error.log()
        return Result.fail(error)