"""Domain exceptions used across the active VAAET pipeline."""

from __future__ import annotations


class VAAETError(Exception):
    """Base class for domain-specific runtime errors."""


class ArtifactNotFoundError(FileNotFoundError, VAAETError):
    """Raised when a required file or external artifact is missing."""


class VideoValidationError(ValueError, VAAETError):
    """Raised when a video cannot be validated or inspected safely."""


class VideoOpenError(RuntimeError, VAAETError):
    """Raised when OpenCV cannot open a video."""


class DatabaseNotConfiguredError(RuntimeError, VAAETError):
    """Raised when optional database settings are missing."""


class DatabaseOperationError(RuntimeError, VAAETError):
    """Raised when a database operation fails in a non-recoverable way."""
