"""
Shared design-rule vocabulary for the CAD package.

Each part module raises its own exception type so a traceback names the part
that failed, but they all report the same :class:`DesignStatus` values. That
keeps the failure vocabulary consistent across parts, in the same way
``IKStatus`` and ``SourceStatus`` do for their own modules.
"""

from __future__ import annotations

from enum import Enum, auto

__all__ = ["DesignStatus", "DesignRuleError"]


class DesignStatus(Enum):
    """Structured reasons a parameter set cannot be built."""

    OK = auto()
    """All clearances satisfied."""

    NEGATIVE_HEIGHT = auto()
    """A height budget leaves no room for the part."""

    WALL_TOO_THIN = auto()
    """A feature sits closer to the outer surface than min_wall_thickness_mm."""

    FEATURE_COLLISION = auto()
    """Two internal features intersect that must not."""

    FASTENER_TOO_SHORT = auto()
    """The specified screw cannot span the assembled stack."""

    INVALID_PARAMETER = auto()
    """A directly-supplied parameter is out of range."""


class DesignRuleError(ValueError):
    """
    Base class for every design rule violation in the CAD package.

    Attributes
    ----------
    status:
        The :class:`DesignStatus` naming the violated constraint.
    """

    def __init__(self, status: DesignStatus, message: str) -> None:
        super().__init__(f"[{status.name}] {message}")
        self.status = status
