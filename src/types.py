"""
Type definitions for hand gesture recognition system.
"""
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable


@dataclass
class ScrollCommand:
    """Command to scroll by a pixel delta."""
    dy_px: int


@dataclass
class TabSwitchCommand:
    """Command to switch tabs in a direction."""
    direction: Literal["left", "right"]


# Keep SwipeCommand for backward compatibility
SwipeCommand = TabSwitchCommand


@dataclass
class PalmState:
    """Current state of palm tracking and gesture recognition."""
    x_px: float
    y_px: float
    vx_px_s: float  # velocity in pixels per second
    vy_px_s: float  # velocity in pixels per second
    armed_scroll: bool  # index + middle extended
    armed_tab_switch: bool  # open palm (all fingers extended)


@runtime_checkable
class ControllerProto(Protocol):
    """Abstract protocol for controllers that execute gesture commands."""
    
    async def scroll(self, dy_px: int) -> None:
        """Execute a scroll command with the given pixel delta."""
        ...
    
    async def switch_tab(self, direction: Literal["left", "right"]) -> None:
        """Switch tabs in the specified direction."""
        ...
