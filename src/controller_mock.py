"""
Mock controller implementation for testing gesture commands.
"""
from typing import Literal
from .types import ControllerProto


class MockController:
    """Mock controller that prints actions instead of executing them."""
    
    def __init__(self):
        """Initialize the mock controller."""
        self.scroll_count = 0
        self.tab_switch_count = 0
    
    async def scroll(self, dy_px: int) -> None:
        """Print scroll command instead of executing it."""
        self.scroll_count += 1
        print(f"[MockController] Scroll: dy_px={dy_px} (call #{self.scroll_count})")
    
    async def switch_tab(self, direction: Literal["left", "right"]) -> None:
        """Print tab switch command instead of executing it."""
        self.tab_switch_count += 1
        print(f"[MockController] Switch tab: direction={direction} (call #{self.tab_switch_count})")
    
    def reset_counters(self) -> None:
        """Reset action counters for testing."""
        self.scroll_count = 0
        self.tab_switch_count = 0
