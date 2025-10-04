"""
Hand Gesture Recognition System

A Python service that reads webcam frames, detects hand landmarks using MediaPipe,
and recognizes gestures for scroll and tab switch commands.
"""

__version__ = "0.1.0"
__author__ = "Healthcare Intake Assistant Team"

from .types import ScrollCommand, SwipeCommand, PalmState, ControllerProto
from .config import load_config, Cfg
from .controller_mock import MockController
from .landmarks import HandsTracker, palm_center, fingers_extended, is_open_hand, is_index_only

__all__ = [
    "ScrollCommand",
    "SwipeCommand", 
    "PalmState",
    "ControllerProto",
    "load_config",
    "Cfg",
    "MockController",
    "HandsTracker",
    "palm_center",
    "fingers_extended",
    "is_open_hand",
    "is_index_only",
]
