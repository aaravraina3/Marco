"""
Configuration management for hand gesture recognition system.
"""
import yaml
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class CameraConfig:
    """Camera configuration settings."""
    index: int
    width: int
    height: int
    fps: int


@dataclass
class MediaPipeConfig:
    """MediaPipe Hands configuration settings."""
    max_num_hands: int
    min_detection_confidence: float
    min_tracking_confidence: float


@dataclass
class ScrollConfig:
    """Scroll gesture configuration."""
    vertical_deadzone_px_s: float
    gain: float
    max_step_px_per_frame: int
    stop_on_hand_lost_ms: int


@dataclass
class SwipeConfig:
    """Swipe gesture configuration."""
    window_ms: int
    vx_thresh_px_s: float
    vy_ratio_max: float
    min_travel_frac: float
    cooldown_ms: int


@dataclass
class GesturesConfig:
    """Gesture recognition configuration."""
    scroll: ScrollConfig
    swipe: SwipeConfig


@dataclass
class DisplayConfig:
    """Display configuration settings."""
    show_landmarks: bool
    show_palm_center: bool
    window_name: str


@dataclass
class Cfg:
    """Main configuration class."""
    camera: CameraConfig
    mediapipe: MediaPipeConfig
    gestures: GesturesConfig
    display: DisplayConfig


def load_config(path: Optional[str] = None) -> Cfg:
    """
    Load configuration from YAML file.
    
    Args:
        path: Path to config file. If None, uses config.default.yaml
        
    Returns:
        Configuration object with all settings
    """
    if path is None:
        # Use default config file in project root
        project_root = Path(__file__).parent.parent
        path = project_root / "config.default.yaml"
    
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)
    
    return _dict_to_config(data)


def _dict_to_config(data: Dict[str, Any]) -> Cfg:
    """Convert dictionary to configuration object."""
    camera_data = data['camera']
    camera = CameraConfig(
        index=camera_data['index'],
        width=camera_data['width'],
        height=camera_data['height'],
        fps=camera_data['fps']
    )
    
    mp_data = data['mediapipe']
    mediapipe = MediaPipeConfig(
        max_num_hands=mp_data['max_num_hands'],
        min_detection_confidence=mp_data['min_detection_confidence'],
        min_tracking_confidence=mp_data['min_tracking_confidence']
    )
    
    gestures_data = data['gestures']
    scroll = ScrollConfig(
        vertical_deadzone_px_s=gestures_data['scroll']['vertical_deadzone_px_s'],
        gain=gestures_data['scroll']['gain'],
        max_step_px_per_frame=gestures_data['scroll']['max_step_px_per_frame'],
        stop_on_hand_lost_ms=gestures_data['scroll']['stop_on_hand_lost_ms']
    )
    swipe = SwipeConfig(
        window_ms=gestures_data['swipe']['window_ms'],
        vx_thresh_px_s=gestures_data['swipe']['vx_thresh_px_s'],
        vy_ratio_max=gestures_data['swipe']['vy_ratio_max'],
        min_travel_frac=gestures_data['swipe']['min_travel_frac'],
        cooldown_ms=gestures_data['swipe']['cooldown_ms']
    )
    gestures = GesturesConfig(scroll=scroll, swipe=swipe)
    
    display_data = data['display']
    display = DisplayConfig(
        show_landmarks=display_data['show_landmarks'],
        show_palm_center=display_data['show_palm_center'],
        window_name=display_data['window_name']
    )
    
    return Cfg(
        camera=camera,
        mediapipe=mediapipe,
        gestures=gestures,
        display=display
    )
