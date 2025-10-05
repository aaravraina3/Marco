"""
Gesture recognition classes that convert hand signals into commands.
"""
import time
from collections import deque
from typing import Optional, Tuple, List
from dataclasses import dataclass

from .types import ScrollCommand, TabSwitchCommand, SwipeCommand, PalmState
from .config import Cfg


@dataclass
class VelocitySample:
    """Sample of velocity and position data at a specific time."""
    timestamp: float
    vx_px_s: float
    vy_px_s: float
    x_px: float
    y_px: float


class ScrollGesture:
    """
    Converts index+middle finger vertical motion into scroll commands.
    
    Features:
    - One-shot swipe detection (not continuous tracking)
    - Directional hysteresis to suppress reset motion
    - Enhanced sensitivity for downward scrolls
    - Hand-lost detection with timeout
    """
    
    def __init__(self, cfg: Cfg):
        """Initialize scroll gesture processor."""
        self.cfg = cfg
        self.last_hand_seen_time: Optional[float] = None
        self.is_scrolling = False
        
        # One-shot gesture detection state
        self.last_gesture_direction: Optional[str] = None  # "up" or "down"
        self.gesture_cooldown_until: float = 0.0
        self.min_gesture_velocity = 80.0  # px/s - minimum velocity for deliberate gesture
        self.gesture_cooldown_ms = 200  # ms between gestures
        self.reset_suppression_factor = 0.3  # suppress opposite direction by this factor
    
    def update(self, landmarks: Optional[List[Tuple[float, float]]], palm: PalmState, 
               t_now: float, frame_wh: Tuple[int, int]) -> Optional[ScrollCommand]:
        """
        Process palm state and return scroll command if conditions are met.
        Uses one-shot gesture detection to avoid jittery reset motion.
        
        Args:
            landmarks: Hand landmarks (None if no hand detected)
            palm: Current palm state with velocity
            t_now: Current timestamp in seconds
            frame_wh: Frame dimensions (width, height)
            
        Returns:
            ScrollCommand if scroll should be performed, None otherwise
        """
        frame_width, frame_height = frame_wh
        
        # Update hand detection state
        if landmarks is not None:
            self.last_hand_seen_time = t_now
            self.is_scrolling = True
        else:
            # Check if we should stop scrolling due to hand loss
            if self.last_hand_seen_time is not None:
                time_since_hand_lost = (t_now - self.last_hand_seen_time) * 1000  # ms
                if time_since_hand_lost > self.cfg.gestures.scroll.stop_on_hand_lost_ms:
                    self.is_scrolling = False
                    self.last_gesture_direction = None  # Reset gesture state
                    return None
        
        # Gate: Must have index+middle armed and be in scrolling state
        if not palm.armed_scroll or not self.is_scrolling:
            self.last_gesture_direction = None  # Reset gesture state
            return None
        
        # Check cooldown
        if t_now < self.gesture_cooldown_until:
            return None
        
        # Determine gesture direction and velocity
        vy = palm.vy_px_s
        abs_vy = abs(vy)
        
        # Must exceed minimum velocity for deliberate gesture
        if abs_vy < self.min_gesture_velocity:
            return None
        
        current_direction = "down" if vy > 0 else "up"
        
        # Apply reset suppression: if this is opposite to last gesture, require higher velocity
        if (self.last_gesture_direction is not None and 
            self.last_gesture_direction != current_direction):
            
            # Suppress reset motion by requiring higher velocity in opposite direction
            suppression_threshold = self.min_gesture_velocity / self.reset_suppression_factor
            if abs_vy < suppression_threshold:
                return None
        
        # Enhanced sensitivity for downward scrolls
        if current_direction == "down":
            # Increase gain for downward motion to make it more sensitive
            enhanced_gain = self.cfg.gestures.scroll.gain * 2.5  # 2.5x more sensitive
            raw_dy = enhanced_gain * vy
        else:
            # Normal gain for upward motion
            raw_dy = self.cfg.gestures.scroll.gain * vy
        
        # Clamp to maximum step size
        max_step = self.cfg.gestures.scroll.max_step_px_per_frame
        dy_px = int(max(-max_step, min(max_step, raw_dy)))
        
        # Don't send zero commands
        if dy_px == 0:
            return None
        
        # Set cooldown and remember direction
        self.gesture_cooldown_until = t_now + (self.gesture_cooldown_ms / 1000.0)
        self.last_gesture_direction = current_direction
        
        return ScrollCommand(dy_px=dy_px)


class TabSwitchGesture:
    """
    Detects horizontal swipe gestures using open palm (all fingers extended).
    
    Features:
    - Triggers when open palm moves strongly left or right
    - Horizontal velocity threshold
    - Vertical/horizontal ratio filtering
    - Minimum travel distance requirement
    - Cooldown between swipes
    """
    
    def __init__(self, cfg: Cfg):
        """Initialize swipe gesture processor."""
        self.cfg = cfg
        self.velocity_buffer: deque[VelocitySample] = deque()
        self.cooldown_until: float = 0.0
        self.swipe_start_x: Optional[float] = None
        self.swipe_start_time: Optional[float] = None
    
    def update(self, landmarks: Optional[List[Tuple[float, float]]], palm: PalmState, 
               t_now: float, frame_wh: Tuple[int, int]) -> Optional[TabSwitchCommand]:
        """
        Process palm state and return swipe command if gesture is detected.
        
        Args:
            landmarks: Hand landmarks (None if no hand detected)
            palm: Current palm state with velocity
            t_now: Current timestamp in seconds
            frame_wh: Frame dimensions (width, height)
            
        Returns:
            SwipeCommand if swipe is detected, None otherwise
        """
        frame_width, frame_height = frame_wh
        
        # Gate: Must have open palm armed and be past cooldown
        if not palm.armed_tab_switch or t_now < self.cooldown_until:
            self._reset_swipe_tracking()
            return None
        
        # Add current sample to buffer if hand is detected
        if landmarks is not None:
            sample = VelocitySample(
                timestamp=t_now,
                vx_px_s=palm.vx_px_s,
                vy_px_s=palm.vy_px_s,
                x_px=palm.x_px,
                y_px=palm.y_px
            )
            self.velocity_buffer.append(sample)
            
            # Initialize swipe tracking
            if self.swipe_start_x is None:
                self.swipe_start_x = palm.x_px
                self.swipe_start_time = t_now
        else:
            # No hand detected, reset tracking
            self._reset_swipe_tracking()
            return None
        
        # Clean old samples from buffer
        window_s = self.cfg.gestures.swipe.window_ms / 1000.0
        cutoff_time = t_now - window_s
        while self.velocity_buffer and self.velocity_buffer[0].timestamp < cutoff_time:
            self.velocity_buffer.popleft()
        
        # Need minimum samples for analysis
        if len(self.velocity_buffer) < 3:
            return None
        
        # Calculate mean velocities over the window
        mean_vx = sum(sample.vx_px_s for sample in self.velocity_buffer) / len(self.velocity_buffer)
        mean_vy = sum(sample.vy_px_s for sample in self.velocity_buffer) / len(self.velocity_buffer)
        
        # Check velocity thresholds
        abs_mean_vx = abs(mean_vx)
        abs_mean_vy = abs(mean_vy)
        
        # Must exceed horizontal velocity threshold
        if abs_mean_vx < self.cfg.gestures.swipe.vx_thresh_px_s:
            return None
        
        # Vertical to horizontal ratio must be below threshold (mostly horizontal)
        if abs_mean_vx > 0:  # Avoid division by zero
            vy_vx_ratio = abs_mean_vy / abs_mean_vx
            if vy_vx_ratio > self.cfg.gestures.swipe.vy_ratio_max:
                return None
        
        # Check minimum travel distance
        if self.swipe_start_x is not None:
            travel_distance = abs(palm.x_px - self.swipe_start_x)
            min_travel = self.cfg.gestures.swipe.min_travel_frac * frame_width
            
            if travel_distance < min_travel:
                return None
        
        # Determine swipe direction
        direction = "right" if mean_vx > 0 else "left"
        
        # Set cooldown
        self.cooldown_until = t_now + (self.cfg.gestures.swipe.cooldown_ms / 1000.0)
        
        # Reset tracking for next swipe
        self._reset_swipe_tracking()
        
        return TabSwitchCommand(direction=direction)
    
    def _reset_swipe_tracking(self):
        """Reset swipe tracking state."""
        self.velocity_buffer.clear()
        self.swipe_start_x = None
        self.swipe_start_time = None


# Backward compatibility alias
SwipeGesture = TabSwitchGesture


class GestureProcessor:
    """
    Main gesture processor that coordinates scroll and swipe detection.
    """
    
    def __init__(self, cfg: Cfg):
        """Initialize gesture processor with configuration."""
        self.cfg = cfg
        self.scroll_gesture = ScrollGesture(cfg)
        self.tab_switch_gesture = TabSwitchGesture(cfg)
        
        # Velocity tracking for palm state
        self.prev_palm_pos: Optional[Tuple[float, float]] = None
        self.prev_timestamp: Optional[float] = None
    
    def process_frame(self, landmarks: Optional[List[Tuple[float, float]]], 
                     t_now: float, frame_wh: Tuple[int, int]) -> Tuple[Optional[ScrollCommand], Optional[TabSwitchCommand], PalmState]:
        """
        Process a frame and return detected commands and palm state.
        
        Args:
            landmarks: Hand landmarks (None if no hand detected)
            t_now: Current timestamp in seconds
            frame_wh: Frame dimensions (width, height)
            
        Returns:
            Tuple of (scroll_command, swipe_command, palm_state)
        """
        from .landmarks import palm_center, is_open_hand, is_index_middle_extended
        
        # Initialize default palm state
        palm_state = PalmState(
            x_px=0.0, y_px=0.0, vx_px_s=0.0, vy_px_s=0.0,
            armed_scroll=False, armed_tab_switch=False
        )
        
        if landmarks is not None:
            # Calculate palm position
            palm_x_norm, palm_y_norm = palm_center(landmarks)
            frame_width, frame_height = frame_wh
            palm_x_px = palm_x_norm * frame_width
            palm_y_px = palm_y_norm * frame_height
            
            # Calculate velocity
            vx_px_s, vy_px_s = 0.0, 0.0
            if self.prev_palm_pos is not None and self.prev_timestamp is not None:
                dt = t_now - self.prev_timestamp
                if dt > 0:
                    dx = palm_x_px - self.prev_palm_pos[0]
                    dy = palm_y_px - self.prev_palm_pos[1]
                    vx_px_s = dx / dt
                    vy_px_s = dy / dt
            
            # Update palm state
            palm_state = PalmState(
                x_px=palm_x_px,
                y_px=palm_y_px,
                vx_px_s=vx_px_s,
                vy_px_s=vy_px_s,
                armed_scroll=is_index_middle_extended(landmarks),
                armed_tab_switch=is_open_hand(landmarks)
            )
            
            # Update tracking
            self.prev_palm_pos = (palm_x_px, palm_y_px)
            self.prev_timestamp = t_now
        else:
            # No hand detected, but keep previous timestamp for velocity calculation
            pass
        
        # Process gestures
        scroll_cmd = self.scroll_gesture.update(landmarks, palm_state, t_now, frame_wh)
        tab_switch_cmd = self.tab_switch_gesture.update(landmarks, palm_state, t_now, frame_wh)
        
        return scroll_cmd, tab_switch_cmd, palm_state
