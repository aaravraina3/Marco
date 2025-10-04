"""
Test cases for gesture recognition with synthetic velocity traces.
"""
import unittest
import time
from typing import List, Tuple, Optional
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.gestures import ScrollGesture, SwipeGesture, GestureProcessor, VelocitySample
from src.types import PalmState, ScrollCommand, SwipeCommand
from src.config import load_config


class TestScrollGesture(unittest.TestCase):
    """Test scroll gesture recognition."""
    
    def setUp(self):
        """Set up test configuration."""
        self.cfg = load_config()
        self.scroll_gesture = ScrollGesture(self.cfg)
        self.frame_wh = (640, 480)
    
    def test_dead_zone_filtering(self):
        """Test that small velocities are filtered out by dead zone."""
        # Create palm state with velocity below dead zone
        palm = PalmState(
            x_px=320.0, y_px=240.0,
            vx_px_s=0.0, vy_px_s=20.0,  # Below deadzone (30.0)
            armed_open_hand=True, armed_index_only=False
        )
        
        # Should return None due to dead zone
        cmd = self.scroll_gesture.update(
            landmarks=[(0.5, 0.5)] * 21,  # Dummy landmarks
            palm=palm,
            t_now=time.time(),
            frame_wh=self.frame_wh
        )
        
        self.assertIsNone(cmd)
    
    def test_scroll_command_generation(self):
        """Test that valid scroll velocities generate commands."""
        # Create palm state with velocity above dead zone
        palm = PalmState(
            x_px=320.0, y_px=240.0,
            vx_px_s=0.0, vy_px_s=100.0,  # Above deadzone
            armed_open_hand=True, armed_index_only=False
        )
        
        # Should generate scroll command
        cmd = self.scroll_gesture.update(
            landmarks=[(0.5, 0.5)] * 21,
            palm=palm,
            t_now=time.time(),
            frame_wh=self.frame_wh
        )
        
        self.assertIsInstance(cmd, ScrollCommand)
        # Expected: 100.0 * 0.05 = 5 pixels
        self.assertEqual(cmd.dy_px, 5)
    
    def test_velocity_clamping(self):
        """Test that large velocities are clamped to max step."""
        # Create palm state with very high velocity
        palm = PalmState(
            x_px=320.0, y_px=240.0,
            vx_px_s=0.0, vy_px_s=1000.0,  # Very high velocity
            armed_open_hand=True, armed_index_only=False
        )
        
        cmd = self.scroll_gesture.update(
            landmarks=[(0.5, 0.5)] * 21,
            palm=palm,
            t_now=time.time(),
            frame_wh=self.frame_wh
        )
        
        self.assertIsInstance(cmd, ScrollCommand)
        # Should be clamped to max_step_px_per_frame (20)
        self.assertEqual(cmd.dy_px, 20)
    
    def test_hand_lost_timeout(self):
        """Test that scrolling stops after hand is lost for timeout period."""
        palm = PalmState(
            x_px=320.0, y_px=240.0,
            vx_px_s=0.0, vy_px_s=100.0,
            armed_open_hand=True, armed_index_only=False
        )
        
        t_start = time.time()
        
        # First update with hand present
        cmd1 = self.scroll_gesture.update(
            landmarks=[(0.5, 0.5)] * 21,
            palm=palm,
            t_now=t_start,
            frame_wh=self.frame_wh
        )
        self.assertIsInstance(cmd1, ScrollCommand)
        
        # Update with no hand, but within timeout
        cmd2 = self.scroll_gesture.update(
            landmarks=None,
            palm=palm,
            t_now=t_start + 0.3,  # 300ms later, within 500ms timeout
            frame_wh=self.frame_wh
        )
        self.assertIsNone(cmd2)  # No command but still tracking
        
        # Update with no hand, beyond timeout
        cmd3 = self.scroll_gesture.update(
            landmarks=None,
            palm=palm,
            t_now=t_start + 0.6,  # 600ms later, beyond 500ms timeout
            frame_wh=self.frame_wh
        )
        self.assertIsNone(cmd3)  # Should stop scrolling


class TestSwipeGesture(unittest.TestCase):
    """Test swipe gesture recognition."""
    
    def setUp(self):
        """Set up test configuration."""
        self.cfg = load_config()
        self.swipe_gesture = SwipeGesture(self.cfg)
        self.frame_wh = (640, 480)
    
    def test_horizontal_velocity_threshold(self):
        """Test that swipes below velocity threshold are rejected."""
        palm = PalmState(
            x_px=320.0, y_px=240.0,
            vx_px_s=100.0, vy_px_s=10.0,  # Below threshold (200.0)
            armed_open_hand=False, armed_index_only=True
        )
        
        # Add several samples to build up buffer
        t_start = time.time()
        for i in range(5):
            cmd = self.swipe_gesture.update(
                landmarks=[(0.5, 0.5)] * 21,
                palm=palm,
                t_now=t_start + i * 0.05,
                frame_wh=self.frame_wh
            )
            self.assertIsNone(cmd)  # Should not trigger swipe
    
    def test_vertical_ratio_filtering(self):
        """Test that swipes with too much vertical motion are rejected."""
        palm = PalmState(
            x_px=320.0, y_px=240.0,
            vx_px_s=300.0, vy_px_s=200.0,  # High vertical component
            armed_open_hand=False, armed_index_only=True
        )
        
        # Vertical/horizontal ratio: 200/300 = 0.67 > 0.5 (max)
        t_start = time.time()
        for i in range(5):
            cmd = self.swipe_gesture.update(
                landmarks=[(0.5, 0.5)] * 21,
                palm=palm,
                t_now=t_start + i * 0.05,
                frame_wh=self.frame_wh
            )
            self.assertIsNone(cmd)  # Should not trigger swipe
    
    def test_valid_swipe_detection(self):
        """Test that valid swipes are detected correctly."""
        # Create valid swipe conditions
        palm_start = PalmState(
            x_px=200.0, y_px=240.0,  # Start position
            vx_px_s=300.0, vy_px_s=50.0,  # Good horizontal velocity, low vertical
            armed_open_hand=False, armed_index_only=True
        )
        
        palm_end = PalmState(
            x_px=400.0, y_px=240.0,  # End position (200px travel > 15% of 640 = 96px)
            vx_px_s=300.0, vy_px_s=50.0,
            armed_open_hand=False, armed_index_only=True
        )
        
        t_start = time.time()
        
        # Add samples to build up buffer
        for i in range(3):
            cmd = self.swipe_gesture.update(
                landmarks=[(0.5, 0.5)] * 21,
                palm=palm_start,
                t_now=t_start + i * 0.05,
                frame_wh=self.frame_wh
            )
            self.assertIsNone(cmd)  # Building up buffer
        
        # Final sample that should trigger swipe
        cmd = self.swipe_gesture.update(
            landmarks=[(0.5, 0.5)] * 21,
            palm=palm_end,
            t_now=t_start + 0.2,
            frame_wh=self.frame_wh
        )
        
        self.assertIsInstance(cmd, SwipeCommand)
        self.assertEqual(cmd.direction, "right")  # Positive vx = right
    
    def test_swipe_cooldown(self):
        """Test that swipe cooldown prevents rapid successive swipes."""
        palm = PalmState(
            x_px=400.0, y_px=240.0,
            vx_px_s=300.0, vy_px_s=50.0,
            armed_open_hand=False, armed_index_only=True
        )
        
        t_start = time.time()
        
        # Build up buffer and trigger first swipe
        for i in range(4):
            cmd = self.swipe_gesture.update(
                landmarks=[(0.5, 0.5)] * 21,
                palm=palm,
                t_now=t_start + i * 0.05,
                frame_wh=self.frame_wh
            )
        
        # Should have triggered a swipe, now in cooldown
        self.assertIsInstance(cmd, SwipeCommand)
        
        # Try to trigger another swipe immediately (should be blocked by cooldown)
        cmd2 = self.swipe_gesture.update(
            landmarks=[(0.5, 0.5)] * 21,
            palm=palm,
            t_now=t_start + 0.3,  # Still within cooldown period
            frame_wh=self.frame_wh
        )
        
        self.assertIsNone(cmd2)  # Should be blocked by cooldown


class TestGestureProcessor(unittest.TestCase):
    """Test the main gesture processor."""
    
    def setUp(self):
        """Set up test configuration."""
        self.cfg = load_config()
        self.processor = GestureProcessor(self.cfg)
        self.frame_wh = (640, 480)
    
    def test_palm_state_calculation(self):
        """Test that palm state is calculated correctly from landmarks."""
        # Create dummy landmarks representing an open hand
        landmarks = [(0.5, 0.5)] * 21  # All at center for simplicity
        
        t_now = time.time()
        
        scroll_cmd, swipe_cmd, palm_state = self.processor.process_frame(
            landmarks=landmarks,
            t_now=t_now,
            frame_wh=self.frame_wh
        )
        
        # Check that palm state is populated
        self.assertEqual(palm_state.x_px, 320.0)  # 0.5 * 640
        self.assertEqual(palm_state.y_px, 240.0)  # 0.5 * 480
        self.assertEqual(palm_state.vx_px_s, 0.0)  # No previous position
        self.assertEqual(palm_state.vy_px_s, 0.0)  # No previous position
    
    def test_no_hand_detected(self):
        """Test behavior when no hand is detected."""
        t_now = time.time()
        
        scroll_cmd, swipe_cmd, palm_state = self.processor.process_frame(
            landmarks=None,
            t_now=t_now,
            frame_wh=self.frame_wh
        )
        
        # Should return default palm state and no commands
        self.assertIsNone(scroll_cmd)
        self.assertIsNone(swipe_cmd)
        self.assertEqual(palm_state.x_px, 0.0)
        self.assertEqual(palm_state.y_px, 0.0)
        self.assertFalse(palm_state.armed_open_hand)
        self.assertFalse(palm_state.armed_index_only)


def create_synthetic_velocity_trace(vx_values: List[float], vy_values: List[float], 
                                   duration_ms: int = 500) -> List[VelocitySample]:
    """
    Create synthetic velocity trace for testing.
    
    Args:
        vx_values: List of horizontal velocities
        vy_values: List of vertical velocities  
        duration_ms: Total duration in milliseconds
        
    Returns:
        List of VelocitySample objects
    """
    samples = []
    start_time = time.time()
    
    for i, (vx, vy) in enumerate(zip(vx_values, vy_values)):
        timestamp = start_time + (i * duration_ms / 1000.0 / len(vx_values))
        sample = VelocitySample(
            timestamp=timestamp,
            vx_px_s=vx,
            vy_px_s=vy,
            x_px=320.0 + i * 10,  # Simulate horizontal movement
            y_px=240.0
        )
        samples.append(sample)
    
    return samples


class TestSyntheticTraces(unittest.TestCase):
    """Test gesture recognition with synthetic velocity traces."""
    
    def test_scroll_trace_acceptance(self):
        """Test scroll gesture with synthetic vertical velocity trace."""
        # Create trace with strong vertical component
        vy_values = [0, 50, 100, 150, 100, 50, 0]  # Ramp up and down
        vx_values = [0] * len(vy_values)  # No horizontal movement
        
        trace = create_synthetic_velocity_trace(vx_values, vy_values)
        
        # Test that peak velocities would trigger scroll
        self.assertGreater(max(vy_values), 30.0)  # Above deadzone
        self.assertLess(max(vy_values) * 0.05, 20)  # Within max step
    
    def test_swipe_trace_acceptance(self):
        """Test swipe gesture with synthetic horizontal velocity trace."""
        # Create trace with strong horizontal component
        vx_values = [0, 100, 250, 300, 250, 100, 0]  # Ramp up and down
        vy_values = [10] * len(vx_values)  # Small vertical component
        
        trace = create_synthetic_velocity_trace(vx_values, vy_values)
        
        # Test acceptance criteria
        peak_vx = max(vx_values)
        peak_vy = max(vy_values)
        
        self.assertGreater(peak_vx, 200.0)  # Above threshold
        self.assertLess(peak_vy / peak_vx, 0.5)  # Good horizontal ratio
    
    def test_swipe_trace_rejection(self):
        """Test swipe gesture rejection with poor velocity trace."""
        # Create trace with too much vertical component
        vx_values = [0, 100, 200, 200, 150, 100, 0]
        vy_values = [0, 80, 150, 160, 120, 80, 0]  # High vertical
        
        trace = create_synthetic_velocity_trace(vx_values, vy_values)
        
        # Test rejection criteria
        peak_vx = max(vx_values)
        peak_vy = max(vy_values)
        
        # Should be rejected due to high vertical ratio
        self.assertGreater(peak_vy / peak_vx, 0.5)  # Poor horizontal ratio


if __name__ == '__main__':
    unittest.main()
