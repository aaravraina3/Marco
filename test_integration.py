"""
Integration test to verify all components can be imported and work together.
"""
import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.types import ScrollCommand, SwipeCommand, PalmState, ControllerProto
from src.config import load_config
from src.controller_mock import MockController


async def test_integration():
    """Test that all components can be imported and used together."""
    print("Testing integration of hand gesture recognition components...")
    
    # Test 1: Import and create types
    print("\n1. Testing type definitions...")
    scroll_cmd = ScrollCommand(dy_px=50)
    swipe_cmd = SwipeCommand(direction="left")
    palm_state = PalmState(
        x_px=320.0, 
        y_px=240.0, 
        vx_px_s=10.0, 
        vy_px_s=-25.0, 
        armed_scroll=True, 
        armed_tab_switch=False
    )
    print(f"✓ ScrollCommand: {scroll_cmd}")
    print(f"✓ SwipeCommand: {swipe_cmd}")
    print(f"✓ PalmState: {palm_state}")
    
    # Test 2: Load configuration
    print("\n2. Testing configuration loading...")
    try:
        config = load_config()
        print(f"✓ Config loaded successfully")
        print(f"  Camera: {config.camera.width}x{config.camera.height} @ {config.camera.fps}fps")
        print(f"  MediaPipe: max_hands={config.mediapipe.max_num_hands}")
        print(f"  Display: {config.display.window_name}")
    except Exception as e:
        print(f"✗ Config loading failed: {e}")
        return False
    
    # Test 3: Mock controller
    print("\n3. Testing mock controller...")
    controller = MockController()
    
    # Test scroll command
    await controller.scroll(scroll_cmd.dy_px)
    
    # Test tab switch command
    await controller.switch_tab(swipe_cmd.direction)
    
    # Verify protocol compliance
    assert isinstance(controller, ControllerProto)
    print("✓ MockController implements ControllerProto correctly")
    
    # Test 4: Landmarks (basic import test)
    print("\n4. Testing landmarks module...")
    try:
        from src.landmarks import HandsTracker, palm_center, fingers_extended, is_open_hand, is_index_middle_extended
        
        # Create tracker (without actually using camera)
        tracker = HandsTracker(
            max_num_hands=config.mediapipe.max_num_hands,
            min_detection_conf=config.mediapipe.min_detection_confidence,
            min_tracking_conf=config.mediapipe.min_tracking_confidence
        )
        print("✓ HandsTracker created successfully")
        
        # Test utility functions with dummy landmarks
        dummy_landmarks = [(0.5, 0.5)] * 21  # 21 dummy landmarks
        center = palm_center(dummy_landmarks)
        print(f"✓ Palm center calculation: {center}")
        
    except Exception as e:
        print(f"✗ Landmarks module test failed: {e}")
        return False
    
    print("\n🎉 All integration tests passed!")
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Run the main application: python -m src.main")
    print("3. Point your camera at your hand and press 'q' to quit")
    
    return True


if __name__ == "__main__":
    success = asyncio.run(test_integration())
    sys.exit(0 if success else 1)
