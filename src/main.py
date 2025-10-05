"""
Main application for hand gesture recognition.
"""
import cv2
import asyncio
import time
from typing import Optional
from .config import load_config
from .landmarks import HandsTracker, palm_center, is_open_hand, is_index_middle_extended, fingers_extended
from .controller_mock import MockController
from .gestures import GestureProcessor

try:
    from .controller_browser import BrowserController
    BROWSER_CONTROLLER_AVAILABLE = True
except ImportError:
    BROWSER_CONTROLLER_AVAILABLE = False 


class GestureRecognitionApp:
    """Main application class for hand gesture recognition."""
    
    def __init__(self, config_path: Optional[str] = None, use_browser: bool = False):
        """Initialize the application with configuration."""
        self.config = load_config(config_path)
        self.tracker = HandsTracker(
            max_num_hands=self.config.mediapipe.max_num_hands,
            min_detection_conf=self.config.mediapipe.min_detection_confidence,
            min_tracking_conf=self.config.mediapipe.min_tracking_confidence
        )
        
        # Choose controller type
        if use_browser and BROWSER_CONTROLLER_AVAILABLE:
            self.controller = BrowserController()
            print("🌐 Using browser controller - make sure browser server is running!")
        else:
            self.controller = MockController()
            if use_browser and not BROWSER_CONTROLLER_AVAILABLE:
                print("⚠️  Browser controller not available, using mock controller")
        
        #Initialize gesture processor
        self.gesture_processor = GestureProcessor(self.config)


        # Initialize camera
        self.cap = cv2.VideoCapture(self.config.camera.index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.config.camera.fps)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {self.config.camera.index}")
    
    async def run(self):
        """Run the main application loop."""
        print(f"Starting {self.config.display.window_name}")
        print("🎯 Gesture Recognition:")
        print("  - Index+Middle + Vertical Motion = Scroll")
        print("  - Open Palm + Horizontal Swipe = Tab Switch")
        print("Press 'q' to quit")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to read frame from camera")
                break
            
            # Process hand landmarks
            landmarks = self.tracker.process(frame)
            
            # Process gestures and get commands
            t_now = time.time()
            frame_wh = (frame.shape[1], frame.shape[0])  # (width, height)
            
            scroll_cmd, swipe_cmd, palm_state = self.gesture_processor.process_frame(
                landmarks=landmarks,
                t_now=t_now,
                frame_wh=frame_wh
            )
            
            # Execute commands
            if scroll_cmd:
                await self.controller.scroll(scroll_cmd.dy_px)
            
            if swipe_cmd:
                await self.controller.switch_tab(swipe_cmd.direction)
            
            # Initialize status
            status_text = "No hand detected"
            gesture_status = ""
            velocity_info = ""
            
            if landmarks:
                # Draw landmarks if enabled
                if self.config.display.show_landmarks:
                    frame = self.tracker.draw_landmarks(frame, landmarks)
                
                # Draw palm center if enabled
                if self.config.display.show_palm_center:
                    palm_x, palm_y = palm_center(landmarks)
                    height, width = frame.shape[:2]
                    palm_px = int(palm_x * width)
                    palm_py = int(palm_y * height)
                    
                    # Draw a larger red dot for palm center
                    cv2.circle(frame, (palm_px, palm_py), 8, (0, 0, 255), -1)
                    cv2.putText(frame, "Palm", (palm_px + 10, palm_py - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # Update status text with palm state info
                fingers_count = fingers_extended(landmarks)
                status_text = f"Hand: {fingers_count} fingers"
                
                # Show velocity information
                velocity_info = f"Vel: vx={palm_state.vx_px_s:.1f}, vy={palm_state.vy_px_s:.1f}"
                
                # Determine gesture status
                if palm_state.armed_scroll:
                    gesture_status = "✅ INDEX+MIDDLE - Scroll Ready"
                elif palm_state.armed_tab_switch:
                    gesture_status = "✅ OPEN PALM - Tab Switch Ready"
                else:
                    gesture_status = "❌ No recognized gesture"
                
                # Show active commands
                if scroll_cmd:
                    gesture_status += f" | SCROLLING: {scroll_cmd.dy_px}px"
                if swipe_cmd:
                    gesture_status += f" | SWIPED: {swipe_cmd.direction}"
            
            # Draw status on frame
            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, velocity_info, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, gesture_status, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                       (0, 255, 0) if "✅" in gesture_status else (0, 0, 255), 2)
            
            # Add instructions
            cv2.putText(frame, "Index+Middle + Move = Scroll", (10, frame.shape[0] - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, "Open Palm + Swipe = Tab Switch", (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display the frame
            cv2.imshow(self.config.display.window_name, frame)
            
            # Check for quit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()


async def main():
    """Entry point for the application."""
    import sys
    
    # Check for --browser flag
    use_browser = "--browser" in sys.argv
    
    if use_browser:
        print("🌐 Browser mode enabled - make sure to start browser server first:")
        print("   python browser_server.py")
        print()
    
    try:
        app = GestureRecognitionApp(use_browser=use_browser)
        await app.run()
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
        # Clean up browser controller if used
        if hasattr(app, 'controller') and hasattr(app.controller, 'close'):
            await app.controller.close()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
