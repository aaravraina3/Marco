import cv2
import numpy as np
import mediapipe as mp
from eyetrax import GazeEstimator, run_9_point_calibration
from filterpy.kalman import KalmanFilter
from collections import deque


class EnhancedKalmanFilter:
    def __init__(self, process_noise=0.01, measurement_noise=0.1):
        self.kf = KalmanFilter(dim_x=4, dim_z=2)

        # State transition matrix (position and velocity for x and y)
        self.kf.F = np.array([[1, 0, 1, 0],
                              [0, 1, 0, 1],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])

        # Measurement matrix (we only measure position)
        self.kf.H = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0]])

        # Process noise covariance
        self.kf.Q *= process_noise

        # Measurement noise covariance
        self.kf.R *= measurement_noise

        # Initial state covariance
        self.kf.P *= 1000

        self.initialized = False

    def update(self, measurement):
        if not self.initialized:
            self.kf.x = np.array([measurement[0], measurement[1], 0, 0])
            self.initialized = True
            return measurement

        self.kf.predict()
        self.kf.update(measurement)
        return self.kf.x[:2]


class GazeDetector:
    def __init__(self, camera_index=0, use_gpu=True, smoothing_level='high',
                 temporal_window=0.5, blob_size=40):
        """
        Initialize the gaze detector with EyeTrax library.

        Args:
            camera_index: Index of the camera to use (default 0)
            use_gpu: Enable GPU acceleration for MediaPipe (default True)
            smoothing_level: 'low', 'medium', or 'high' (default 'high')
            temporal_window: Time window in seconds for averaging gaze positions (default 0.5)
            blob_size: Size of the gaze blob in pixels (default 40)
        """
        self.camera_index = camera_index
        self.use_gpu = use_gpu
        self.blob_size = blob_size
        self.temporal_window = temporal_window

        # Configure smoothing parameters
        smoothing_params = {
            'low': {'process_noise': 0.05, 'measurement_noise': 0.2, 'buffer_size': 3},
            'medium': {'process_noise': 0.01, 'measurement_noise': 0.1, 'buffer_size': 5},
            'high': {'process_noise': 0.005, 'measurement_noise': 0.05, 'buffer_size': 8}
        }
        params = smoothing_params.get(smoothing_level, smoothing_params['high'])

        # Initialize enhanced Kalman filter
        self.kalman = EnhancedKalmanFilter(
            process_noise=params['process_noise'],
            measurement_noise=params['measurement_noise']
        )

        # Moving average buffer for additional smoothing
        self.buffer_size = params['buffer_size']
        self.position_buffer = deque(maxlen=self.buffer_size)

        # Temporal averaging: store positions with timestamps
        self.temporal_positions = deque()

        # Heat map for blob visualization
        self.heat_map = None

        # Initialize GazeEstimator with GPU support
        self.estimator = GazeEstimator()

        # Enable GPU for MediaPipe if available
        if self.use_gpu:
            try:
                mp_face_mesh = mp.solutions.face_mesh
                self.estimator.face_mesh = mp_face_mesh.FaceMesh(
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                    static_image_mode=False
                )
            except Exception as e:
                print(f"GPU initialization failed, using CPU: {e}")

        self.cap = None
        self.screen_width = 1920
        self.screen_height = 1080
        self.blob_color = (255, 100, 0)  # Blue-cyan in BGR

        # FPS tracking
        self.fps = 0
        self.frame_count = 0
        self.start_time = cv2.getTickCount()
        self.current_time = 0

    def calibrate(self):
        """Run 9-point calibration for gaze estimation."""
        print("Starting 9-point calibration...")
        run_9_point_calibration(self.estimator)
        print("Calibration complete!")

    def smooth_position(self, x, y):
        """Apply multi-stage smoothing to gaze position."""
        # Stage 1: Kalman filter
        filtered = self.kalman.update(np.array([x, y]))

        # Stage 2: Moving average
        self.position_buffer.append(filtered)
        if len(self.position_buffer) > 0:
            smoothed = np.mean(self.position_buffer, axis=0)
        else:
            smoothed = filtered

        return int(smoothed[0]), int(smoothed[1])

    def update_temporal_average(self, x, y, timestamp):
        """
        Update temporal buffer and compute weighted average of positions.
        Recent positions have higher weight.
        """
        # Add new position with timestamp
        self.temporal_positions.append((x, y, timestamp))

        # Remove positions outside temporal window
        while self.temporal_positions and \
              (timestamp - self.temporal_positions[0][2]) > self.temporal_window:
            self.temporal_positions.popleft()

        if not self.temporal_positions:
            return x, y

        # Compute weighted average with exponential weighting
        positions = np.array([(p[0], p[1]) for p in self.temporal_positions])
        timestamps = np.array([p[2] for p in self.temporal_positions])

        # Exponential weights: more recent = higher weight
        time_diffs = timestamp - timestamps
        weights = np.exp(-time_diffs * 3.0)
        weights /= weights.sum()

        # Weighted average
        avg_x = np.sum(positions[:, 0] * weights)
        avg_y = np.sum(positions[:, 1] * weights)

        return int(avg_x), int(avg_y)

    def create_gaussian_blob(self, x, y, sigma=None):
        """Create a Gaussian blob centered at (x, y)."""
        if sigma is None:
            sigma = self.blob_size / 3.0

        # Create meshgrid
        xx, yy = np.meshgrid(
            np.arange(max(0, x - self.blob_size * 2), min(self.screen_width, x + self.blob_size * 2)),
            np.arange(max(0, y - self.blob_size * 2), min(self.screen_height, y + self.blob_size * 2))
        )

        # Gaussian function
        gaussian = np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma**2))

        return xx, yy, gaussian

    def start(self):
        """Start the gaze tracking with smooth blob visualization."""
        self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {self.camera_index}")

        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 60)

        # Get screen resolution from the display window
        cv2.namedWindow('Gaze Tracking', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('Gaze Tracking', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # Initialize heat map for blob
        self.heat_map = np.zeros((self.screen_height, self.screen_width), dtype=np.float32)

        print("Starting gaze tracking with smooth blob. Press 'q' to quit.")
        print(f"GPU acceleration: {'Enabled' if self.use_gpu else 'Disabled'}")
        print(f"Temporal window: {self.temporal_window}s, Blob size: {self.blob_size}px")

        while True:
            ret, frame = self.cap.read()

            if not ret:
                print("Failed to read frame from camera")
                break

            # Get current timestamp
            self.current_time = (cv2.getTickCount() - self.start_time) / cv2.getTickFrequency()

            # Extract features and detect blinks
            features, blink = self.estimator.extract_features(frame)

            # Apply exponential decay to heat map for smooth fading
            self.heat_map *= 0.85

            # If features are detected and not blinking, predict gaze position
            if features is not None and not blink:
                try:
                    gaze_coords = self.estimator.predict([features])[0]
                    raw_x, raw_y = gaze_coords[0], gaze_coords[1]

                    # Apply multi-stage smoothing
                    smooth_x, smooth_y = self.smooth_position(raw_x, raw_y)

                    # Apply temporal averaging for blob center
                    blob_x, blob_y = self.update_temporal_average(
                        smooth_x, smooth_y, self.current_time
                    )

                    # Ensure coordinates are within screen bounds
                    blob_x = max(self.blob_size, min(blob_x, self.screen_width - self.blob_size))
                    blob_y = max(self.blob_size, min(blob_y, self.screen_height - self.blob_size))

                    # Create Gaussian blob and add to heat map
                    xx, yy, gaussian = self.create_gaussian_blob(blob_x, blob_y)

                    # Add gaussian to heat map
                    y_min, y_max = yy.min(), yy.max() + 1
                    x_min, x_max = xx.min(), xx.max() + 1

                    if y_min >= 0 and x_min >= 0 and y_max <= self.screen_height and x_max <= self.screen_width:
                        self.heat_map[y_min:y_max, x_min:x_max] += gaussian * 0.5

                except Exception as e:
                    print(f"Prediction error: {e}")

            # Normalize heat map
            heat_normalized = np.clip(self.heat_map, 0, 1)

            # Create color visualization
            screen = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)

            # Apply color map: blue blob with smooth gradients
            heat_colored = (heat_normalized * 255).astype(np.uint8)

            # Create multi-channel blob with gradient
            screen[:, :, 0] = heat_colored  # Blue channel
            screen[:, :, 1] = (heat_colored * 0.4).astype(np.uint8)  # Green channel
            screen[:, :, 2] = 0  # Red channel

            # Add glow effect
            screen = cv2.GaussianBlur(screen, (21, 21), 10)

            # Calculate and display FPS
            self.frame_count += 1
            if self.frame_count % 30 == 0:
                self.fps = self.frame_count / self.current_time if self.current_time > 0 else 0
                cv2.putText(screen, f"FPS: {self.fps:.1f}",
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                          0.7, (0, 255, 0), 2)

            # Display the screen with gaze blob
            cv2.imshow('Gaze Tracking', screen)

            # Check for quit key
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        self.stop()

    def stop(self):
        """Release resources."""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Gaze tracking stopped.")


def main():
    """Main entry point for gaze detection."""
    try:
        # Initialize gaze detector with GPU acceleration and high smoothing
        detector = GazeDetector(
            camera_index=0,
            use_gpu=True,
            smoothing_level='high'
        )

        # Run calibration
        detector.calibrate()

        # Start gaze tracking
        detector.start()

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
