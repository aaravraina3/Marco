import cv2
import mediapipe as mp
import numpy as np
from face_mesh_analyzer import FaceMeshAnalyzer, FaceAnalysisResult
from facial_landmarks import analyze_facial_paralysis_mediapipe
from tremor_analyzer import create_tremor_analyzer

mp_face_mesh = mp.solutions.face_mesh  # Add Face Mesh Model
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)  # Face Mesh Model

cap = cv2.VideoCapture(0)  # Initialize the Video Camera
cap.set(cv2.CAP_PROP_ZOOM, 1.0)  # Initialize zoom level

# Initialize Face Mesh Analyzer
face_analyzer = FaceMeshAnalyzer()
is_face_analyzer_calibrated = False

LEFT_IRIS = [
    474,
    475,
    476,
    477,
]  # Track Left Iris Landmarks, Boundary point of the iris
RIGHT_IRIS = [469, 470, 471, 472]
LEFT_EYE = [33, 133]  # Track Left Eye Landmarks, Edge of the eye
RIGHT_EYE = [362, 263]  # Track Right Eye Landmarks, Edge of the eye
LEFT_EYE_TOP = 159  # Track Left Eye Top Landmark, Top of the eye
LEFT_EYE_BOTTOM = 145  # Add Left Eye Bottom Landmark
RIGHT_EYE_TOP = 386  # Add Right Eye Top Landmark
RIGHT_EYE_BOTTOM = 374  # Add Right Eye Bottom Landmark

# Manual Gaze Detection Parameter Tuning, I Just kept tweaking these values until it looked decent
BLINK_THRESHOLD = 0.2

# Distance detection parameters
DISTANCE_HISTORY = []
HISTORY_SIZE = 5
SUPER_CLOSE_THRESHOLD = 310.0  # Super close to screen
CLOSE_THRESHOLD = 240.0  # Slightly close to screen
FAR_THRESHOLD = 150.0  # Slightly far from screen
SUPER_FAR_THRESHOLD = 120.0  # Super far from screen
OPTIMAL_DISTANCE = 200.0  # Your default optimal distance

# Zoom control parameters
current_zoom = 1.0
zoom_step = 0.1
min_zoom = 0.5
max_zoom = 3.0
zoom_history = []
ZOOM_HISTORY_SIZE = 10

# Simple fixed thresholds
BLINK_THRESHOLD = 0.25
SEVERE_SQUINT_THRESHOLD = 0.28
SQUINT_THRESHOLD = 0.32
NORMAL_THRESHOLD = 0.35

# Squinting detection parameters - Dynamic thresholds based on calibration
ear_history = []
EAR_HISTORY_SIZE = 15
squint_delay_frames = 0
SQUINT_DELAY_THRESHOLD = 8  # Frames to wait before confirming squinting
current_squint_status = "NORMAL"

# Dynamic thresholds (will be set after calibration)
BLINK_THRESHOLD = 0.25  # Eye closure detection
SEVERE_SQUINT_THRESHOLD = 0.28  # Severe squinting
SQUINT_THRESHOLD = 0.32  # Light squinting
NORMAL_THRESHOLD = 0.35  # Normal eye openness


def midpoint(p1, p2):
    return (
        (p1[0] + p2[0]) // 2,
        (p1[1] + p2[1]) // 2,
    )  # Calculate the midpoint of two points


def euclidean_dist(p1, p2):
    return (
        (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2
    ) ** 0.5  # Calculate the Euclidean distance between two points


def eye_aspect_ratio(top, bottom, horizontal):
    vertical_dist = euclidean_dist(top, bottom)
    horizontal_dist = euclidean_dist(horizontal[0], horizontal[1])
    return (
        vertical_dist / horizontal_dist if horizontal_dist > 0 else 0
    )  # Calculate the eye aspect ratio


def calculate_eye_distance(left_eye_coords, right_eye_coords):
    """Calculate distance between eye centers"""
    left_center = midpoint(left_eye_coords[0], left_eye_coords[1])
    right_center = midpoint(right_eye_coords[0], right_eye_coords[1])
    return euclidean_dist(left_center, right_center)


def detect_distance(eye_distance):
    """Detect if person is close, far, or optimal distance"""
    global DISTANCE_HISTORY

    # Add to history for smoothing
    DISTANCE_HISTORY.append(eye_distance)
    if len(DISTANCE_HISTORY) > HISTORY_SIZE:
        DISTANCE_HISTORY.pop(0)

    # Use smoothed distance
    smoothed_distance = sum(DISTANCE_HISTORY) / len(DISTANCE_HISTORY)

    if smoothed_distance > SUPER_CLOSE_THRESHOLD:
        return "SUPER CLOSE", smoothed_distance
    elif smoothed_distance > CLOSE_THRESHOLD:
        return "SLIGHTLY CLOSE", smoothed_distance
    elif smoothed_distance < SUPER_FAR_THRESHOLD:
        return "SUPER FAR", smoothed_distance
    elif smoothed_distance < FAR_THRESHOLD:
        return "SLIGHTLY FAR", smoothed_distance
    else:
        return "OPTIMAL", smoothed_distance


def adjust_camera_zoom(distance_status, smoothed_distance):
    """Adjust camera zoom based on distance detection"""
    global current_zoom, zoom_history

    # Calculate target zoom based on distance
    target_zoom = current_zoom

    if distance_status == "SUPER CLOSE":
        target_zoom = current_zoom - zoom_step * 2  # Zoom out more aggressively
    elif distance_status == "SLIGHTLY CLOSE":
        target_zoom = current_zoom - zoom_step  # Zoom out slightly
    elif distance_status == "SLIGHTLY FAR":
        target_zoom = current_zoom + zoom_step  # Zoom in slightly
    elif distance_status == "SUPER FAR":
        target_zoom = current_zoom + zoom_step * 2  # Zoom in more aggressively
    else:  # OPTIMAL
        # Gradually return to normal zoom
        if current_zoom > 1.0:
            target_zoom = current_zoom - zoom_step * 0.5
        elif current_zoom < 1.0:
            target_zoom = current_zoom + zoom_step * 0.5

    # Clamp zoom to valid range
    target_zoom = max(min_zoom, min(max_zoom, target_zoom))

    # Add to zoom history for smoothing
    zoom_history.append(target_zoom)
    if len(zoom_history) > ZOOM_HISTORY_SIZE:
        zoom_history.pop(0)

    # Use smoothed zoom
    smoothed_zoom = sum(zoom_history) / len(zoom_history)

    # Apply zoom if it changed significantly
    if abs(smoothed_zoom - current_zoom) > 0.05:
        current_zoom = smoothed_zoom

        # Try different zoom methods
        zoom_applied = False

        # Method 1: Try CAP_PROP_ZOOM
        if cap.set(cv2.CAP_PROP_ZOOM, current_zoom):
            zoom_applied = True
            print(f"Applied zoom via CAP_PROP_ZOOM: {current_zoom}")

        # Method 2: Try CAP_PROP_FOCUS (some cameras use this)
        elif cap.set(cv2.CAP_PROP_FOCUS, int(current_zoom * 100)):
            zoom_applied = True
            print(f"Applied zoom via CAP_PROP_FOCUS: {current_zoom}")

        # Method 3: Try CAP_PROP_BRIGHTNESS as a proxy (some cameras)
        elif cap.set(cv2.CAP_PROP_BRIGHTNESS, int(current_zoom * 50)):
            zoom_applied = True
            print(f"Applied zoom via CAP_PROP_BRIGHTNESS: {current_zoom}")

        if not zoom_applied:
            print(
                f"Zoom change detected but camera doesn't support zoom control: {current_zoom}"
            )

    return current_zoom


def calculate_eye_aspect_ratio(face_landmarks, frame_width, frame_height):
    """Calculate Eye Aspect Ratio (EAR) using proper eye contour landmarks"""
    # Get eye contour landmarks (not iris landmarks)
    # Left eye: 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246
    # Right eye: 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398

    # Left eye landmarks
    left_eye_landmarks = [
        33,
        7,
        163,
        144,
        145,
        153,
        154,
        155,
        133,
        173,
        157,
        158,
        159,
        160,
        161,
        246,
    ]
    # Right eye landmarks
    right_eye_landmarks = [
        362,
        382,
        381,
        380,
        374,
        373,
        390,
        249,
        263,
        466,
        388,
        387,
        386,
        385,
        384,
        398,
    ]

    # Convert to pixel coordinates
    left_eye_coords = [
        (
            int(face_landmarks.landmark[i].x * frame_width),
            int(face_landmarks.landmark[i].y * frame_height),
        )
        for i in left_eye_landmarks
    ]
    right_eye_coords = [
        (
            int(face_landmarks.landmark[i].x * frame_width),
            int(face_landmarks.landmark[i].y * frame_height),
        )
        for i in right_eye_landmarks
    ]

    # Calculate EAR using 6-point method
    # Use landmarks: 0=left corner, 1=top, 2=right corner, 3=bottom, 4=top-left, 5=top-right

    # Left eye EAR
    left_vertical_1 = euclidean_dist(
        left_eye_coords[1], left_eye_coords[3]
    )  # Top to bottom
    left_vertical_2 = euclidean_dist(
        left_eye_coords[4], left_eye_coords[5]
    )  # Top-left to top-right
    left_horizontal = euclidean_dist(
        left_eye_coords[0], left_eye_coords[2]
    )  # Left to right corner
    left_ear = (
        (left_vertical_1 + left_vertical_2) / (2.0 * left_horizontal)
        if left_horizontal > 0
        else 0.35
    )

    # Right eye EAR
    right_vertical_1 = euclidean_dist(
        right_eye_coords[1], right_eye_coords[3]
    )  # Top to bottom
    right_vertical_2 = euclidean_dist(
        right_eye_coords[4], right_eye_coords[5]
    )  # Top-left to top-right
    right_horizontal = euclidean_dist(
        right_eye_coords[0], right_eye_coords[2]
    )  # Left to right corner
    right_ear = (
        (right_vertical_1 + right_vertical_2) / (2.0 * right_horizontal)
        if right_horizontal > 0
        else 0.35
    )

    # Average EAR
    avg_ear = (left_ear + right_ear) / 2.0

    return avg_ear


def detect_squinting(ear_value):
    """Detect squinting using simple fixed thresholds"""
    global ear_history, squint_delay_frames, current_squint_status

    # Add EAR to history for smoothing
    ear_history.append(ear_value)
    if len(ear_history) > EAR_HISTORY_SIZE:
        ear_history.pop(0)

    # Use smoothed EAR for more stable detection
    smoothed_ear = sum(ear_history) / len(ear_history)

    # Determine immediate squinting level using fixed thresholds
    if smoothed_ear < BLINK_THRESHOLD:
        immediate_status = "BLINKING"
    elif smoothed_ear < SEVERE_SQUINT_THRESHOLD:
        immediate_status = "SEVERE SQUINTING"
    elif smoothed_ear < SQUINT_THRESHOLD:
        immediate_status = "SQUINTING"
    else:
        immediate_status = "NORMAL"

    # Apply delay logic for squinting (but not blinking)
    if immediate_status == "BLINKING":
        # Blinking is immediate - no delay needed
        current_squint_status = "BLINKING"
        squint_delay_frames = 0
    elif immediate_status in ["SQUINTING", "SEVERE SQUINTING"]:
        # Squinting requires delay to confirm (distinguish from quick blinks)
        if current_squint_status == "NORMAL":
            squint_delay_frames += 1
            if squint_delay_frames >= SQUINT_DELAY_THRESHOLD:
                current_squint_status = immediate_status
                squint_delay_frames = 0
        else:
            # Already squinting, update immediately
            current_squint_status = immediate_status
    else:  # NORMAL
        # Reset delay when returning to normal
        squint_delay_frames = 0
        current_squint_status = "NORMAL"

    return current_squint_status, smoothed_ear


def debug_camera_properties():
    """Debug function to check what camera properties are available"""
    print("=== Camera Properties Debug ===")
    zoom_prop = cap.get(cv2.CAP_PROP_ZOOM)
    focus_prop = cap.get(cv2.CAP_PROP_FOCUS)
    brightness_prop = cap.get(cv2.CAP_PROP_BRIGHTNESS)

    print(f"CAP_PROP_ZOOM: {zoom_prop}")
    print(f"CAP_PROP_FOCUS: {focus_prop}")
    print(f"CAP_PROP_BRIGHTNESS: {brightness_prop}")
    print("===============================")


def main():
    # Debug camera properties first
    debug_camera_properties()

    screen_width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )  # Get the width of the screen (width of the frame)
    screen_height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )  # Get the height of the screen (height of the frame)
    
    # Initialize face analyzer calibration flag
    is_face_analyzer_calibrated = False
    
    # Initialize tremor analyzer (resting tremor by default)
    tremor_analyzer = create_tremor_analyzer("resting")

    while True:
        ret, frame = cap.read()  # Read the frame from the camera
        if not ret:
            break

        frame = cv2.cvtColor(
            frame, cv2.COLOR_BGR2RGB
        )  # Convert the frame from BGR to RGB to normalize the color space
        frame.flags.writeable = False
        results = face_mesh.process(frame)
        frame.flags.writeable = True
        frame = cv2.cvtColor(
            frame, cv2.COLOR_RGB2BGR
        )  # Convert the frame from RGB to BGR to display the frame
        frame = cv2.flip(frame, 1)  # Flip the frame horizontally to fix inverted camera

        h, w = frame.shape[:2]  # Get the height and width of the frame
        is_blinking = "Not Blinking"
        distance_status = "NO FACE"
        eye_distance = 0.0
        smoothed_distance = 0.0
        current_zoom = 1.0
        squint_status = "NO FACE"
        smoothed_ear = 0.0
        # Initialize variables for display
        avg_ear = 0.0

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:

                # Get coordinates before flipping (for MediaPipe processing)
                left_iris_coords = [
                    (
                        int(face_landmarks.landmark[i].x * w),
                        int(face_landmarks.landmark[i].y * h),
                    )
                    for i in LEFT_IRIS
                ]  # Get the coordinates of the left iris
                right_iris_coords = [
                    (
                        int(face_landmarks.landmark[i].x * w),
                        int(face_landmarks.landmark[i].y * h),
                    )
                    for i in RIGHT_IRIS
                ]  # Get the coordinates of the right iris

                # Flip coordinates to match the flipped frame
                left_iris_coords = [
                    (w - coord[0], coord[1]) for coord in left_iris_coords
                ]
                right_iris_coords = [
                    (w - coord[0], coord[1]) for coord in right_iris_coords
                ]

                for coord in left_iris_coords + right_iris_coords:
                    cv2.circle(
                        frame, coord, 2, (0, 0, 255), -1
                    )  # Draw circles on the iris landmarks

                left_iris_center = midpoint(left_iris_coords[0], left_iris_coords[2])
                right_iris_center = midpoint(right_iris_coords[0], right_iris_coords[2])

                cv2.circle(
                    frame, left_iris_center, 3, (0, 255, 0), -1
                )  # Draw circles on the iris center
                cv2.circle(
                    frame, right_iris_center, 3, (0, 255, 0), -1
                )  # Draw circles on the iris center

                # Get eye coordinates and flip them to match the flipped frame
                left_eye_left = (
                    w - int(face_landmarks.landmark[LEFT_EYE[0]].x * w),
                    int(face_landmarks.landmark[LEFT_EYE[0]].y * h),
                )  # Get the coordinates of the left eye
                left_eye_right = (
                    w - int(face_landmarks.landmark[LEFT_EYE[1]].x * w),
                    int(face_landmarks.landmark[LEFT_EYE[1]].y * h),
                )  # Get the coordinates of the left eye
                left_eye_top = (
                    w - int(face_landmarks.landmark[LEFT_EYE_TOP].x * w),
                    int(face_landmarks.landmark[LEFT_EYE_TOP].y * h),
                )  # Get the coordinates of the left eye
                left_eye_bottom = (
                    w - int(face_landmarks.landmark[LEFT_EYE_BOTTOM].x * w),
                    int(face_landmarks.landmark[LEFT_EYE_BOTTOM].y * h),
                )  # Get the coordinates of the left eye

                right_eye_left = (
                    w - int(face_landmarks.landmark[RIGHT_EYE[1]].x * w),
                    int(face_landmarks.landmark[RIGHT_EYE[1]].y * h),
                )  # Get the coordinates of the right eye
                right_eye_right = (
                    w - int(face_landmarks.landmark[RIGHT_EYE[0]].x * w),
                    int(face_landmarks.landmark[RIGHT_EYE[0]].y * h),
                )  # Get the coordinates of the right eye
                right_eye_top = (
                    w - int(face_landmarks.landmark[RIGHT_EYE_TOP].x * w),
                    int(face_landmarks.landmark[RIGHT_EYE_TOP].y * h),
                )  # Get the coordinates of the right eye
                right_eye_bottom = (
                    w - int(face_landmarks.landmark[RIGHT_EYE_BOTTOM].x * w),
                    int(face_landmarks.landmark[RIGHT_EYE_BOTTOM].y * h),
                )  # Get the coordinates of the right eye

                left_ear = eye_aspect_ratio(
                    left_eye_top, left_eye_bottom, [left_eye_left, left_eye_right]
                )  # Calculate the eye aspect ratio of the left eye
                right_ear = eye_aspect_ratio(
                    right_eye_top, right_eye_bottom, [right_eye_left, right_eye_right]
                )  # Calculate the eye aspect ratio of the right eye
                avg_ear = (left_ear + right_ear) / 2

                if (
                    avg_ear < BLINK_THRESHOLD
                ):  # If the eye aspect ratio is less than the threshold, the eye is blinking
                    is_blinking = "Blinking"

                # Calculate distance between eyes
                left_eye_coords = [left_eye_left, left_eye_right]
                right_eye_coords = [right_eye_left, right_eye_right]
                eye_distance = calculate_eye_distance(left_eye_coords, right_eye_coords)
                distance_status, smoothed_distance = detect_distance(eye_distance)

                # Calculate comprehensive EAR for squinting detection
                left_eye_full_coords = [
                    left_eye_left,
                    left_eye_top,
                    left_eye_right,
                    left_eye_bottom,
                ]
                right_eye_full_coords = [
                    right_eye_left,
                    right_eye_top,
                    right_eye_right,
                    right_eye_bottom,
                ]
                comprehensive_ear = calculate_eye_aspect_ratio(face_landmarks, w, h)

                # Detect squinting using research-based EAR approach
                squint_status, smoothed_ear = detect_squinting(comprehensive_ear)

                # Update blinking status based on squinting detection
                if squint_status == "BLINKING":
                    is_blinking = "Blinking"

                # Face Mesh Analysis
                if not is_face_analyzer_calibrated:
                    # Calibrate face analyzer on first face detection
                    if face_analyzer.calibrate(face_landmarks, w, h):
                        is_face_analyzer_calibrated = True
                        print("Face analyzer calibrated successfully")
                
                # Analyze facial features (temporarily disabled due to division by zero error)
                # face_analysis_result = face_analyzer.analyze_face(face_landmarks, w, h, frame)
                # smoothed_face_result = face_analyzer.get_smoothed_result()
                
                # FaCiPa Facial Paralysis Analysis using MediaPipe
                paralysis_result = analyze_facial_paralysis_mediapipe(face_landmarks, w, h)
                
                # Update tremor analyzer with face distance for depth estimation
                tremor_analyzer.update_hand_depth_estimate(smoothed_distance, smoothed_distance * 0.1)  # Rough conversion
                
                # Tremor Analysis
                tremor_result = tremor_analyzer.analyze_tremor(frame)

                # Adjust camera zoom based on distance
                current_zoom_level = adjust_camera_zoom(
                    distance_status, smoothed_distance
                )

        # Create a clean left-side info panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 280), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Draw distance information with color coding - BIGGER
        if distance_status == "OPTIMAL":
            distance_color = (0, 255, 0)  # Green
        elif distance_status in ["SLIGHTLY CLOSE", "SLIGHTLY FAR"]:
            distance_color = (0, 255, 255)  # Yellow
        else:  # SUPER CLOSE, SUPER FAR
            distance_color = (0, 0, 255)  # Red

        cv2.putText(frame, "Healthcare Eye Tracker", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Distance: {distance_status}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, distance_color, 2)
        cv2.putText(frame, f"Eye Distance: {smoothed_distance:.1f}px", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Zoom: {current_zoom:.2f}x", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Draw squinting information with color coding - BIGGER
        if squint_status == "NORMAL":
            squint_color = (0, 255, 0)  # Green
        elif squint_status == "SQUINTING":
            squint_color = (0, 255, 255)  # Yellow
        elif squint_status == "SEVERE SQUINTING":
            squint_color = (0, 0, 255)  # Red
        else:  # BLINKING
            squint_color = (255, 0, 255)  # Magenta

        cv2.putText(frame, f"Eye Status: {squint_status}", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, squint_color, 2)
        cv2.putText(frame, f"EAR: {smoothed_ear:.3f}", (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Instructions - BIGGER
        cv2.putText(frame, "Press 'q' to quit", (20, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Draw face analysis overlay if calibrated (temporarily disabled)
        # if 'is_face_analyzer_calibrated' in locals() and is_face_analyzer_calibrated and 'smoothed_face_result' in locals():
        #     face_analyzer.draw_analysis_overlay(frame, smoothed_face_result)
        
        # Draw FaCiPa paralysis analysis overlay - BIGGER AND CLEANER
        if 'paralysis_result' in locals():
            paralysis_color = (0, 0, 255) if paralysis_result['status'] else (0, 255, 0)
            paralysis_text = "PARALYSIS DETECTED" if paralysis_result['status'] else "NORMAL"
            
            # Create a semi-transparent background box - STRETCHED DOWN
            overlay = frame.copy()
            cv2.rectangle(overlay, (w - 450, 50), (w - 10, 400), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # Main status - BIG AND BOLD
            cv2.putText(frame, f"FaCiPa Analysis:", 
                       (w - 440, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(frame, paralysis_text, 
                       (w - 440, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.2, paralysis_color, 3)
            
            # Asymmetries count - BIGGER
            cv2.putText(frame, f"Asymmetries: {paralysis_result['total_asymmetries']}/6", 
                       (w - 440, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Confidence - BIGGER
            confidence = paralysis_result.get('confidence', 0.0)
            cv2.putText(frame, f"Confidence: {confidence:.2f}", 
                       (w - 440, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Detailed asymmetries - BIGGER AND SPREAD OUT with updated thresholds
            if 'asymmetries' in paralysis_result:
                y_offset = 250
                # Define thresholds for each feature (updated to match new conservative values)
                thresholds = {
                    "eyebrow_inner": 30, "eyebrow_outer": 30, 
                    "eye_inner": 25, "eye_outer": 25, 
                    "nose": 25, "mouth": 30
                }
                for feature, value in paralysis_result['asymmetries'].items():
                    threshold = thresholds.get(feature, 20)
                    color = (0, 0, 255) if value >= threshold else (0, 255, 0)
                    cv2.putText(frame, f"{feature}: {value:.1f}px", 
                               (w - 440, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    y_offset += 25

        # Draw Tremor Analysis overlay
        if 'tremor_result' in locals() and tremor_result['hand_detected']:
            tremor_color = (0, 0, 255) if tremor_result['tremor_detected'] else (0, 255, 0)
            tremor_text = tremor_analyzer.get_tremor_status_text()
            
            # Create tremor analysis panel on the right side
            tremor_overlay = frame.copy()
            cv2.rectangle(tremor_overlay, (w - 450, 420), (w - 10, 600), (0, 0, 0), -1)
            cv2.addWeighted(tremor_overlay, 0.7, frame, 0.3, 0, frame)
            
            # Tremor status - BIG AND BOLD
            cv2.putText(frame, "Tremor Analysis:", 
                       (w - 440, 460), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(frame, tremor_text, 
                       (w - 440, 500), cv2.FONT_HERSHEY_SIMPLEX, 1.0, tremor_color, 2)
            
            # Tremor details
            cv2.putText(frame, f"Amplitude: {tremor_result['amplitude_cm']:.2f}cm", 
                       (w - 440, 540), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"UPDRS Rating: {tremor_result['updrs_rating']}/4", 
                       (w - 440, 570), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Draw hand landmarks if detected
            if tremor_result['landmarks']:
                frame = tremor_analyzer.draw_hand_landmarks(frame, tremor_result['landmarks'])

        cv2.imshow("Healthcare Eye Tracker", frame)  # Display the frame
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()  # Release the video camera
    cv2.destroyAllWindows()


if __name__ == "__main__":  # Main function
    main()
