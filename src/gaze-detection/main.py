import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh #Add Face Mesh Model
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) #Face Mesh Model

cap = cv2.VideoCapture(0) # Initialize the Video Camera

LEFT_IRIS = [474, 475, 476, 477] #Track Left Iris Landmarks, Boundary point of the iris
RIGHT_IRIS = [469, 470, 471, 472]
LEFT_EYE = [33, 133] #Track Left Eye Landmarks, Edge of the eye
RIGHT_EYE = [362, 263] #Track Right Eye Landmarks, Edge of the eye
LEFT_EYE_TOP = 159 #Track Left Eye Top Landmark, Top of the eye
LEFT_EYE_BOTTOM = 145 #Add Left Eye Bottom Landmark
RIGHT_EYE_TOP = 386 #Add Right Eye Top Landmark
RIGHT_EYE_BOTTOM = 374 #Add Right Eye Bottom Landmark

#Manual Gaze Detection Parameter Tuning, I Just kept tweaking these values until it looked decent
BLINK_THRESHOLD = 0.2

# Distance detection parameters
DISTANCE_HISTORY = []
HISTORY_SIZE = 5
SUPER_CLOSE_THRESHOLD = 310.0  # Super close to screen
CLOSE_THRESHOLD = 240.0        # Slightly close to screen
FAR_THRESHOLD = 150.0          # Slightly far from screen
SUPER_FAR_THRESHOLD = 120.0    # Super far from screen
OPTIMAL_DISTANCE = 200.0       # Your default optimal distance

def midpoint(p1, p2):
    return ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2) #Calculate the midpoint of two points

def euclidean_dist(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5 #Calculate the Euclidean distance between two points

def eye_aspect_ratio(top, bottom, horizontal):
    vertical_dist = euclidean_dist(top, bottom)
    horizontal_dist = euclidean_dist(horizontal[0], horizontal[1])
    return vertical_dist / horizontal_dist if horizontal_dist > 0 else 0 #Calculate the eye aspect ratio

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

def main():
    screen_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) #Get the width of the screen (width of the frame)
    screen_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) #Get the height of the screen (height of the frame)

    while True:
        ret, frame = cap.read() #Read the frame from the camera
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #Convert the frame from BGR to RGB to normalize the color space
        frame.flags.writeable = False
        results = face_mesh.process(frame)
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) #Convert the frame from RGB to BGR to display the frame
        frame = cv2.flip(frame, 1) #Flip the frame horizontally to fix inverted camera

        h, w = frame.shape[:2] #Get the height and width of the frame
        is_blinking = "Not Blinking"
        distance_status = "NO FACE"
        eye_distance = 0.0

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:

                # Get coordinates before flipping (for MediaPipe processing)
                left_iris_coords = [(int(face_landmarks.landmark[i].x * w),
                                    int(face_landmarks.landmark[i].y * h)) for i in LEFT_IRIS] #Get the coordinates of the left iris
                right_iris_coords = [(int(face_landmarks.landmark[i].x * w),
                                     int(face_landmarks.landmark[i].y * h)) for i in RIGHT_IRIS] #Get the coordinates of the right iris

                # Flip coordinates to match the flipped frame
                left_iris_coords = [(w - coord[0], coord[1]) for coord in left_iris_coords]
                right_iris_coords = [(w - coord[0], coord[1]) for coord in right_iris_coords]

                for coord in left_iris_coords + right_iris_coords:
                    cv2.circle(frame, coord, 2, (0, 0, 255), -1) #Draw circles on the iris landmarks

                left_iris_center = midpoint(left_iris_coords[0], left_iris_coords[2])
                right_iris_center = midpoint(right_iris_coords[0], right_iris_coords[2])

                cv2.circle(frame, left_iris_center, 3, (0, 255, 0), -1) #Draw circles on the iris center
                cv2.circle(frame, right_iris_center, 3, (0, 255, 0), -1) #Draw circles on the iris center

                # Get eye coordinates and flip them to match the flipped frame
                left_eye_left = (w - int(face_landmarks.landmark[LEFT_EYE[0]].x * w),
                                int(face_landmarks.landmark[LEFT_EYE[0]].y * h)) #Get the coordinates of the left eye
                left_eye_right = (w - int(face_landmarks.landmark[LEFT_EYE[1]].x * w),
                                 int(face_landmarks.landmark[LEFT_EYE[1]].y * h)) #Get the coordinates of the left eye
                left_eye_top = (w - int(face_landmarks.landmark[LEFT_EYE_TOP].x * w),
                               int(face_landmarks.landmark[LEFT_EYE_TOP].y * h)) #Get the coordinates of the left eye
                left_eye_bottom = (w - int(face_landmarks.landmark[LEFT_EYE_BOTTOM].x * w),
                                  int(face_landmarks.landmark[LEFT_EYE_BOTTOM].y * h)) #Get the coordinates of the left eye

                right_eye_left = (w - int(face_landmarks.landmark[RIGHT_EYE[1]].x * w),
                                 int(face_landmarks.landmark[RIGHT_EYE[1]].y * h)) #Get the coordinates of the right eye
                right_eye_right = (w - int(face_landmarks.landmark[RIGHT_EYE[0]].x * w),
                                  int(face_landmarks.landmark[RIGHT_EYE[0]].y * h)) #Get the coordinates of the right eye       
                right_eye_top = (w - int(face_landmarks.landmark[RIGHT_EYE_TOP].x * w),
                                int(face_landmarks.landmark[RIGHT_EYE_TOP].y * h)) #Get the coordinates of the right eye
                right_eye_bottom = (w - int(face_landmarks.landmark[RIGHT_EYE_BOTTOM].x * w),
                                   int(face_landmarks.landmark[RIGHT_EYE_BOTTOM].y * h)) #Get the coordinates of the right eye

                left_ear = eye_aspect_ratio(left_eye_top, left_eye_bottom, [left_eye_left, left_eye_right]) #Calculate the eye aspect ratio of the left eye
                right_ear = eye_aspect_ratio(right_eye_top, right_eye_bottom, [right_eye_left, right_eye_right]) #Calculate the eye aspect ratio of the right eye
                avg_ear = (left_ear + right_ear) / 2

                if avg_ear < BLINK_THRESHOLD: #If the eye aspect ratio is less than the threshold, the eye is blinking
                    is_blinking = "Blinking"

                # Calculate distance between eyes
                left_eye_coords = [left_eye_left, left_eye_right]
                right_eye_coords = [right_eye_left, right_eye_right]
                eye_distance = calculate_eye_distance(left_eye_coords, right_eye_coords)
                distance_status, smoothed_distance = detect_distance(eye_distance)

        # Draw status text
        cv2.putText(frame, is_blinking, (w - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) #Display the blinking status
        
        # Draw distance information with color coding
        if distance_status == "OPTIMAL":
            distance_color = (0, 255, 0)  # Green
        elif distance_status in ["SLIGHTLY CLOSE", "SLIGHTLY FAR"]:
            distance_color = (0, 255, 255)  # Yellow
        else:  # SUPER CLOSE, SUPER FAR
            distance_color = (0, 0, 255)  # Red
            
        cv2.putText(frame, f"Distance: {distance_status}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, distance_color, 2)
        cv2.putText(frame, f"Eye Dist: {smoothed_distance:.1f}px", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw frame info
        cv2.putText(frame, f"Frame: {w}x{h}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Press 'q' to quit", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2) #Display instructions

        cv2.imshow("Healthcare Eye Tracker", frame) #Display the frame
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release() #Release the video camera
    cv2.destroyAllWindows()

if __name__ == "__main__": #Main function
    main()