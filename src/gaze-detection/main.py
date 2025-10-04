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
VERTICAL_SENSITIVITY = 2.75
HORIZONTAL_SENSITIVITY = 1.5
VERTICAL_OFFSET = -550
HORIZONTAL_OFFSET = -600
SMOOTHING_FACTOR = 0.75 #Higher value = more smoothing, less responsive; lower value = less smoothing, more responsive (0.0 - 1.0)

prev_gaze_x = None #Previous Gaze X Coordinate
prev_gaze_y = None #Previous Gaze Y Coordinate

def midpoint(p1, p2):
    return ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2) #Calculate the midpoint of two points

def euclidean_dist(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5 #Calculate the Euclidean distance between two points

def eye_aspect_ratio(top, bottom, horizontal):
    vertical_dist = euclidean_dist(top, bottom)
    horizontal_dist = euclidean_dist(horizontal[0], horizontal[1])
    return vertical_dist / horizontal_dist if horizontal_dist > 0 else 0 #Calculate the eye aspect ratio

def main():
    global prev_gaze_x, prev_gaze_y
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

        is_blinking = "Not Blinking"

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w = frame.shape[:2] #Get the height and width of the frame (height and width of the frame)

                left_iris_coords = [(int(face_landmarks.landmark[i].x * w),
                                    int(face_landmarks.landmark[i].y * h)) for i in LEFT_IRIS] #Get the coordinates of the left iris
                right_iris_coords = [(int(face_landmarks.landmark[i].x * w),
                                     int(face_landmarks.landmark[i].y * h)) for i in RIGHT_IRIS] #Get the coordinates of the right iris

                for coord in left_iris_coords + right_iris_coords:
                    cv2.circle(frame, coord, 2, (0, 0, 255), -1) #Draw circles on the iris landmarks

                left_iris_center = midpoint(left_iris_coords[0], left_iris_coords[2])
                right_iris_center = midpoint(right_iris_coords[0], right_iris_coords[2])

                cv2.circle(frame, left_iris_center, 3, (0, 255, 0), -1) #Draw circles on the iris center
                cv2.circle(frame, right_iris_center, 3, (0, 255, 0), -1) #Draw circles on the iris center

                left_eye_left = (int(face_landmarks.landmark[LEFT_EYE[0]].x * w),
                                int(face_landmarks.landmark[LEFT_EYE[0]].y * h)) #Get the coordinates of the left eye
                left_eye_right = (int(face_landmarks.landmark[LEFT_EYE[1]].x * w),
                                 int(face_landmarks.landmark[LEFT_EYE[1]].y * h)) #Get the coordinates of the left eye
                left_eye_top = (int(face_landmarks.landmark[LEFT_EYE_TOP].x * w),
                               int(face_landmarks.landmark[LEFT_EYE_TOP].y * h)) #Get the coordinates of the left eye
                left_eye_bottom = (int(face_landmarks.landmark[LEFT_EYE_BOTTOM].x * w),
                                  int(face_landmarks.landmark[LEFT_EYE_BOTTOM].y * h)) #Get the coordinates of the left eye

                right_eye_left = (int(face_landmarks.landmark[RIGHT_EYE[1]].x * w),
                                 int(face_landmarks.landmark[RIGHT_EYE[1]].y * h)) #Get the coordinates of the right eye
                right_eye_right = (int(face_landmarks.landmark[RIGHT_EYE[0]].x * w),
                                  int(face_landmarks.landmark[RIGHT_EYE[0]].y * h)) #Get the coordinates of the right eye       
                right_eye_top = (int(face_landmarks.landmark[RIGHT_EYE_TOP].x * w),
                                int(face_landmarks.landmark[RIGHT_EYE_TOP].y * h)) #Get the coordinates of the right eye
                right_eye_bottom = (int(face_landmarks.landmark[RIGHT_EYE_BOTTOM].x * w),
                                   int(face_landmarks.landmark[RIGHT_EYE_BOTTOM].y * h)) #Get the coordinates of the right eye

                left_ear = eye_aspect_ratio(left_eye_top, left_eye_bottom, [left_eye_left, left_eye_right]) #Calculate the eye aspect ratio of the left eye
                right_ear = eye_aspect_ratio(right_eye_top, right_eye_bottom, [right_eye_left, right_eye_right]) #Calculate the eye aspect ratio of the right eye
                avg_ear = (left_ear + right_ear) / 2

                if avg_ear < BLINK_THRESHOLD: #If the eye aspect ratio is less than the threshold, the eye is blinking
                    is_blinking = "Blinking"

                left_eye_width = euclidean_dist(left_eye_left, left_eye_right) #Calculate the width of the left eye
                right_eye_width = euclidean_dist(right_eye_left, right_eye_right) #Calculate the width of the right eye

                left_iris_x_ratio = (left_iris_center[0] - left_eye_left[0]) / left_eye_width if left_eye_width > 0 else 0.5 #Calculate the ratio of the left iris to the left eye
                left_iris_y_ratio = (left_iris_center[1] - left_eye_top[1]) / euclidean_dist(left_eye_top, left_eye_bottom) if euclidean_dist(left_eye_top, left_eye_bottom) > 0 else 0.5

                right_iris_x_ratio = (right_iris_center[0] - right_eye_left[0]) / right_eye_width if right_eye_width > 0 else 0.5 #Calculate the ratio of the right iris to the right eye
                right_iris_y_ratio = (right_iris_center[1] - right_eye_top[1]) / euclidean_dist(right_eye_top, right_eye_bottom) if euclidean_dist(right_eye_top, right_eye_bottom) > 0 else 0.5 #Calculate the ratio of the right iris to the right eye

                avg_x_ratio = (left_iris_x_ratio + right_iris_x_ratio) / 2 #Calculate the average ratio of the iris to the eye
                avg_y_ratio = (left_iris_y_ratio + right_iris_y_ratio) / 2 #Calculate the average ratio of the iris to the eye

                gaze_x = int((1 - avg_x_ratio) * w * HORIZONTAL_SENSITIVITY) + HORIZONTAL_OFFSET #Calculate the gaze x coordinate
                gaze_y = int((1 - avg_y_ratio) * h * VERTICAL_SENSITIVITY) + VERTICAL_OFFSET #Calculate the gaze y coordinate

                if prev_gaze_x is not None and prev_gaze_y is not None:
                    gaze_x = int(prev_gaze_x * SMOOTHING_FACTOR + gaze_x * (1 - SMOOTHING_FACTOR)) #Smooth the gaze x coordinate
                    gaze_y = int(prev_gaze_y * SMOOTHING_FACTOR + gaze_y * (1 - SMOOTHING_FACTOR)) #Smooth the gaze y coordinate

                prev_gaze_x = gaze_x #Update the previous gaze x coordinate
                prev_gaze_y = gaze_y #Update the previous gaze y coordinate

                gaze_x = max(0, min(gaze_x, w - 1)) #Clip the gaze x coordinate to the width of the frame
                gaze_y = max(0, min(gaze_y, h - 1)) #Clip the gaze y coordinate to the height of the frame

                cv2.circle(frame, (gaze_x, gaze_y), 5, (255, 0, 0), -1) #Draw a circle on the gaze point

        cv2.putText(frame, is_blinking, (w - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) #Display the blinking status

        cv2.imshow("Frame", frame) #Display the frame
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release() #Release the video camera
    cv2.destroyAllWindows()

if __name__ == "__main__": #Main function
    main()