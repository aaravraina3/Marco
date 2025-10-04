from collections import OrderedDict
import numpy as np
import cv2
import json
import base64

def data_uri_to_cv2_img(uri):
    encoded_data = uri.split(',')[1]
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

#costruct the argument parser and parse tha arguments
#ap=argparse.ArgumentParser()
#ap.add_argument("-p","--shape-predictor", required=True,
 #                   help="path to facial landmark predictor")
#ap.add_argument("-i", "--image", required=True,
#                    help="path to input image")
#args=vars(ap.parse_args())
#shape_predictor = args["shape_predictor"]
#imageP = args["image"]
imageP = "D:\GithubRepo\FaCiPa-FacialDetec68coordinates\face.jpg"

FACIAL_LANDMARKS_IDXS = OrderedDict([
	("mouth", (48, 68)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
	("nose", (27, 36)),
	("jaw", (0, 17))
])

def rect_to_bb(rect):
        # take a bounding predicted by dlib and convert it
        # to the format (x, y, w, h) as we would normally do
        # with OpenCV
        x = rect.left()
        y = rect.top()
        w = rect.right() - x
        h = rect.bottom() - y
        # return a tuple of (x, y, w, h)
        return (x, y, w, h)



def shape_to_np(shape, dtype="int"):
	coords = np.zeros((68, 2), dtype=dtype)
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	# return the list of (x, y)-coordinates
	return coords

def analyze_facial_paralysis_mediapipe(face_landmarks, frame_width, frame_height):
    """
    Analyze facial paralysis using MediaPipe face landmarks
    Based on FaCiPa algorithm - measures asymmetries between left and right facial features
    """
    durum = dict()
    durum["status"] = False  # status=False
    
    # MediaPipe face mesh landmark indices (mapped to 68-point dlib landmarks)
    # These correspond to the key points used in the original FaCiPa algorithm
    
    # Eyebrow landmarks (converted to MediaPipe indices)
    eyebrow_right_inner = 63    # Right eyebrow inner (was 21 in dlib)
    eyebrow_right_outer = 17    # Right eyebrow outer (was 17 in dlib) 
    eyebrow_left_inner = 105    # Left eyebrow inner (was 22 in dlib)
    eyebrow_left_outer = 66     # Left eyebrow outer (was 26 in dlib)
    
    # Eye landmarks
    eye_right_inner = 133       # Right eye inner (was 39 in dlib)
    eye_right_outer = 33        # Right eye outer (was 36 in dlib)
    eye_left_inner = 362        # Left eye inner (was 42 in dlib)
    eye_left_outer = 263        # Left eye outer (was 45 in dlib)
    
    # Nose landmarks
    nose_right = 174            # Right nostril (was 31 in dlib)
    nose_left = 397             # Left nostril (was 35 in dlib)
    
    # Mouth landmarks
    mouth_right = 61            # Right mouth corner (was 48 in dlib)
    mouth_left = 291            # Left mouth corner (was 54 in dlib)
    
    # Get Y coordinates for asymmetry analysis
    eyebrowri_y = face_landmarks.landmark[eyebrow_right_inner].y * frame_height
    eyebrowrj_y = face_landmarks.landmark[eyebrow_right_outer].y * frame_height
    eyebrowli_y = face_landmarks.landmark[eyebrow_left_inner].y * frame_height
    eyebrowlj_y = face_landmarks.landmark[eyebrow_left_outer].y * frame_height
    
    eyeri_y = face_landmarks.landmark[eye_right_inner].y * frame_height
    eyerj_y = face_landmarks.landmark[eye_right_outer].y * frame_height
    eyeli_y = face_landmarks.landmark[eye_left_inner].y * frame_height
    eyelj_y = face_landmarks.landmark[eye_left_outer].y * frame_height
    
    noser_y = face_landmarks.landmark[nose_right].y * frame_height
    nosel_y = face_landmarks.landmark[nose_left].y * frame_height
    
    lipr_y = face_landmarks.landmark[mouth_right].y * frame_height
    lipl_y = face_landmarks.landmark[mouth_left].y * frame_height
    
    # Calculate asymmetries (same logic as original FaCiPa)
    farkeyebrowi = abs(eyebrowri_y - eyebrowli_y)  # eyebrow asymmetry (inner)
    farkeyebrowj = abs(eyebrowrj_y - eyebrowlj_y)  # eyebrow asymmetry (outer)
    farkeyei = abs(eyeri_y - eyeli_y)              # eye asymmetry (inner)
    farkeyej = abs(eyerj_y - eyelj_y)              # eye asymmetry (outer)
    farknose = abs(nosel_y - noser_y)              # nose asymmetry
    farklip = abs(lipl_y - lipr_y)                 # mouth asymmetry
    
    # Count asymmetries - MORE CONSERVATIVE THRESHOLDS for stable normal detection
    # Higher thresholds to reduce false positives while maintaining paralysis detection
    sum = 0
    if farkeyebrowi >= 30:  # Further increased for eyebrow sensitivity
        sum += 1
    if farkeyebrowj >= 30:  # Further increased for eyebrow sensitivity  
        sum += 1
    if farkeyei >= 25:      # Increased for eye sensitivity
        sum += 1
    if farkeyej >= 25:      # Increased for eye sensitivity
        sum += 1
    if farknose >= 25:      # Increased for nose sensitivity
        sum += 1
    if farklip >= 30:       # Increased for mouth sensitivity
        sum += 1
    
    # Detect paralysis if 4 or more asymmetries (more conservative threshold)
    if sum >= 4:
        durum["status"] = True
    
    # Add detailed results
    durum["asymmetries"] = {
        "eyebrow_inner": farkeyebrowi,
        "eyebrow_outer": farkeyebrowj,
        "eye_inner": farkeyei,
        "eye_outer": farkeyej,
        "nose": farknose,
        "mouth": farklip
    }
    durum["total_asymmetries"] = sum
    durum["confidence"] = min(1.0, sum / 6.0)  # Confidence based on number of asymmetries
    
    print(f"Facial Paralysis Analysis: {'DETECTED' if durum['status'] else 'NORMAL'} (asymmetries: {sum}/6)")
    
    return durum


if __name__ == "__main__":
    durum = resim_analiz(imageP)
    with open("durum.json", "w") as f:
        json.dump(durum, f)

