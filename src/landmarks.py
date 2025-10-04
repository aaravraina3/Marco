"""
Hand landmark detection and gesture recognition using MediaPipe.
"""
import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, List, Tuple


class HandsTracker:
    """Hand landmark tracker using MediaPipe Hands."""
    
    def __init__(self, max_num_hands: int = 1, min_detection_conf: float = 0.6, min_tracking_conf: float = 0.6):
        """
        Initialize the hands tracker.
        
        Args:
            max_num_hands: Maximum number of hands to detect
            min_detection_conf: Minimum confidence for hand detection
            min_tracking_conf: Minimum confidence for hand tracking
        """
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_conf,
            min_tracking_confidence=min_tracking_conf
        )
        self.mp_drawing = mp.solutions.drawing_utils
    
    def process(self, frame_bgr: np.ndarray) -> Optional[List[Tuple[float, float]]]:
        """
        Process a frame and return hand landmarks.
        
        Args:
            frame_bgr: Input frame in BGR format
            
        Returns:
            List of 21 (x, y) coordinates in [0..1] range, or None if no hand detected
        """
        # Convert BGR to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            # Return landmarks for the first detected hand
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks = []
            
            for landmark in hand_landmarks.landmark:
                landmarks.append((landmark.x, landmark.y))
            
            return landmarks
        
        return None
    
    def draw_landmarks(self, frame: np.ndarray, landmarks: List[Tuple[float, float]]) -> np.ndarray:
        """
        Draw hand landmarks on the frame.
        
        Args:
            frame: Input frame
            landmarks: List of (x, y) coordinates in [0..1] range
            
        Returns:
            Frame with landmarks drawn
        """
        height, width = frame.shape[:2]
        
        # Convert normalized coordinates to pixel coordinates and draw
        for i, (x, y) in enumerate(landmarks):
            px = int(x * width)
            py = int(y * height)
            cv2.circle(frame, (px, py), 3, (0, 255, 0), -1)
            cv2.putText(frame, str(i), (px + 5, py - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        return frame


def palm_center(landmarks: List[Tuple[float, float]]) -> Tuple[float, float]:
    """
    Calculate the center of the palm.
    
    Args:
        landmarks: List of 21 hand landmarks
        
    Returns:
        (x, y) coordinates of palm center in [0..1] range
    """
    # Palm landmarks: wrist (0), thumb base (5), index base (9), middle base (13), pinky base (17)
    palm_indices = [0, 5, 9, 13, 17]
    
    x_sum = sum(landmarks[i][0] for i in palm_indices)
    y_sum = sum(landmarks[i][1] for i in palm_indices)
    
    return (x_sum / len(palm_indices), y_sum / len(palm_indices))


def fingers_extended(landmarks: List[Tuple[float, float]]) -> int:
    """
    Count the number of extended fingers.
    
    Args:
        landmarks: List of 21 hand landmarks
        
    Returns:
        Number of extended fingers (0-5)
    """
    # Finger tip and PIP joint indices
    # Thumb: tip=4, pip=3 (special case - use x coordinate)
    # Index: tip=8, pip=6
    # Middle: tip=12, pip=10
    # Ring: tip=16, pip=14
    # Pinky: tip=20, pip=18
    
    extended_count = 0
    
    # Check thumb (horizontal extension)
    if landmarks[4][0] > landmarks[3][0]:  # thumb tip x > thumb pip x
        extended_count += 1
    
    # Check other fingers (vertical extension)
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]
    
    for tip_idx, pip_idx in zip(finger_tips, finger_pips):
        if landmarks[tip_idx][1] < landmarks[pip_idx][1]:  # tip y < pip y (inverted y-axis)
            extended_count += 1
    
    return extended_count


def is_open_hand(landmarks: List[Tuple[float, float]]) -> bool:
    """
    Check if the hand is in an open palm gesture (all fingers extended).
    
    Args:
        landmarks: List of 21 hand landmarks
        
    Returns:
        True if all fingers are extended
    """
    return fingers_extended(landmarks) >= 5


def is_index_middle_extended(landmarks: List[Tuple[float, float]]) -> bool:
    """
    Check if index and middle fingers are extended (for scroll gesture).
    Thumb can also be extended, but ring + pinky must be curled.
    
    Args:
        landmarks: List of 21 hand landmarks
        
    Returns:
        True if index and middle are extended, ring and pinky are curled
    """
    # Check finger extensions
    thumb_extended = landmarks[4][0] > landmarks[3][0]  # thumb tip x > thumb pip x
    index_extended = landmarks[8][1] < landmarks[6][1]  # index tip y < pip y
    middle_extended = landmarks[12][1] < landmarks[10][1]  # middle tip y < pip y
    ring_extended = landmarks[16][1] < landmarks[14][1]  # ring tip y < pip y
    pinky_extended = landmarks[20][1] < landmarks[18][1]  # pinky tip y < pip y
    
    # Index and middle must be extended
    # Ring and pinky must NOT be extended
    # Thumb can be either way
    return (index_extended and middle_extended and 
            not ring_extended and not pinky_extended)


def is_index_only(landmarks: List[Tuple[float, float]]) -> bool:
    """
    Check if only the index finger is extended.
    
    Args:
        landmarks: List of 21 hand landmarks
        
    Returns:
        True if only index finger is extended
    """
    # Check if index finger is extended
    index_extended = landmarks[8][1] < landmarks[6][1]  # tip y < pip y
    
    # Check if other fingers are NOT extended
    thumb_extended = landmarks[4][0] > landmarks[3][0]
    middle_extended = landmarks[12][1] < landmarks[10][1]
    ring_extended = landmarks[16][1] < landmarks[14][1]
    pinky_extended = landmarks[20][1] < landmarks[18][1]
    
    return index_extended and not (thumb_extended or middle_extended or ring_extended or pinky_extended)
