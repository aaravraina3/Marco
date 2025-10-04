"""
Feature Extractors - Utility class for extracting facial measurements from MediaPipe landmarks
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


class FeatureExtractors:
    """
    Utility class for extracting facial feature measurements from MediaPipe landmarks
    """
    
    def __init__(self):
        """Initialize the feature extractors"""
        # MediaPipe face mesh landmark indices for key facial features
        self.landmark_indices = {
            # Forehead landmarks
            'forehead_top': 10,
            'forehead_center': 9,
            'forehead_left': 21,
            'forehead_right': 251,
            
            # Eyebrow landmarks
            'left_eyebrow_outer': 70,
            'left_eyebrow_inner': 55,
            'left_eyebrow_top': 63,
            'right_eyebrow_outer': 300,
            'right_eyebrow_inner': 285,
            'right_eyebrow_top': 293,
            
            # Eye landmarks
            'left_eye_outer': 33,
            'left_eye_inner': 133,
            'left_eye_top': 159,
            'left_eye_bottom': 145,
            'right_eye_outer': 362,
            'right_eye_inner': 263,
            'right_eye_top': 386,
            'right_eye_bottom': 374,
            
            # Nose landmarks
            'nose_tip': 1,
            'nose_bridge': 6,
            'nose_left': 31,
            'nose_right': 35,
            
            # Mouth landmarks
            'mouth_left': 61,
            'mouth_right': 291,
            'mouth_top': 13,
            'mouth_bottom': 14,
            'mouth_center': 0,
            
            # Face contour landmarks
            'chin': 18,
            'jaw_left': 172,
            'jaw_right': 397,
            'cheek_left': 116,
            'cheek_right': 345,
            
            # Additional reference points
            'temple_left': 162,
            'temple_right': 389,
            'ear_left': 234,
            'ear_right': 454
        }
    
    def extract_baseline_measurements(self, face_landmarks, frame_width: int, frame_height: int) -> Dict:
        """
        Extract baseline measurements for calibration
        
        Args:
            face_landmarks: MediaPipe face landmarks
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
            
        Returns:
            Dict: Baseline measurements
        """
        measurements = {}
        
        # Convert landmarks to pixel coordinates
        landmarks_2d = self._landmarks_to_2d(face_landmarks, frame_width, frame_height)
        
        # Forehead measurements
        measurements['forehead_height'] = self._calculate_forehead_height(landmarks_2d)
        measurements['forehead_width'] = self._calculate_forehead_width(landmarks_2d)
        
        # Eyebrow measurements
        measurements['eyebrow_height'] = self._calculate_eyebrow_height(landmarks_2d)
        measurements['eyebrow_width'] = self._calculate_eyebrow_width(landmarks_2d)
        
        # Eye measurements
        measurements['eye_height'] = self._calculate_eye_height(landmarks_2d)
        measurements['eye_width'] = self._calculate_eye_width(landmarks_2d)
        
        # Mouth measurements
        measurements['mouth_width'] = self._calculate_mouth_width(landmarks_2d)
        measurements['mouth_height'] = self._calculate_mouth_height(landmarks_2d)
        
        # Face measurements
        measurements['face_width'] = self._calculate_face_width(landmarks_2d)
        measurements['face_height'] = self._calculate_face_height(landmarks_2d)
        
        # Head pose measurements
        measurements['head_tilt'] = self._calculate_head_tilt(landmarks_2d)
        measurements['head_turn'] = self._calculate_head_turn(landmarks_2d)
        measurements['head_nod'] = self._calculate_head_nod(landmarks_2d)
        
        return measurements
    
    def extract_current_measurements(self, face_landmarks, frame_width: int, frame_height: int) -> Dict:
        """
        Extract current measurements for analysis
        
        Args:
            face_landmarks: MediaPipe face landmarks
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
            
        Returns:
            Dict: Current measurements
        """
        # Same as baseline for now, but could be extended for real-time adjustments
        return self.extract_baseline_measurements(face_landmarks, frame_width, frame_height)
    
    def _landmarks_to_2d(self, face_landmarks, frame_width: int, frame_height: int) -> Dict:
        """
        Convert MediaPipe landmarks to 2D pixel coordinates
        
        Args:
            face_landmarks: MediaPipe face landmarks
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
            
        Returns:
            Dict: 2D coordinates for each landmark
        """
        landmarks_2d = {}
        
        for name, index in self.landmark_indices.items():
            if index < len(face_landmarks.landmark):
                landmark = face_landmarks.landmark[index]
                landmarks_2d[name] = (
                    int(landmark.x * frame_width),
                    int(landmark.y * frame_height)
                )
        
        return landmarks_2d
    
    def _calculate_forehead_height(self, landmarks_2d: Dict) -> float:
        """Calculate forehead height"""
        if 'forehead_top' in landmarks_2d and 'forehead_center' in landmarks_2d:
            top = landmarks_2d['forehead_top']
            center = landmarks_2d['forehead_center']
            return abs(top[1] - center[1])
        return 0.0
    
    def _calculate_forehead_width(self, landmarks_2d: Dict) -> float:
        """Calculate forehead width"""
        if 'forehead_left' in landmarks_2d and 'forehead_right' in landmarks_2d:
            left = landmarks_2d['forehead_left']
            right = landmarks_2d['forehead_right']
            return abs(right[0] - left[0])
        return 0.0
    
    def _calculate_eyebrow_height(self, landmarks_2d: Dict) -> float:
        """Calculate average eyebrow height"""
        left_height = 0.0
        right_height = 0.0
        
        if 'left_eyebrow_top' in landmarks_2d and 'left_eye_top' in landmarks_2d:
            eyebrow = landmarks_2d['left_eyebrow_top']
            eye = landmarks_2d['left_eye_top']
            left_height = abs(eyebrow[1] - eye[1])
        
        if 'right_eyebrow_top' in landmarks_2d and 'right_eye_top' in landmarks_2d:
            eyebrow = landmarks_2d['right_eyebrow_top']
            eye = landmarks_2d['right_eye_top']
            right_height = abs(eyebrow[1] - eye[1])
        
        return (left_height + right_height) / 2.0 if left_height > 0 and right_height > 0 else max(left_height, right_height)
    
    def _calculate_eyebrow_width(self, landmarks_2d: Dict) -> float:
        """Calculate eyebrow width"""
        left_width = 0.0
        right_width = 0.0
        
        if 'left_eyebrow_outer' in landmarks_2d and 'left_eyebrow_inner' in landmarks_2d:
            outer = landmarks_2d['left_eyebrow_outer']
            inner = landmarks_2d['left_eyebrow_inner']
            left_width = abs(outer[0] - inner[0])
        
        if 'right_eyebrow_outer' in landmarks_2d and 'right_eyebrow_inner' in landmarks_2d:
            outer = landmarks_2d['right_eyebrow_outer']
            inner = landmarks_2d['right_eyebrow_inner']
            right_width = abs(outer[0] - inner[0])
        
        return (left_width + right_width) / 2.0 if left_width > 0 and right_width > 0 else max(left_width, right_width)
    
    def _calculate_eye_height(self, landmarks_2d: Dict) -> float:
        """Calculate average eye height"""
        left_height = 0.0
        right_height = 0.0
        
        if 'left_eye_top' in landmarks_2d and 'left_eye_bottom' in landmarks_2d:
            top = landmarks_2d['left_eye_top']
            bottom = landmarks_2d['left_eye_bottom']
            left_height = abs(bottom[1] - top[1])
        
        if 'right_eye_top' in landmarks_2d and 'right_eye_bottom' in landmarks_2d:
            top = landmarks_2d['right_eye_top']
            bottom = landmarks_2d['right_eye_bottom']
            right_height = abs(bottom[1] - top[1])
        
        return (left_height + right_height) / 2.0 if left_height > 0 and right_height > 0 else max(left_height, right_height)
    
    def _calculate_eye_width(self, landmarks_2d: Dict) -> float:
        """Calculate average eye width"""
        left_width = 0.0
        right_width = 0.0
        
        if 'left_eye_outer' in landmarks_2d and 'left_eye_inner' in landmarks_2d:
            outer = landmarks_2d['left_eye_outer']
            inner = landmarks_2d['left_eye_inner']
            left_width = abs(outer[0] - inner[0])
        
        if 'right_eye_outer' in landmarks_2d and 'right_eye_inner' in landmarks_2d:
            outer = landmarks_2d['right_eye_outer']
            inner = landmarks_2d['right_eye_inner']
            right_width = abs(outer[0] - inner[0])
        
        return (left_width + right_width) / 2.0 if left_width > 0 and right_width > 0 else max(left_width, right_width)
    
    def _calculate_mouth_width(self, landmarks_2d: Dict) -> float:
        """Calculate mouth width"""
        if 'mouth_left' in landmarks_2d and 'mouth_right' in landmarks_2d:
            left = landmarks_2d['mouth_left']
            right = landmarks_2d['mouth_right']
            return abs(right[0] - left[0])
        return 0.0
    
    def _calculate_mouth_height(self, landmarks_2d: Dict) -> float:
        """Calculate mouth height"""
        if 'mouth_top' in landmarks_2d and 'mouth_bottom' in landmarks_2d:
            top = landmarks_2d['mouth_top']
            bottom = landmarks_2d['mouth_bottom']
            return abs(bottom[1] - top[1])
        return 0.0
    
    def _calculate_face_width(self, landmarks_2d: Dict) -> float:
        """Calculate face width"""
        if 'jaw_left' in landmarks_2d and 'jaw_right' in landmarks_2d:
            left = landmarks_2d['jaw_left']
            right = landmarks_2d['jaw_right']
            return abs(right[0] - left[0])
        return 0.0
    
    def _calculate_face_height(self, landmarks_2d: Dict) -> float:
        """Calculate face height"""
        if 'forehead_top' in landmarks_2d and 'chin' in landmarks_2d:
            top = landmarks_2d['forehead_top']
            bottom = landmarks_2d['chin']
            return abs(bottom[1] - top[1])
        return 0.0
    
    def _calculate_head_tilt(self, landmarks_2d: Dict) -> float:
        """Calculate head tilt angle"""
        if 'left_eye_outer' in landmarks_2d and 'right_eye_outer' in landmarks_2d:
            left = landmarks_2d['left_eye_outer']
            right = landmarks_2d['right_eye_outer']
            # Calculate angle from horizontal
            dx = right[0] - left[0]
            dy = right[1] - left[1]
            if dx != 0:
                return np.arctan(dy / dx) * 180 / np.pi
        return 0.0
    
    def _calculate_head_turn(self, landmarks_2d: Dict) -> float:
        """Calculate head turn angle"""
        if 'nose_tip' in landmarks_2d and 'nose_bridge' in landmarks_2d:
            tip = landmarks_2d['nose_tip']
            bridge = landmarks_2d['nose_bridge']
            # Calculate angle from vertical
            dx = tip[0] - bridge[0]
            dy = tip[1] - bridge[1]
            if dy != 0:
                return np.arctan(dx / dy) * 180 / np.pi
        return 0.0
    
    def _calculate_head_nod(self, landmarks_2d: Dict) -> float:
        """Calculate head nod angle"""
        if 'forehead_center' in landmarks_2d and 'chin' in landmarks_2d:
            forehead = landmarks_2d['forehead_center']
            chin = landmarks_2d['chin']
            # Calculate angle from vertical
            dx = chin[0] - forehead[0]
            dy = chin[1] - forehead[1]
            if dy != 0:
                return np.arctan(dx / dy) * 180 / np.pi
        return 0.0
