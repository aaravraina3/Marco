# -*- coding: utf-8 -*-
"""
Tremor Analyzer for Healthcare Eye Tracker
Based on cv-tremor-amplitude by James Bungay
Simplified for real-time integration with MediaPipe hand tracking
"""

import cv2
import numpy as np
import math
import mediapipe as mp
from enum import Enum
from collections import deque
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


class TremorType(Enum):
    RESTING = "resting"
    POSTURAL = "postural"


class TremorAnalyzer:
    def __init__(self, tremor_type=TremorType.RESTING):
        """
        Initialize tremor analyzer for real-time hand tremor detection
        
        Args:
            tremor_type: Type of tremor to detect (resting or postural)
        """
        self.tremor_type = tremor_type
        self.hand_detector = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        )
        
        # Hand landmarks to track (simplified to key points)
        if tremor_type == TremorType.RESTING:
            # For resting tremor: track MCP joints (knuckles)
            self.landmarks_to_track = [
                mp_hands.HandLandmark.INDEX_FINGER_MCP,
                mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
                mp_hands.HandLandmark.RING_FINGER_MCP
            ]
        else:
            # For postural tremor: track thumb landmarks
            self.landmarks_to_track = [
                mp_hands.HandLandmark.THUMB_MCP,
                mp_hands.HandLandmark.THUMB_IP,
                mp_hands.HandLandmark.THUMB_TIP
            ]
        
        # Store tremor path data
        self.tremor_paths = [deque(maxlen=60) for _ in self.landmarks_to_track]  # ~2 seconds at 30fps
        self.frame_times = deque(maxlen=60)
        self.last_update_time = time.time()
        
        # Tremor analysis parameters
        self.min_frames_for_analysis = 30  # Minimum frames needed for analysis
        self.current_amplitude = 0.0
        self.updrs_rating = 0
        self.tremor_detected = False
        
        # Camera parameters (simplified - can be calibrated)
        self.estimated_depth_cm = 50  # Estimated hand depth
        self.pixel_size_cm = 0.1  # Approximate pixel size at 50cm depth
        
    def update_hand_depth_estimate(self, face_distance_px, face_distance_cm):
        """
        Update hand depth estimate based on face distance measurement
        """
        if face_distance_px > 0 and face_distance_cm > 0:
            # Estimate hand depth based on face distance
            # Hands are typically 10-20cm closer than face
            self.estimated_depth_cm = max(30, face_distance_cm - 15)
            
            # Update pixel size estimate (simplified calculation)
            # Assuming 1080p camera with ~50mm focal length equivalent
            self.pixel_size_cm = (self.estimated_depth_cm * 0.05) / 1080
    
    def analyze_tremor(self, frame):
        """
        Analyze tremor in real-time from video frame
        
        Args:
            frame: OpenCV frame from camera
            
        Returns:
            dict: Tremor analysis results
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hand_detector.process(frame_rgb)
        
        current_time = time.time()
        self.frame_times.append(current_time)
        
        if results.multi_hand_landmarks:
            # Get landmarks for first detected hand
            landmarks = results.multi_hand_landmarks[0]
            
            # Extract x coordinates for tremor analysis
            for i, landmark_idx in enumerate(self.landmarks_to_track):
                x_coord = landmarks.landmark[landmark_idx].x * frame.shape[1]
                self.tremor_paths[i].append(x_coord)
            
            # Analyze tremor if we have enough data
            if len(self.tremor_paths[0]) >= self.min_frames_for_analysis:
                self._calculate_tremor_amplitude()
                
            return {
                'hand_detected': True,
                'tremor_detected': self.tremor_detected,
                'amplitude_cm': self.current_amplitude,
                'updrs_rating': self.updrs_rating,
                'landmarks': landmarks,
                'tremor_paths': [list(path) for path in self.tremor_paths]
            }
        else:
            return {
                'hand_detected': False,
                'tremor_detected': False,
                'amplitude_cm': 0.0,
                'updrs_rating': 0,
                'landmarks': None,
                'tremor_paths': []
            }
    
    def _calculate_tremor_amplitude(self):
        """
        Calculate tremor amplitude from the collected tremor paths
        """
        if len(self.tremor_paths[0]) < self.min_frames_for_analysis:
            return
        
        # Calculate amplitude for each landmark
        amplitudes = []
        
        for path in self.tremor_paths:
            if len(path) < 10:
                continue
                
            # Convert path to numpy array for analysis
            path_array = np.array(path)
            
            # Center the path around zero
            centered_path = path_array - np.mean(path_array)
            
            # Calculate peak-to-peak amplitude using differentiation method
            # (simplified version of the original algorithm)
            if len(centered_path) > 5:
                # Find peaks and troughs using gradient changes
                diff = np.diff(centered_path)
                sign_changes = np.diff(np.sign(diff))
                peaks_troughs = []
                
                for i in range(1, len(sign_changes)):
                    if sign_changes[i] != 0:  # Sign change detected
                        peaks_troughs.append(centered_path[i])
                
                if len(peaks_troughs) >= 2:
                    # Calculate amplitude as difference between max and min
                    amplitude_pixels = max(peaks_troughs) - min(peaks_troughs)
                    amplitude_cm = amplitude_pixels * self.pixel_size_cm
                    amplitudes.append(amplitude_cm)
        
        if amplitudes:
            # Take median amplitude across all landmarks
            self.current_amplitude = np.median(amplitudes)
            
            # Determine if tremor is detected (threshold: 0.5cm)
            self.tremor_detected = self.current_amplitude > 0.5
            
            # Calculate UPDRS rating
            self.updrs_rating = self._calculate_updrs_rating(self.current_amplitude)
        else:
            self.current_amplitude = 0.0
            self.tremor_detected = False
            self.updrs_rating = 0
    
    def _calculate_updrs_rating(self, amplitude_cm):
        """
        Convert tremor amplitude to MDS-UPDRS rating
        Based on clinical standards
        """
        if amplitude_cm <= 0.01:
            return 0  # No tremor
        elif amplitude_cm < 1.0:
            return 1  # Slight tremor
        elif amplitude_cm < 3.0:
            return 2  # Mild tremor
        elif amplitude_cm < 10.0:
            return 3  # Moderate tremor
        else:
            return 4  # Severe tremor
    
    def draw_hand_landmarks(self, frame, landmarks):
        """
        Draw hand landmarks on frame
        """
        if landmarks:
            mp_drawing.draw_landmarks(
                frame,
                landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
        return frame
    
    def get_tremor_status_text(self):
        """
        Get formatted tremor status text for display
        """
        if self.tremor_detected:
            return f"TREMOR DETECTED - {self.updrs_rating}/4"
        else:
            return "NO TREMOR"
    
    def get_tremor_details(self):
        """
        Get detailed tremor information
        """
        return {
            'amplitude_cm': self.current_amplitude,
            'updrs_rating': self.updrs_rating,
            'tremor_detected': self.tremor_detected,
            'tremor_type': self.tremor_type.value,
            'landmarks_tracked': len(self.landmarks_to_track),
            'frames_analyzed': len(self.tremor_paths[0]) if self.tremor_paths else 0
        }
    
    def reset_analysis(self):
        """
        Reset tremor analysis data
        """
        self.tremor_paths = [deque(maxlen=60) for _ in self.landmarks_to_track]
        self.frame_times = deque(maxlen=60)
        self.current_amplitude = 0.0
        self.updrs_rating = 0
        self.tremor_detected = False


def create_tremor_analyzer(tremor_type="resting"):
    """
    Factory function to create tremor analyzer
    
    Args:
        tremor_type: "resting" or "postural"
    
    Returns:
        TremorAnalyzer instance
    """
    if tremor_type.lower() == "postural":
        return TremorAnalyzer(TremorType.POSTURAL)
    else:
        return TremorAnalyzer(TremorType.RESTING)
