"""
Facial Paralysis Analyzer - Detects facial paralysis using asymmetry analysis
Based on FaCiPa algorithm adapted for MediaPipe facial landmarks
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional


class FacialParalysisAnalyzer:
    """
    Analyzes facial paralysis using asymmetry detection
    Based on FaCiPa algorithm adapted for MediaPipe facial landmarks
    """
    
    def __init__(self):
        """Initialize the facial paralysis analyzer"""
        # Asymmetry thresholds (in pixels)
        self.EYEBROW_ASYMMETRY_THRESHOLD = 15
        self.EYE_ASYMMETRY_THRESHOLD = 15
        self.NOSE_ASYMMETRY_THRESHOLD = 15
        self.MOUTH_ASYMMETRY_THRESHOLD = 15
        
        # Minimum number of asymmetries to detect paralysis
        self.MIN_ASYMMETRIES_FOR_PARALYSIS = 2
        
        # Analysis history for smoothing
        self.analysis_history: List[Dict] = []
        self.max_history_size = 10
        
        # MediaPipe face mesh landmark indices for asymmetry detection
        self.landmark_indices = {
            # Eyebrows (using MediaPipe face mesh indices)
            'right_eyebrow_outer': 70,   # Right eyebrow outer
            'right_eyebrow_inner': 63,   # Right eyebrow inner
            'left_eyebrow_inner': 105,   # Left eyebrow inner
            'left_eyebrow_outer': 66,    # Left eyebrow outer
            
            # Eyes (using MediaPipe face mesh indices)
            'right_eye_outer': 33,       # Right eye outer corner
            'right_eye_inner': 133,      # Right eye inner corner
            'left_eye_inner': 362,       # Left eye inner corner
            'left_eye_outer': 263,       # Left eye outer corner
            
            # Nose (using MediaPipe face mesh indices)
            'nose_right': 174,           # Right nostril
            'nose_left': 397,            # Left nostril
            
            # Mouth (using MediaPipe face mesh indices)
            'mouth_right': 61,           # Right mouth corner
            'mouth_left': 291            # Left mouth corner
        }
        
        self.model_loaded = True
        print("MediaPipe-based facial paralysis analyzer initialized")
    
    def analyze_paralysis(self, face_landmarks, frame_width: int, frame_height: int) -> Dict:
        """
        Analyze facial paralysis using MediaPipe face landmarks
        
        Args:
            face_landmarks: MediaPipe face landmarks
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
            
        Returns:
            Dict: Analysis results
        """
        if not self.model_loaded:
            return {
                'paralysis_detected': False,
                'confidence': 0.0,
                'asymmetries': {},
                'total_asymmetries': 0,
                'error': 'Model not loaded'
            }
        
        try:
            # Convert MediaPipe landmarks to pixel coordinates
            landmarks = []
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)
                landmarks.append([x, y])
            
            landmarks = np.array(landmarks)
            
            # Calculate asymmetries
            asymmetries = self._calculate_asymmetries(landmarks)
            
            # Determine if paralysis is detected
            total_asymmetries = sum(1 for asym in asymmetries.values() if asym['detected'])
            paralysis_detected = total_asymmetries >= self.MIN_ASYMMETRIES_FOR_PARALYSIS
            
            # Calculate confidence based on number and severity of asymmetries
            confidence = self._calculate_confidence(asymmetries, total_asymmetries)
            
            result = {
                'paralysis_detected': paralysis_detected,
                'confidence': confidence,
                'asymmetries': asymmetries,
                'total_asymmetries': total_asymmetries,
                'landmarks': landmarks.tolist() if len(landmarks) > 0 else []
            }
            
            # Add to history for smoothing
            self.analysis_history.append(result)
            if len(self.analysis_history) > self.max_history_size:
                self.analysis_history.pop(0)
            
            return result
            
        except Exception as e:
            return {
                'paralysis_detected': False,
                'confidence': 0.0,
                'asymmetries': {},
                'total_asymmetries': 0,
                'error': f'Analysis failed: {str(e)}'
            }
    
    def _calculate_asymmetries(self, landmarks: np.ndarray) -> Dict:
        """
        Calculate facial asymmetries based on 68-point landmarks
        
        Args:
            landmarks: 68-point facial landmarks
            
        Returns:
            Dict: Asymmetry analysis results
        """
        asymmetries = {}
        
        # Eyebrow asymmetry
        right_eyebrow_y = landmarks[self.landmark_indices['right_eyebrow_inner']][1]
        left_eyebrow_y = landmarks[self.landmark_indices['left_eyebrow_inner']][1]
        eyebrow_diff = abs(left_eyebrow_y - right_eyebrow_y)
        
        asymmetries['eyebrow'] = {
            'detected': eyebrow_diff >= self.EYEBROW_ASYMMETRY_THRESHOLD,
            'difference': eyebrow_diff,
            'threshold': self.EYEBROW_ASYMMETRY_THRESHOLD,
            'severity': min(1.0, eyebrow_diff / self.EYEBROW_ASYMMETRY_THRESHOLD)
        }
        
        # Eye asymmetry
        right_eye_y = landmarks[self.landmark_indices['right_eye_inner']][1]
        left_eye_y = landmarks[self.landmark_indices['left_eye_inner']][1]
        eye_diff = abs(left_eye_y - right_eye_y)
        
        asymmetries['eye'] = {
            'detected': eye_diff >= self.EYE_ASYMMETRY_THRESHOLD,
            'difference': eye_diff,
            'threshold': self.EYE_ASYMMETRY_THRESHOLD,
            'severity': min(1.0, eye_diff / self.EYE_ASYMMETRY_THRESHOLD)
        }
        
        # Nose asymmetry
        right_nose_y = landmarks[self.landmark_indices['nose_right']][1]
        left_nose_y = landmarks[self.landmark_indices['nose_left']][1]
        nose_diff = abs(left_nose_y - right_nose_y)
        
        asymmetries['nose'] = {
            'detected': nose_diff >= self.NOSE_ASYMMETRY_THRESHOLD,
            'difference': nose_diff,
            'threshold': self.NOSE_ASYMMETRY_THRESHOLD,
            'severity': min(1.0, nose_diff / self.NOSE_ASYMMETRY_THRESHOLD)
        }
        
        # Mouth asymmetry
        right_mouth_y = landmarks[self.landmark_indices['mouth_right']][1]
        left_mouth_y = landmarks[self.landmark_indices['mouth_left']][1]
        mouth_diff = abs(left_mouth_y - right_mouth_y)
        
        asymmetries['mouth'] = {
            'detected': mouth_diff >= self.MOUTH_ASYMMETRY_THRESHOLD,
            'difference': mouth_diff,
            'threshold': self.MOUTH_ASYMMETRY_THRESHOLD,
            'severity': min(1.0, mouth_diff / self.MOUTH_ASYMMETRY_THRESHOLD)
        }
        
        return asymmetries
    
    def _calculate_confidence(self, asymmetries: Dict, total_asymmetries: int) -> float:
        """
        Calculate confidence score for paralysis detection
        
        Args:
            asymmetries: Asymmetry analysis results
            total_asymmetries: Total number of detected asymmetries
            
        Returns:
            float: Confidence score (0.0 to 1.0)
        """
        if total_asymmetries == 0:
            return 0.0
        
        # Base confidence from number of asymmetries
        base_confidence = min(1.0, total_asymmetries / 4.0)
        
        # Weight by severity of asymmetries
        severity_sum = sum(asym['severity'] for asym in asymmetries.values() if asym['detected'])
        severity_factor = min(1.0, severity_sum / 4.0)
        
        # Combine base confidence and severity
        confidence = (base_confidence + severity_factor) / 2.0
        
        return confidence
    
    def get_smoothed_result(self) -> Dict:
        """
        Get smoothed analysis result based on recent history
        
        Returns:
            Dict: Smoothed analysis result
        """
        if not self.analysis_history:
            return {
                'paralysis_detected': False,
                'confidence': 0.0,
                'asymmetries': {},
                'total_asymmetries': 0
            }
        
        # Use recent results for smoothing
        recent_results = self.analysis_history[-5:]  # Last 5 results
        
        # Average confidence
        avg_confidence = sum(r['confidence'] for r in recent_results) / len(recent_results)
        
        # Most common paralysis detection
        paralysis_detections = [r['paralysis_detected'] for r in recent_results]
        smoothed_paralysis = sum(paralysis_detections) / len(paralysis_detections) > 0.5
        
        # Average total asymmetries
        avg_asymmetries = sum(r['total_asymmetries'] for r in recent_results) / len(recent_results)
        
        return {
            'paralysis_detected': smoothed_paralysis,
            'confidence': avg_confidence,
            'total_asymmetries': avg_asymmetries
        }
    
    def draw_paralysis_overlay(self, frame: np.ndarray, result: Dict) -> None:
        """
        Draw paralysis analysis overlay on the frame
        
        Args:
            frame: OpenCV frame
            result: Analysis result to display
        """
        h, w = frame.shape[:2]
        
        # Create overlay background
        overlay = frame.copy()
        cv2.rectangle(overlay, (w - 400, 10), (w - 10, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Display results
        y_offset = 30
        line_height = 25
        
        # Paralysis status
        status_color = (0, 0, 255) if result.get('paralysis_detected', False) else (0, 255, 0)
        status_text = "PARALYSIS DETECTED" if result.get('paralysis_detected', False) else "NORMAL"
        cv2.putText(frame, f"Status: {status_text}", 
                   (w - 390, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        y_offset += line_height
        
        # Confidence
        confidence = result.get('confidence', 0.0)
        cv2.putText(frame, f"Confidence: {confidence:.2f}", 
                   (w - 390, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += line_height
        
        # Total asymmetries
        total_asymmetries = result.get('total_asymmetries', 0)
        cv2.putText(frame, f"Asymmetries: {total_asymmetries}/4", 
                   (w - 390, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += line_height
        
        # Individual asymmetries
        asymmetries = result.get('asymmetries', {})
        for feature, asym in asymmetries.items():
            if asym.get('detected', False):
                color = (0, 0, 255)  # Red for detected
                text = f"{feature.upper()}: {asym['difference']:.1f}px"
            else:
                color = (0, 255, 0)  # Green for normal
                text = f"{feature.upper()}: Normal"
            
            cv2.putText(frame, text, 
                       (w - 390, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += line_height
    
    def set_thresholds(self, eyebrow: int = 15, eye: int = 15, nose: int = 15, mouth: int = 15) -> None:
        """
        Set custom asymmetry thresholds
        
        Args:
            eyebrow: Eyebrow asymmetry threshold in pixels
            eye: Eye asymmetry threshold in pixels
            nose: Nose asymmetry threshold in pixels
            mouth: Mouth asymmetry threshold in pixels
        """
        self.EYEBROW_ASYMMETRY_THRESHOLD = eyebrow
        self.EYE_ASYMMETRY_THRESHOLD = eye
        self.NOSE_ASYMMETRY_THRESHOLD = nose
        self.MOUTH_ASYMMETRY_THRESHOLD = mouth
        print(f"Paralysis thresholds set - eyebrow: {eyebrow}, eye: {eye}, nose: {nose}, mouth: {mouth}")
    
    def reset_history(self) -> None:
        """Reset analysis history"""
        self.analysis_history.clear()
        print("Paralysis analysis history reset")
