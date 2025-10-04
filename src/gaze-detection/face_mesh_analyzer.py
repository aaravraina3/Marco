"""
Face Mesh Analyzer - Main orchestrator for facial feature analysis
Analyzes forehead, eyebrows, smile, hair, and other facial features
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from feature_extractors import FeatureExtractors
from forehead_analyzer import ForeheadAnalyzer
from eyebrow_analyzer import EyebrowAnalyzer
from smile_analyzer import SmileAnalyzer
from hair_analyzer import HairAnalyzer
from expression_classifier import ExpressionClassifier
from facial_paralysis_analyzer import FacialParalysisAnalyzer


@dataclass
class FaceAnalysisResult:
    """Container for all facial analysis results"""
    # Forehead analysis
    forehead_raised: bool = False
    forehead_lowered: bool = False
    forehead_intensity: float = 0.0
    
    # Eyebrow analysis
    left_eyebrow_raised: bool = False
    right_eyebrow_raised: bool = False
    eyebrow_raise_intensity: float = 0.0
    
    # Smile analysis
    is_smiling: bool = False
    smile_intensity: float = 0.0
    smile_confidence: float = 0.0
    
    # Hair/head analysis
    head_tilt_left: bool = False
    head_tilt_right: bool = False
    head_nod_up: bool = False
    head_nod_down: bool = False
    head_turn_left: bool = False
    head_turn_right: bool = False
    
    # Overall expression
    expression_type: str = "neutral"
    expression_confidence: float = 0.0
    
    # Facial paralysis detection
    paralysis_detected: bool = False
    paralysis_confidence: float = 0.0
    total_asymmetries: int = 0
    
    # Raw measurements
    forehead_height: float = 0.0
    eyebrow_height: float = 0.0
    mouth_width: float = 0.0
    mouth_height: float = 0.0
    face_width: float = 0.0
    face_height: float = 0.0


class FaceMeshAnalyzer:
    """
    Main face mesh analyzer that coordinates all facial feature analysis
    """
    
    def __init__(self):
        """Initialize the face mesh analyzer with all sub-analyzers"""
        self.feature_extractors = FeatureExtractors()
        self.forehead_analyzer = ForeheadAnalyzer()
        self.eyebrow_analyzer = EyebrowAnalyzer()
        self.smile_analyzer = SmileAnalyzer()
        self.hair_analyzer = HairAnalyzer()
        self.expression_classifier = ExpressionClassifier()
        # self.paralysis_analyzer = FacialParalysisAnalyzer()  # Temporarily disabled
        
        # Analysis history for smoothing
        self.analysis_history: List[FaceAnalysisResult] = []
        self.max_history_size = 10
        
        # Baseline measurements (set during calibration)
        self.baseline_measurements: Optional[Dict] = None
        self.is_calibrated = False
        
    def calibrate(self, face_landmarks, frame_width: int, frame_height: int) -> bool:
        """
        Calibrate the analyzer with baseline measurements
        
        Args:
            face_landmarks: MediaPipe face landmarks
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
            
        Returns:
            bool: True if calibration successful
        """
        try:
            # Extract baseline measurements
            self.baseline_measurements = self.feature_extractors.extract_baseline_measurements(
                face_landmarks, frame_width, frame_height
            )
            
            # Calibrate individual analyzers
            self.forehead_analyzer.calibrate(self.baseline_measurements)
            self.eyebrow_analyzer.calibrate(self.baseline_measurements)
            self.smile_analyzer.calibrate(self.baseline_measurements)
            self.hair_analyzer.calibrate(self.baseline_measurements)
            
            self.is_calibrated = True
            print("Face mesh analyzer calibrated successfully")
            return True
            
        except Exception as e:
            print(f"Calibration failed: {e}")
            return False
    
    def analyze_face(self, face_landmarks, frame_width: int, frame_height: int, frame: np.ndarray = None) -> FaceAnalysisResult:
        """
        Analyze all facial features and return comprehensive results
        
        Args:
            face_landmarks: MediaPipe face landmarks
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
            frame: Optional frame for paralysis analysis
            
        Returns:
            FaceAnalysisResult: Comprehensive facial analysis results
        """
        if not self.is_calibrated:
            # Return default result if not calibrated
            return FaceAnalysisResult()
        
        try:
            # Extract current measurements
            current_measurements = self.feature_extractors.extract_current_measurements(
                face_landmarks, frame_width, frame_height
            )
            
            # Analyze each feature
            forehead_result = self.forehead_analyzer.analyze(current_measurements)
            eyebrow_result = self.eyebrow_analyzer.analyze(current_measurements)
            smile_result = self.smile_analyzer.analyze(current_measurements)
            hair_result = self.hair_analyzer.analyze(current_measurements)
            
            # Classify overall expression
            expression_result = self.expression_classifier.classify_expression(
                forehead_result, eyebrow_result, smile_result, hair_result
            )
            
            # Analyze facial paralysis if frame is provided (temporarily disabled)
            paralysis_result = {}
            # if frame is not None:
            #     paralysis_result = self.paralysis_analyzer.analyze_paralysis(frame)
            
            # Create comprehensive result
            result = FaceAnalysisResult(
                # Forehead
                forehead_raised=forehead_result.get('raised', False),
                forehead_lowered=forehead_result.get('lowered', False),
                forehead_intensity=forehead_result.get('intensity', 0.0),
                
                # Eyebrows
                left_eyebrow_raised=eyebrow_result.get('left_raised', False),
                right_eyebrow_raised=eyebrow_result.get('right_raised', False),
                eyebrow_raise_intensity=eyebrow_result.get('intensity', 0.0),
                
                # Smile
                is_smiling=smile_result.get('is_smiling', False),
                smile_intensity=smile_result.get('intensity', 0.0),
                smile_confidence=smile_result.get('confidence', 0.0),
                
                # Head movement
                head_tilt_left=hair_result.get('tilt_left', False),
                head_tilt_right=hair_result.get('tilt_right', False),
                head_nod_up=hair_result.get('nod_up', False),
                head_nod_down=hair_result.get('nod_down', False),
                head_turn_left=hair_result.get('turn_left', False),
                head_turn_right=hair_result.get('turn_right', False),
                
                # Overall expression
                expression_type=expression_result.get('type', 'neutral'),
                expression_confidence=expression_result.get('confidence', 0.0),
                
                # Facial paralysis
                paralysis_detected=paralysis_result.get('paralysis_detected', False),
                paralysis_confidence=paralysis_result.get('confidence', 0.0),
                total_asymmetries=paralysis_result.get('total_asymmetries', 0),
                
                # Raw measurements
                forehead_height=current_measurements.get('forehead_height', 0.0),
                eyebrow_height=current_measurements.get('eyebrow_height', 0.0),
                mouth_width=current_measurements.get('mouth_width', 0.0),
                mouth_height=current_measurements.get('mouth_height', 0.0),
                face_width=current_measurements.get('face_width', 0.0),
                face_height=current_measurements.get('face_height', 0.0)
            )
            
            # Add to history for smoothing
            self.analysis_history.append(result)
            if len(self.analysis_history) > self.max_history_size:
                self.analysis_history.pop(0)
            
            return result
            
        except Exception as e:
            print(f"Face analysis failed: {e}")
            return FaceAnalysisResult()
    
    def get_smoothed_result(self) -> FaceAnalysisResult:
        """
        Get smoothed analysis result based on recent history
        
        Returns:
            FaceAnalysisResult: Smoothed analysis result
        """
        if not self.analysis_history:
            return FaceAnalysisResult()
        
        # Simple averaging for smoothing
        recent_results = self.analysis_history[-5:]  # Last 5 results
        
        smoothed = FaceAnalysisResult()
        
        # Average boolean values (threshold at 0.5)
        smoothed.forehead_raised = sum(r.forehead_raised for r in recent_results) / len(recent_results) > 0.5
        smoothed.forehead_lowered = sum(r.forehead_lowered for r in recent_results) / len(recent_results) > 0.5
        smoothed.left_eyebrow_raised = sum(r.left_eyebrow_raised for r in recent_results) / len(recent_results) > 0.5
        smoothed.right_eyebrow_raised = sum(r.right_eyebrow_raised for r in recent_results) / len(recent_results) > 0.5
        smoothed.is_smiling = sum(r.is_smiling for r in recent_results) / len(recent_results) > 0.5
        
        # Average float values
        smoothed.forehead_intensity = sum(r.forehead_intensity for r in recent_results) / len(recent_results)
        smoothed.eyebrow_raise_intensity = sum(r.eyebrow_raise_intensity for r in recent_results) / len(recent_results)
        smoothed.smile_intensity = sum(r.smile_intensity for r in recent_results) / len(recent_results)
        smoothed.smile_confidence = sum(r.smile_confidence for r in recent_results) / len(recent_results)
        
        # Most common expression type
        expression_types = [r.expression_type for r in recent_results]
        smoothed.expression_type = max(set(expression_types), key=expression_types.count)
        smoothed.expression_confidence = sum(r.expression_confidence for r in recent_results) / len(recent_results)
        
        return smoothed
    
    def draw_analysis_overlay(self, frame, result: FaceAnalysisResult) -> None:
        """
        Draw analysis results overlay on the frame
        
        Args:
            frame: OpenCV frame
            result: FaceAnalysisResult to display
        """
        h, w = frame.shape[:2]
        
        # Create overlay background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 300), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Display results
        y_offset = 30
        line_height = 25
        
        # Forehead analysis
        cv2.putText(frame, f"Forehead: {'RAISED' if result.forehead_raised else 'LOWERED' if result.forehead_lowered else 'NORMAL'}", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y_offset += line_height
        
        # Eyebrow analysis
        cv2.putText(frame, f"Eyebrows: {'RAISED' if result.left_eyebrow_raised or result.right_eyebrow_raised else 'NORMAL'}", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        y_offset += line_height
        
        # Smile analysis
        cv2.putText(frame, f"Smile: {'YES' if result.is_smiling else 'NO'} ({result.smile_intensity:.2f})", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += line_height
        
        # Head movement
        head_movement = []
        if result.head_tilt_left: head_movement.append("TILT-L")
        if result.head_tilt_right: head_movement.append("TILT-R")
        if result.head_nod_up: head_movement.append("NOD-UP")
        if result.head_nod_down: head_movement.append("NOD-DOWN")
        if result.head_turn_left: head_movement.append("TURN-L")
        if result.head_turn_right: head_movement.append("TURN-R")
        
        head_text = " | ".join(head_movement) if head_movement else "STABLE"
        cv2.putText(frame, f"Head: {head_text}", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        y_offset += line_height
        
        # Overall expression
        cv2.putText(frame, f"Expression: {result.expression_type.upper()}", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        y_offset += line_height
        
        # Facial paralysis detection
        paralysis_color = (0, 0, 255) if result.paralysis_detected else (0, 255, 0)
        paralysis_text = "PARALYSIS DETECTED" if result.paralysis_detected else "NORMAL"
        cv2.putText(frame, f"Paralysis: {paralysis_text}", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, paralysis_color, 2)
        y_offset += line_height
        
        if result.paralysis_detected:
            cv2.putText(frame, f"Confidence: {result.paralysis_confidence:.2f}", 
                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += line_height
            cv2.putText(frame, f"Asymmetries: {result.total_asymmetries}/4", 
                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += line_height
        
        # Raw measurements
        cv2.putText(frame, f"Forehead H: {result.forehead_height:.1f}", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_offset += line_height
        cv2.putText(frame, f"Eyebrow H: {result.eyebrow_height:.1f}", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_offset += line_height
        cv2.putText(frame, f"Mouth W: {result.mouth_width:.1f}", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_offset += line_height
        cv2.putText(frame, f"Face W: {result.face_width:.1f}", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def reset_calibration(self) -> None:
        """Reset calibration and clear history"""
        self.baseline_measurements = None
        self.is_calibrated = False
        self.analysis_history.clear()
        print("Face mesh analyzer calibration reset")
