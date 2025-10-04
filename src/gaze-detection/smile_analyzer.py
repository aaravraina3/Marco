"""
Smile Analyzer - Analyzes smile and mouth expressions
"""

from typing import Dict, List, Optional
import math


class SmileAnalyzer:
    """
    Analyzes smile and mouth expressions
    """
    
    def __init__(self):
        """Initialize the smile analyzer"""
        self.baseline_mouth_width = 0.0
        self.baseline_mouth_height = 0.0
        self.is_calibrated = False
        
        # Thresholds for smile detection
        self.smile_width_threshold = 0.15  # 15% increase from baseline
        self.smile_height_threshold = 0.10  # 10% increase from baseline
        
        # History for smoothing
        self.width_history: List[float] = []
        self.height_history: List[float] = []
        self.max_history_size = 10
    
    def calibrate(self, baseline_measurements: Dict) -> None:
        """
        Calibrate the analyzer with baseline measurements
        
        Args:
            baseline_measurements: Dictionary containing baseline measurements
        """
        if 'mouth_width' in baseline_measurements and 'mouth_height' in baseline_measurements:
            self.baseline_mouth_width = baseline_measurements['mouth_width']
            self.baseline_mouth_height = baseline_measurements['mouth_height']
            self.is_calibrated = True
            print(f"Smile analyzer calibrated - baseline width: {self.baseline_mouth_width:.2f}, height: {self.baseline_mouth_height:.2f}")
        else:
            print("Warning: No mouth measurements in baseline measurements")
    
    def analyze(self, current_measurements: Dict) -> Dict:
        """
        Analyze current mouth measurements
        
        Args:
            current_measurements: Dictionary containing current measurements
            
        Returns:
            Dict: Analysis results
        """
        if not self.is_calibrated:
            return {
                'is_smiling': False,
                'intensity': 0.0,
                'confidence': 0.0,
                'width_change': 0.0,
                'height_change': 0.0
            }
        
        current_width = current_measurements.get('mouth_width', self.baseline_mouth_width)
        current_height = current_measurements.get('mouth_height', self.baseline_mouth_height)
        
        # Add to history for smoothing
        self.width_history.append(current_width)
        self.height_history.append(current_height)
        if len(self.width_history) > self.max_history_size:
            self.width_history.pop(0)
        if len(self.height_history) > self.max_history_size:
            self.height_history.pop(0)
        
        # Calculate smoothed measurements
        smoothed_width = sum(self.width_history) / len(self.width_history)
        smoothed_height = sum(self.height_history) / len(self.height_history)
        
        # Calculate change percentages
        width_change = (smoothed_width - self.baseline_mouth_width) / self.baseline_mouth_width
        height_change = (smoothed_height - self.baseline_mouth_height) / self.baseline_mouth_height
        
        # Determine if smiling
        is_smiling = (width_change > self.smile_width_threshold and 
                     height_change > self.smile_height_threshold)
        
        # Calculate intensity (0.0 to 1.0)
        intensity = 0.0
        if is_smiling:
            # Combine width and height changes for intensity
            width_intensity = min(1.0, (width_change - self.smile_width_threshold) / (2 * self.smile_width_threshold))
            height_intensity = min(1.0, (height_change - self.smile_height_threshold) / (2 * self.smile_height_threshold))
            intensity = (width_intensity + height_intensity) / 2.0
        
        # Calculate confidence based on consistency
        confidence = 0.0
        if is_smiling:
            # Higher confidence if both width and height increase significantly
            width_confidence = min(1.0, width_change / (2 * self.smile_width_threshold))
            height_confidence = min(1.0, height_change / (2 * self.smile_height_threshold))
            confidence = (width_confidence + height_confidence) / 2.0
        else:
            # Low confidence if not smiling
            confidence = 0.1
        
        return {
            'is_smiling': is_smiling,
            'intensity': intensity,
            'confidence': confidence,
            'width_change': width_change,
            'height_change': height_change,
            'current_width': smoothed_width,
            'current_height': smoothed_height,
            'baseline_width': self.baseline_mouth_width,
            'baseline_height': self.baseline_mouth_height
        }
    
    def get_smile_type(self, analysis_result: Dict) -> str:
        """
        Get smile type description
        
        Args:
            analysis_result: Result from analyze() method
            
        Returns:
            str: Smile type description
        """
        if not analysis_result['is_smiling']:
            return "neutral"
        
        intensity = analysis_result['intensity']
        if intensity > 0.8:
            return "big_smile"
        elif intensity > 0.6:
            return "wide_smile"
        elif intensity > 0.4:
            return "moderate_smile"
        elif intensity > 0.2:
            return "slight_smile"
        else:
            return "hint_of_smile"
    
    def get_smile_emotion(self, analysis_result: Dict) -> str:
        """
        Get emotion based on smile analysis
        
        Args:
            analysis_result: Result from analyze() method
            
        Returns:
            str: Emotion description
        """
        if not analysis_result['is_smiling']:
            return "neutral"
        
        intensity = analysis_result['intensity']
        confidence = analysis_result['confidence']
        
        if intensity > 0.7 and confidence > 0.7:
            return "happy"
        elif intensity > 0.5 and confidence > 0.5:
            return "pleased"
        elif intensity > 0.3 and confidence > 0.3:
            return "content"
        else:
            return "slightly_positive"
    
    def get_mouth_expression(self, analysis_result: Dict) -> str:
        """
        Get overall mouth expression
        
        Args:
            analysis_result: Result from analyze() method
            
        Returns:
            str: Mouth expression description
        """
        if analysis_result['is_smiling']:
            return self.get_smile_type(analysis_result)
        
        # Check for other expressions
        width_change = analysis_result['width_change']
        height_change = analysis_result['height_change']
        
        if width_change < -0.1:  # Mouth getting narrower
            return "pursed"
        elif height_change > 0.1:  # Mouth getting taller
            return "open"
        elif height_change < -0.1:  # Mouth getting shorter
            return "compressed"
        else:
            return "neutral"
    
    def reset_calibration(self) -> None:
        """Reset calibration and clear history"""
        self.baseline_mouth_width = 0.0
        self.baseline_mouth_height = 0.0
        self.is_calibrated = False
        self.width_history.clear()
        self.height_history.clear()
        print("Smile analyzer calibration reset")
    
    def set_thresholds(self, width_threshold: float = 0.15, height_threshold: float = 0.10) -> None:
        """
        Set custom thresholds for smile detection
        
        Args:
            width_threshold: Threshold for detecting smile width increase (0.0 to 1.0)
            height_threshold: Threshold for detecting smile height increase (0.0 to 1.0)
        """
        self.smile_width_threshold = max(0.0, min(1.0, width_threshold))
        self.smile_height_threshold = max(0.0, min(1.0, height_threshold))
        print(f"Smile thresholds set - width: {self.smile_width_threshold:.2f}, height: {self.smile_height_threshold:.2f}")
