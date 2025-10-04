"""
Eyebrow Analyzer - Analyzes eyebrow movement and expressions
"""

from typing import Dict, List, Optional


class EyebrowAnalyzer:
    """
    Analyzes eyebrow movement and expressions
    """
    
    def __init__(self):
        """Initialize the eyebrow analyzer"""
        self.baseline_eyebrow_height = 0.0
        self.baseline_eyebrow_width = 0.0
        self.is_calibrated = False
        
        # Thresholds for eyebrow movement detection
        self.raise_threshold = 0.12  # 12% increase from baseline
        self.lower_threshold = 0.12  # 12% decrease from baseline
        
        # History for smoothing
        self.height_history: List[float] = []
        self.left_height_history: List[float] = []
        self.right_height_history: List[float] = []
        self.max_history_size = 10
    
    def calibrate(self, baseline_measurements: Dict) -> None:
        """
        Calibrate the analyzer with baseline measurements
        
        Args:
            baseline_measurements: Dictionary containing baseline measurements
        """
        if 'eyebrow_height' in baseline_measurements:
            self.baseline_eyebrow_height = baseline_measurements['eyebrow_height']
            self.is_calibrated = True
            print(f"Eyebrow analyzer calibrated - baseline height: {self.baseline_eyebrow_height:.2f}")
        else:
            print("Warning: No eyebrow height in baseline measurements")
    
    def analyze(self, current_measurements: Dict) -> Dict:
        """
        Analyze current eyebrow measurements
        
        Args:
            current_measurements: Dictionary containing current measurements
            
        Returns:
            Dict: Analysis results
        """
        if not self.is_calibrated:
            return {
                'left_raised': False,
                'right_raised': False,
                'both_raised': False,
                'intensity': 0.0,
                'height_change': 0.0,
                'asymmetry': 0.0
            }
        
        current_height = current_measurements.get('eyebrow_height', self.baseline_eyebrow_height)
        
        # Add to history for smoothing
        self.height_history.append(current_height)
        if len(self.height_history) > self.max_history_size:
            self.height_history.pop(0)
        
        # Calculate smoothed height
        smoothed_height = sum(self.height_history) / len(self.height_history)
        
        # Calculate height change percentage
        height_change = (smoothed_height - self.baseline_eyebrow_height) / self.baseline_eyebrow_height
        
        # Determine movement
        raised = height_change > self.raise_threshold
        lowered = height_change < -self.lower_threshold
        
        # Calculate intensity (0.0 to 1.0)
        intensity = 0.0
        if raised:
            intensity = min(1.0, (height_change - self.raise_threshold) / (2 * self.raise_threshold))
        elif lowered:
            intensity = min(1.0, (-height_change - self.lower_threshold) / (2 * self.lower_threshold))
        
        # For now, assume both eyebrows move together
        # In a more advanced implementation, you'd track left and right separately
        left_raised = raised
        right_raised = raised
        both_raised = raised
        
        # Calculate asymmetry (difference between left and right)
        # For now, assume no asymmetry
        asymmetry = 0.0
        
        return {
            'left_raised': left_raised,
            'right_raised': right_raised,
            'both_raised': both_raised,
            'intensity': intensity,
            'height_change': height_change,
            'asymmetry': asymmetry,
            'current_height': smoothed_height,
            'baseline_height': self.baseline_eyebrow_height
        }
    
    def get_eyebrow_expression(self, analysis_result: Dict) -> str:
        """
        Get eyebrow expression description
        
        Args:
            analysis_result: Result from analyze() method
            
        Returns:
            str: Expression description
        """
        if analysis_result['both_raised']:
            if analysis_result['intensity'] > 0.7:
                return "highly_raised"
            elif analysis_result['intensity'] > 0.4:
                return "moderately_raised"
            else:
                return "slightly_raised"
        elif analysis_result['height_change'] < -self.lower_threshold:
            if analysis_result['intensity'] > 0.7:
                return "highly_lowered"
            elif analysis_result['intensity'] > 0.4:
                return "moderately_lowered"
            else:
                return "slightly_lowered"
        else:
            return "neutral"
    
    def get_eyebrow_emotion(self, analysis_result: Dict) -> str:
        """
        Get emotion based on eyebrow movement
        
        Args:
            analysis_result: Result from analyze() method
            
        Returns:
            str: Emotion description
        """
        if analysis_result['both_raised']:
            if analysis_result['intensity'] > 0.6:
                return "surprised"
            elif analysis_result['intensity'] > 0.3:
                return "questioning"
            else:
                return "interested"
        elif analysis_result['height_change'] < -self.lower_threshold:
            if analysis_result['intensity'] > 0.6:
                return "concerned"
            elif analysis_result['intensity'] > 0.3:
                return "focused"
            else:
                return "concentrating"
        else:
            return "neutral"
    
    def reset_calibration(self) -> None:
        """Reset calibration and clear history"""
        self.baseline_eyebrow_height = 0.0
        self.baseline_eyebrow_width = 0.0
        self.is_calibrated = False
        self.height_history.clear()
        self.left_height_history.clear()
        self.right_height_history.clear()
        print("Eyebrow analyzer calibration reset")
    
    def set_thresholds(self, raise_threshold: float = 0.12, lower_threshold: float = 0.12) -> None:
        """
        Set custom thresholds for eyebrow movement detection
        
        Args:
            raise_threshold: Threshold for detecting raised eyebrows (0.0 to 1.0)
            lower_threshold: Threshold for detecting lowered eyebrows (0.0 to 1.0)
        """
        self.raise_threshold = max(0.0, min(1.0, raise_threshold))
        self.lower_threshold = max(0.0, min(1.0, lower_threshold))
        print(f"Eyebrow thresholds set - raise: {self.raise_threshold:.2f}, lower: {self.lower_threshold:.2f}")
