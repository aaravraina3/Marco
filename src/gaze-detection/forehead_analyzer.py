"""
Forehead Analyzer - Analyzes forehead movement and expressions
"""

from typing import Dict, List, Optional


class ForeheadAnalyzer:
    """
    Analyzes forehead movement and expressions
    """
    
    def __init__(self):
        """Initialize the forehead analyzer"""
        self.baseline_forehead_height = 0.0
        self.baseline_forehead_width = 0.0
        self.is_calibrated = False
        
        # Thresholds for forehead movement detection
        self.raise_threshold = 0.15  # 15% increase from baseline
        self.lower_threshold = 0.15  # 15% decrease from baseline
        
        # History for smoothing
        self.height_history: List[float] = []
        self.max_history_size = 10
    
    def calibrate(self, baseline_measurements: Dict) -> None:
        """
        Calibrate the analyzer with baseline measurements
        
        Args:
            baseline_measurements: Dictionary containing baseline measurements
        """
        if 'forehead_height' in baseline_measurements:
            self.baseline_forehead_height = baseline_measurements['forehead_height']
            self.is_calibrated = True
            print(f"Forehead analyzer calibrated - baseline height: {self.baseline_forehead_height:.2f}")
        else:
            print("Warning: No forehead height in baseline measurements")
    
    def analyze(self, current_measurements: Dict) -> Dict:
        """
        Analyze current forehead measurements
        
        Args:
            current_measurements: Dictionary containing current measurements
            
        Returns:
            Dict: Analysis results
        """
        if not self.is_calibrated:
            return {
                'raised': False,
                'lowered': False,
                'intensity': 0.0,
                'height_change': 0.0
            }
        
        current_height = current_measurements.get('forehead_height', self.baseline_forehead_height)
        
        # Add to history for smoothing
        self.height_history.append(current_height)
        if len(self.height_history) > self.max_history_size:
            self.height_history.pop(0)
        
        # Calculate smoothed height
        smoothed_height = sum(self.height_history) / len(self.height_history)
        
        # Calculate height change percentage
        height_change = (smoothed_height - self.baseline_forehead_height) / self.baseline_forehead_height
        
        # Determine movement
        raised = height_change > self.raise_threshold
        lowered = height_change < -self.lower_threshold
        
        # Calculate intensity (0.0 to 1.0)
        intensity = 0.0
        if raised:
            intensity = min(1.0, (height_change - self.raise_threshold) / (2 * self.raise_threshold))
        elif lowered:
            intensity = min(1.0, (-height_change - self.lower_threshold) / (2 * self.lower_threshold))
        
        return {
            'raised': raised,
            'lowered': lowered,
            'intensity': intensity,
            'height_change': height_change,
            'current_height': smoothed_height,
            'baseline_height': self.baseline_forehead_height
        }
    
    def get_forehead_expression(self, analysis_result: Dict) -> str:
        """
        Get forehead expression description
        
        Args:
            analysis_result: Result from analyze() method
            
        Returns:
            str: Expression description
        """
        if analysis_result['raised']:
            if analysis_result['intensity'] > 0.7:
                return "highly_raised"
            elif analysis_result['intensity'] > 0.4:
                return "moderately_raised"
            else:
                return "slightly_raised"
        elif analysis_result['lowered']:
            if analysis_result['intensity'] > 0.7:
                return "highly_lowered"
            elif analysis_result['intensity'] > 0.4:
                return "moderately_lowered"
            else:
                return "slightly_lowered"
        else:
            return "neutral"
    
    def reset_calibration(self) -> None:
        """Reset calibration and clear history"""
        self.baseline_forehead_height = 0.0
        self.baseline_forehead_width = 0.0
        self.is_calibrated = False
        self.height_history.clear()
        print("Forehead analyzer calibration reset")
    
    def set_thresholds(self, raise_threshold: float = 0.15, lower_threshold: float = 0.15) -> None:
        """
        Set custom thresholds for forehead movement detection
        
        Args:
            raise_threshold: Threshold for detecting raised forehead (0.0 to 1.0)
            lower_threshold: Threshold for detecting lowered forehead (0.0 to 1.0)
        """
        self.raise_threshold = max(0.0, min(1.0, raise_threshold))
        self.lower_threshold = max(0.0, min(1.0, lower_threshold))
        print(f"Forehead thresholds set - raise: {self.raise_threshold:.2f}, lower: {self.lower_threshold:.2f}")
