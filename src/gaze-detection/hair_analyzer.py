"""
Hair Analyzer - Analyzes head movement, tilts, nods, and turns
"""

from typing import Dict, List, Optional
import math


class HairAnalyzer:
    """
    Analyzes head movement, tilts, nods, and turns
    """
    
    def __init__(self):
        """Initialize the hair analyzer"""
        self.baseline_head_tilt = 0.0
        self.baseline_head_turn = 0.0
        self.baseline_head_nod = 0.0
        self.is_calibrated = False
        
        # Thresholds for head movement detection (in degrees)
        self.tilt_threshold = 5.0  # 5 degrees
        self.turn_threshold = 10.0  # 10 degrees
        self.nod_threshold = 8.0  # 8 degrees
        
        # History for smoothing
        self.tilt_history: List[float] = []
        self.turn_history: List[float] = []
        self.nod_history: List[float] = []
        self.max_history_size = 10
    
    def calibrate(self, baseline_measurements: Dict) -> None:
        """
        Calibrate the analyzer with baseline measurements
        
        Args:
            baseline_measurements: Dictionary containing baseline measurements
        """
        if all(key in baseline_measurements for key in ['head_tilt', 'head_turn', 'head_nod']):
            self.baseline_head_tilt = baseline_measurements['head_tilt']
            self.baseline_head_turn = baseline_measurements['head_turn']
            self.baseline_head_nod = baseline_measurements['head_nod']
            self.is_calibrated = True
            print(f"Hair analyzer calibrated - baseline tilt: {self.baseline_head_tilt:.2f}°, turn: {self.baseline_head_turn:.2f}°, nod: {self.baseline_head_nod:.2f}°")
        else:
            print("Warning: Missing head pose measurements in baseline measurements")
    
    def analyze(self, current_measurements: Dict) -> Dict:
        """
        Analyze current head pose measurements
        
        Args:
            current_measurements: Dictionary containing current measurements
            
        Returns:
            Dict: Analysis results
        """
        if not self.is_calibrated:
            return {
                'tilt_left': False,
                'tilt_right': False,
                'nod_up': False,
                'nod_down': False,
                'turn_left': False,
                'turn_right': False,
                'tilt_angle': 0.0,
                'turn_angle': 0.0,
                'nod_angle': 0.0
            }
        
        current_tilt = current_measurements.get('head_tilt', self.baseline_head_tilt)
        current_turn = current_measurements.get('head_turn', self.baseline_head_turn)
        current_nod = current_measurements.get('head_nod', self.baseline_head_nod)
        
        # Add to history for smoothing
        self.tilt_history.append(current_tilt)
        self.turn_history.append(current_turn)
        self.nod_history.append(current_nod)
        if len(self.tilt_history) > self.max_history_size:
            self.tilt_history.pop(0)
        if len(self.turn_history) > self.max_history_size:
            self.turn_history.pop(0)
        if len(self.nod_history) > self.max_history_size:
            self.nod_history.pop(0)
        
        # Calculate smoothed angles
        smoothed_tilt = sum(self.tilt_history) / len(self.tilt_history)
        smoothed_turn = sum(self.turn_history) / len(self.turn_history)
        smoothed_nod = sum(self.nod_history) / len(self.nod_history)
        
        # Calculate angle differences from baseline
        tilt_diff = smoothed_tilt - self.baseline_head_tilt
        turn_diff = smoothed_turn - self.baseline_head_turn
        nod_diff = smoothed_nod - self.baseline_head_nod
        
        # Determine head movements
        tilt_left = tilt_diff < -self.tilt_threshold
        tilt_right = tilt_diff > self.tilt_threshold
        
        nod_up = nod_diff < -self.nod_threshold
        nod_down = nod_diff > self.nod_threshold
        
        turn_left = turn_diff < -self.turn_threshold
        turn_right = turn_diff > self.turn_threshold
        
        return {
            'tilt_left': tilt_left,
            'tilt_right': tilt_right,
            'nod_up': nod_up,
            'nod_down': nod_down,
            'turn_left': turn_left,
            'turn_right': turn_right,
            'tilt_angle': smoothed_tilt,
            'turn_angle': smoothed_turn,
            'nod_angle': smoothed_nod,
            'tilt_diff': tilt_diff,
            'turn_diff': turn_diff,
            'nod_diff': nod_diff,
            'baseline_tilt': self.baseline_head_tilt,
            'baseline_turn': self.baseline_head_turn,
            'baseline_nod': self.baseline_head_nod
        }
    
    def get_head_movement_type(self, analysis_result: Dict) -> str:
        """
        Get head movement type description
        
        Args:
            analysis_result: Result from analyze() method
            
        Returns:
            str: Head movement type description
        """
        movements = []
        
        if analysis_result['tilt_left']:
            movements.append("tilt_left")
        elif analysis_result['tilt_right']:
            movements.append("tilt_right")
        
        if analysis_result['nod_up']:
            movements.append("nod_up")
        elif analysis_result['nod_down']:
            movements.append("nod_down")
        
        if analysis_result['turn_left']:
            movements.append("turn_left")
        elif analysis_result['turn_right']:
            movements.append("turn_right")
        
        if not movements:
            return "stable"
        elif len(movements) == 1:
            return movements[0]
        else:
            return "complex_movement"
    
    def get_head_emotion(self, analysis_result: Dict) -> str:
        """
        Get emotion based on head movement
        
        Args:
            analysis_result: Result from analyze() method
            
        Returns:
            str: Emotion description
        """
        if analysis_result['nod_up']:
            return "confident"
        elif analysis_result['nod_down']:
            return "submissive"
        elif analysis_result['tilt_left'] or analysis_result['tilt_right']:
            return "curious"
        elif analysis_result['turn_left'] or analysis_result['turn_right']:
            return "distracted"
        else:
            return "focused"
    
    def get_head_attention(self, analysis_result: Dict) -> str:
        """
        Get attention level based on head movement
        
        Args:
            analysis_result: Result from analyze() method
            
        Returns:
            str: Attention level description
        """
        if analysis_result['turn_left'] or analysis_result['turn_right']:
            return "distracted"
        elif analysis_result['tilt_left'] or analysis_result['tilt_right']:
            return "questioning"
        elif analysis_result['nod_up']:
            return "engaged"
        elif analysis_result['nod_down']:
            return "focused"
        else:
            return "attentive"
    
    def get_movement_intensity(self, analysis_result: Dict) -> float:
        """
        Get movement intensity (0.0 to 1.0)
        
        Args:
            analysis_result: Result from analyze() method
            
        Returns:
            float: Movement intensity
        """
        max_intensity = 0.0
        
        # Calculate intensity for each movement type
        if analysis_result['tilt_left'] or analysis_result['tilt_right']:
            tilt_intensity = min(1.0, abs(analysis_result['tilt_diff']) / (2 * self.tilt_threshold))
            max_intensity = max(max_intensity, tilt_intensity)
        
        if analysis_result['nod_up'] or analysis_result['nod_down']:
            nod_intensity = min(1.0, abs(analysis_result['nod_diff']) / (2 * self.nod_threshold))
            max_intensity = max(max_intensity, nod_intensity)
        
        if analysis_result['turn_left'] or analysis_result['turn_right']:
            turn_intensity = min(1.0, abs(analysis_result['turn_diff']) / (2 * self.turn_threshold))
            max_intensity = max(max_intensity, turn_intensity)
        
        return max_intensity
    
    def reset_calibration(self) -> None:
        """Reset calibration and clear history"""
        self.baseline_head_tilt = 0.0
        self.baseline_head_turn = 0.0
        self.baseline_head_nod = 0.0
        self.is_calibrated = False
        self.tilt_history.clear()
        self.turn_history.clear()
        self.nod_history.clear()
        print("Hair analyzer calibration reset")
    
    def set_thresholds(self, tilt_threshold: float = 5.0, turn_threshold: float = 10.0, nod_threshold: float = 8.0) -> None:
        """
        Set custom thresholds for head movement detection
        
        Args:
            tilt_threshold: Threshold for detecting head tilt (in degrees)
            turn_threshold: Threshold for detecting head turn (in degrees)
            nod_threshold: Threshold for detecting head nod (in degrees)
        """
        self.tilt_threshold = max(0.0, tilt_threshold)
        self.turn_threshold = max(0.0, turn_threshold)
        self.nod_threshold = max(0.0, nod_threshold)
        print(f"Head movement thresholds set - tilt: {self.tilt_threshold:.1f}°, turn: {self.turn_threshold:.1f}°, nod: {self.nod_threshold:.1f}°")
