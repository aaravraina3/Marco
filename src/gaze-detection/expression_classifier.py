"""
Expression Classifier - Classifies overall facial expressions based on individual feature analysis
"""

from typing import Dict, List, Optional


class ExpressionClassifier:
    """
    Classifies overall facial expressions based on individual feature analysis
    """
    
    def __init__(self):
        """Initialize the expression classifier"""
        self.expression_history: List[str] = []
        self.max_history_size = 15
        
        # Expression weights for classification
        self.expression_weights = {
            'happy': {
                'smile_intensity': 0.6,
                'eyebrow_raise': 0.2,
                'forehead_raise': 0.1,
                'head_nod': 0.1
            },
            'surprised': {
                'eyebrow_raise': 0.5,
                'forehead_raise': 0.3,
                'smile_intensity': 0.1,
                'head_nod': 0.1
            },
            'confused': {
                'eyebrow_raise': 0.3,
                'head_tilt': 0.4,
                'smile_intensity': 0.0,
                'forehead_raise': 0.3
            },
            'focused': {
                'eyebrow_lower': 0.4,
                'forehead_lower': 0.3,
                'head_nod': 0.2,
                'smile_intensity': 0.1
            },
            'concerned': {
                'eyebrow_lower': 0.4,
                'forehead_lower': 0.3,
                'head_tilt': 0.2,
                'smile_intensity': 0.1
            },
            'neutral': {
                'smile_intensity': 0.0,
                'eyebrow_raise': 0.0,
                'forehead_raise': 0.0,
                'head_movement': 0.0
            }
        }
    
    def classify_expression(self, forehead_result: Dict, eyebrow_result: Dict, 
                          smile_result: Dict, hair_result: Dict) -> Dict:
        """
        Classify overall facial expression based on individual feature analysis
        
        Args:
            forehead_result: Result from ForeheadAnalyzer
            eyebrow_result: Result from EyebrowAnalyzer
            smile_result: Result from SmileAnalyzer
            hair_result: Result from HairAnalyzer
            
        Returns:
            Dict: Expression classification results
        """
        # Extract feature scores
        feature_scores = self._extract_feature_scores(forehead_result, eyebrow_result, smile_result, hair_result)
        
        # Calculate expression probabilities
        expression_scores = self._calculate_expression_scores(feature_scores)
        
        # Get the most likely expression
        best_expression = max(expression_scores.items(), key=lambda x: x[1])
        expression_type = best_expression[0]
        confidence = best_expression[1]
        
        # Add to history for smoothing
        self.expression_history.append(expression_type)
        if len(self.expression_history) > self.max_history_size:
            self.expression_history.pop(0)
        
        # Get smoothed expression
        smoothed_expression = self._get_smoothed_expression()
        
        return {
            'type': smoothed_expression,
            'confidence': confidence,
            'raw_type': expression_type,
            'scores': expression_scores,
            'feature_scores': feature_scores
        }
    
    def _extract_feature_scores(self, forehead_result: Dict, eyebrow_result: Dict, 
                               smile_result: Dict, hair_result: Dict) -> Dict:
        """
        Extract normalized feature scores from individual analyzer results
        
        Args:
            forehead_result: Result from ForeheadAnalyzer
            eyebrow_result: Result from EyebrowAnalyzer
            smile_result: Result from SmileAnalyzer
            hair_result: Result from HairAnalyzer
            
        Returns:
            Dict: Normalized feature scores
        """
        scores = {}
        
        # Smile intensity (0.0 to 1.0)
        scores['smile_intensity'] = smile_result.get('intensity', 0.0)
        
        # Eyebrow raise/lower
        eyebrow_intensity = eyebrow_result.get('intensity', 0.0)
        if eyebrow_result.get('both_raised', False):
            scores['eyebrow_raise'] = eyebrow_intensity
            scores['eyebrow_lower'] = 0.0
        elif eyebrow_result.get('height_change', 0.0) < 0:
            scores['eyebrow_raise'] = 0.0
            scores['eyebrow_lower'] = eyebrow_intensity
        else:
            scores['eyebrow_raise'] = 0.0
            scores['eyebrow_lower'] = 0.0
        
        # Forehead raise/lower
        forehead_intensity = forehead_result.get('intensity', 0.0)
        if forehead_result.get('raised', False):
            scores['forehead_raise'] = forehead_intensity
            scores['forehead_lower'] = 0.0
        elif forehead_result.get('lowered', False):
            scores['forehead_raise'] = 0.0
            scores['forehead_lower'] = forehead_intensity
        else:
            scores['forehead_raise'] = 0.0
            scores['forehead_lower'] = 0.0
        
        # Head movement
        head_movement_intensity = self._get_head_movement_intensity(hair_result)
        scores['head_movement'] = head_movement_intensity
        
        # Specific head movements
        scores['head_tilt'] = 1.0 if (hair_result.get('tilt_left', False) or hair_result.get('tilt_right', False)) else 0.0
        scores['head_nod'] = 1.0 if (hair_result.get('nod_up', False) or hair_result.get('nod_down', False)) else 0.0
        
        return scores
    
    def _get_head_movement_intensity(self, hair_result: Dict) -> float:
        """
        Calculate overall head movement intensity
        
        Args:
            hair_result: Result from HairAnalyzer
            
        Returns:
            float: Head movement intensity (0.0 to 1.0)
        """
        movement_count = 0
        if hair_result.get('tilt_left', False): movement_count += 1
        if hair_result.get('tilt_right', False): movement_count += 1
        if hair_result.get('nod_up', False): movement_count += 1
        if hair_result.get('nod_down', False): movement_count += 1
        if hair_result.get('turn_left', False): movement_count += 1
        if hair_result.get('turn_right', False): movement_count += 1
        
        return min(1.0, movement_count / 3.0)  # Normalize to 0.0-1.0
    
    def _calculate_expression_scores(self, feature_scores: Dict) -> Dict:
        """
        Calculate expression scores based on feature scores and weights
        
        Args:
            feature_scores: Normalized feature scores
            
        Returns:
            Dict: Expression scores
        """
        expression_scores = {}
        
        for expression, weights in self.expression_weights.items():
            score = 0.0
            total_weight = 0.0
            
            for feature, weight in weights.items():
                if feature in feature_scores:
                    score += feature_scores[feature] * weight
                    total_weight += weight
            
            # Normalize score
            if total_weight > 0:
                expression_scores[expression] = score / total_weight
            else:
                expression_scores[expression] = 0.0
        
        return expression_scores
    
    def _get_smoothed_expression(self) -> str:
        """
        Get smoothed expression based on recent history
        
        Returns:
            str: Smoothed expression type
        """
        if not self.expression_history:
            return "neutral"
        
        # Use the most common expression in recent history
        recent_expressions = self.expression_history[-5:]  # Last 5 expressions
        return max(set(recent_expressions), key=recent_expressions.count)
    
    def get_expression_description(self, expression_type: str) -> str:
        """
        Get human-readable description of expression
        
        Args:
            expression_type: Expression type string
            
        Returns:
            str: Human-readable description
        """
        descriptions = {
            'happy': "Happy and cheerful",
            'surprised': "Surprised or amazed",
            'confused': "Confused or puzzled",
            'focused': "Focused and concentrated",
            'concerned': "Concerned or worried",
            'neutral': "Neutral expression"
        }
        
        return descriptions.get(expression_type, "Unknown expression")
    
    def get_expression_confidence_level(self, confidence: float) -> str:
        """
        Get confidence level description
        
        Args:
            confidence: Confidence score (0.0 to 1.0)
            
        Returns:
            str: Confidence level description
        """
        if confidence > 0.8:
            return "very_high"
        elif confidence > 0.6:
            return "high"
        elif confidence > 0.4:
            return "medium"
        elif confidence > 0.2:
            return "low"
        else:
            return "very_low"
    
    def reset_history(self) -> None:
        """Reset expression history"""
        self.expression_history.clear()
        print("Expression classifier history reset")
    
    def set_expression_weights(self, expression: str, weights: Dict) -> None:
        """
        Set custom weights for an expression
        
        Args:
            expression: Expression name
            weights: Dictionary of feature weights
        """
        if expression in self.expression_weights:
            self.expression_weights[expression].update(weights)
            print(f"Updated weights for expression: {expression}")
        else:
            print(f"Unknown expression: {expression}")
    
    def add_custom_expression(self, expression: str, weights: Dict) -> None:
        """
        Add a custom expression with its weights
        
        Args:
            expression: Expression name
            weights: Dictionary of feature weights
        """
        self.expression_weights[expression] = weights
        print(f"Added custom expression: {expression}")
