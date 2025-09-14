import logging
import numpy as np
from typing import Dict, List, Any
import jiwer
from ..models.asr import ASRResult
from ..analysis.preprocess import text_normalizer
from ..settings import settings

logger = logging.getLogger(__name__)


class ClarityAnalyzer:
    """Analyze speech clarity based on ASR confidence and consistency."""
    
    def __init__(self):
        self.confidence_threshold = settings.analysis.clarity.get("confidence_threshold", 0.8)
    
    def analyze(self, asr_result: ASRResult) -> Dict[str, Any]:
        """Analyze speech clarity metrics."""
        try:
            # ASR confidence-based metrics
            confidence_metrics = self._analyze_confidence(asr_result)
            
            # Word Error Rate approximation (self-consistency)
            consistency_metrics = self._analyze_consistency(asr_result)
            
            # Combine into overall clarity score
            clarity_score = self._calculate_overall_clarity(confidence_metrics, consistency_metrics)
            
            # Generate assessment and recommendations
            assessment = self._assess_clarity(clarity_score)
            recommendations = self._get_recommendations(clarity_score, confidence_metrics, consistency_metrics)
            
            return {
                "clarity_score": clarity_score,
                "confidence_metrics": confidence_metrics,
                "consistency_metrics": consistency_metrics,
                "assessment": assessment,
                "recommendations": recommendations,
                "unclear_words": self._find_unclear_words(asr_result)
            }
            
        except Exception as e:
            logger.error(f"Error in clarity analysis: {e}")
            return {
                "clarity_score": 0,
                "assessment": "unknown",
                "error": str(e)
            }
    
    def _analyze_confidence(self, asr_result: ASRResult) -> Dict[str, Any]:
        """Analyze ASR confidence scores."""
        words = asr_result.words
        if not words:
            return {
                "mean_confidence": 0.0,
                "min_confidence": 0.0,
                "low_confidence_count": 0,
                "low_confidence_percentage": 0.0
            }
        
        confidences = [w.get("confidence", 0.0) for w in words]
        low_confidence_words = [c for c in confidences if c < self.confidence_threshold]
        
        return {
            "mean_confidence": round(np.mean(confidences), 3),
            "min_confidence": round(np.min(confidences), 3),
            "std_confidence": round(np.std(confidences), 3),
            "low_confidence_count": len(low_confidence_words),
            "low_confidence_percentage": round(len(low_confidence_words) / len(words) * 100, 1)
        }
    
    def _analyze_consistency(self, asr_result: ASRResult) -> Dict[str, Any]:
        """Analyze speech consistency using text normalization variants."""
        original_text = asr_result.text
        
        if not original_text.strip():
            return {
                "consistency_score": 0.0,
                "normalized_wer": 0.0
            }
        
        try:
            # Create normalized versions
            normalized = text_normalizer.normalize_for_comparison(original_text)
            no_fillers = text_normalizer.remove_fillers(original_text)
            no_fillers_normalized = text_normalizer.normalize_for_comparison(no_fillers)
            
            # Calculate "self-WER" between versions
            if normalized and no_fillers_normalized:
                wer_score = jiwer.wer(normalized, no_fillers_normalized)
                # Convert WER to consistency (inverse relationship)
                consistency_score = max(0, 100 - (wer_score * 100))
            else:
                wer_score = 0.0
                consistency_score = 100.0
            
            return {
                "consistency_score": round(consistency_score, 1),
                "normalized_wer": round(wer_score, 3),
                "original_length": len(original_text),
                "normalized_length": len(normalized)
            }
            
        except Exception as e:
            logger.warning(f"Error in consistency analysis: {e}")
            return {
                "consistency_score": 50.0,
                "normalized_wer": 0.0
            }
    
    def _calculate_overall_clarity(self, 
                                 confidence_metrics: Dict[str, Any], 
                                 consistency_metrics: Dict[str, Any]) -> int:
        """Calculate overall clarity score (0-100)."""
        # Weight factors
        confidence_weight = 0.7
        consistency_weight = 0.3
        
        # Confidence component (scale 0-100)
        confidence_component = confidence_metrics.get("mean_confidence", 0.0) * 100
        
        # Consistency component (already 0-100)
        consistency_component = consistency_metrics.get("consistency_score", 0.0)
        
        # Weighted average
        overall_score = (
            confidence_component * confidence_weight + 
            consistency_component * consistency_weight
        )
        
        return round(overall_score)
    
    def _assess_clarity(self, clarity_score: int) -> str:
        """Assess clarity level."""
        if clarity_score >= 90:
            return "excellent"
        elif clarity_score >= 80:
            return "good"
        elif clarity_score >= 70:
            return "fair"
        elif clarity_score >= 60:
            return "poor"
        else:
            return "very_poor"
    
    def _find_unclear_words(self, asr_result: ASRResult) -> List[Dict[str, Any]]:
        """Find words with low confidence scores."""
        unclear_words = []
        
        for word in asr_result.words:
            confidence = word.get("confidence", 0.0)
            if confidence < self.confidence_threshold:
                unclear_words.append({
                    "word": word["word"],
                    "confidence": round(confidence, 3),
                    "start": word.get("start", 0.0),
                    "end": word.get("end", 0.0)
                })
        
        # Sort by lowest confidence first
        unclear_words.sort(key=lambda x: x["confidence"])
        
        return unclear_words
    
    def _get_recommendations(self, 
                           clarity_score: int,
                           confidence_metrics: Dict[str, Any],
                           consistency_metrics: Dict[str, Any]) -> List[str]:
        """Generate clarity improvement recommendations."""
        recommendations = []
        
        # Overall recommendations
        if clarity_score >= 90:
            recommendations.append("Excellent clarity! Your speech is very clear and easy to understand.")
        elif clarity_score >= 80:
            recommendations.append("Good clarity overall. Minor improvements could make your speech even clearer.")
        elif clarity_score >= 70:
            recommendations.append("Fair clarity. Focus on articulation and speaking pace to improve understanding.")
        elif clarity_score >= 60:
            recommendations.append("Poor clarity. Work on speaking more slowly and clearly enunciating words.")
        else:
            recommendations.append("Very poor clarity. Significant improvement needed in articulation and pace.")
        
        # Confidence-specific recommendations
        low_conf_pct = confidence_metrics.get("low_confidence_percentage", 0)
        if low_conf_pct > 20:
            recommendations.append(
                f"{low_conf_pct:.1f}% of your words were unclear. "
                "Focus on speaking more distinctly and at a moderate pace."
            )
        elif low_conf_pct > 10:
            recommendations.append(
                f"{low_conf_pct:.1f}% of your words were unclear. "
                "Pay attention to articulating consonants clearly."
            )
        
        # Consistency recommendations
        consistency_score = consistency_metrics.get("consistency_score", 0)
        if consistency_score < 80:
            recommendations.append(
                "Your speech patterns are inconsistent. "
                "Practice maintaining steady rhythm and clear pronunciation."
            )
        
        # Technical recommendations
        mean_confidence = confidence_metrics.get("mean_confidence", 0)
        if mean_confidence < 0.7:
            recommendations.append(
                "Consider speaking closer to the microphone or in a quieter environment. "
                "Background noise may be affecting clarity detection."
            )
        
        return recommendations
    
    def get_clarity_timeline(self, asr_result: ASRResult, window_size: float = 5.0) -> List[Dict[str, Any]]:
        """Generate clarity timeline for visualization."""
        words = asr_result.words
        if not words:
            return []
        
        timeline = []
        current_window = []
        window_start = words[0].get("start", 0.0)
        
        for word in words:
            word_time = word.get("start", 0.0)
            
            # Check if we need to start a new window
            if word_time - window_start >= window_size:
                if current_window:
                    # Calculate window metrics
                    confidences = [w.get("confidence", 0.0) for w in current_window]
                    avg_confidence = np.mean(confidences)
                    
                    timeline.append({
                        "start": window_start,
                        "end": word_time,
                        "confidence": round(avg_confidence, 3),
                        "word_count": len(current_window)
                    })
                
                current_window = []
                window_start = word_time
            
            current_window.append(word)
        
        # Add final window
        if current_window:
            confidences = [w.get("confidence", 0.0) for w in current_window]
            avg_confidence = np.mean(confidences)
            
            timeline.append({
                "start": window_start,
                "end": current_window[-1].get("end", window_start),
                "confidence": round(avg_confidence, 3),
                "word_count": len(current_window)
            })
        
        return timeline


# Global clarity analyzer instance
clarity_analyzer = ClarityAnalyzer()