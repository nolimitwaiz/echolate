import numpy as np
import logging
from typing import Dict, List, Any, Tuple
from ..models.asr import ASRResult
from ..models.vad import VADSegment
from ..settings import settings

logger = logging.getLogger(__name__)


class PacingAnalyzer:
    """Analyze speech pacing and rhythm."""
    
    def __init__(self):
        self.config = settings.analysis.wpm
        self.slow_threshold = self.config.get("slow_threshold", 120)
        self.fast_threshold = self.config.get("fast_threshold", 180)
        self.target_range = self.config.get("target_range", [140, 160])
    
    def analyze(self, asr_result: ASRResult, vad_segments: List[VADSegment]) -> Dict[str, Any]:
        """Analyze speech pacing metrics."""
        try:
            # Calculate basic WPM
            wpm_stats = self._calculate_wpm(asr_result, vad_segments)
            
            # Analyze pacing stability
            stability_metrics = self._analyze_stability(asr_result)
            
            # Get pacing assessment
            assessment = self._assess_pacing(wpm_stats["overall_wpm"])
            
            return {
                **wpm_stats,
                **stability_metrics,
                "assessment": assessment,
                "target_range": self.target_range,
                "recommendations": self._get_recommendations(wpm_stats["overall_wpm"], stability_metrics)
            }
            
        except Exception as e:
            logger.error(f"Error in pacing analysis: {e}")
            return {
                "overall_wpm": 0,
                "speech_duration": 0,
                "word_count": 0,
                "assessment": "unknown",
                "error": str(e)
            }
    
    def _calculate_wpm(self, asr_result: ASRResult, vad_segments: List[VADSegment]) -> Dict[str, Any]:
        """Calculate words per minute metrics."""
        # Count words (excluding fillers for clean WPM)
        words = asr_result.words
        clean_words = [w for w in words if not self._is_filler_word(w["word"])]
        
        # Calculate total speaking time from VAD
        speech_segments = [seg for seg in vad_segments if seg.is_speech]
        total_speech_duration = sum(seg.duration for seg in speech_segments)
        
        if total_speech_duration == 0:
            return {
                "overall_wpm": 0,
                "clean_wpm": 0,
                "speech_duration": 0,
                "word_count": len(words),
                "clean_word_count": len(clean_words)
            }
        
        # Calculate WPM
        overall_wpm = (len(words) / total_speech_duration) * 60
        clean_wpm = (len(clean_words) / total_speech_duration) * 60
        
        return {
            "overall_wpm": round(overall_wpm, 1),
            "clean_wpm": round(clean_wpm, 1),
            "speech_duration": round(total_speech_duration, 2),
            "word_count": len(words),
            "clean_word_count": len(clean_words)
        }
    
    def _analyze_stability(self, asr_result: ASRResult, window_size: float = 10.0) -> Dict[str, Any]:
        """Analyze pacing stability over time using sliding windows."""
        words = asr_result.words
        if len(words) < 10:  # Need minimum words for stability analysis
            return {
                "stability_score": 100.0,
                "wpm_variance": 0.0,
                "wpm_timeline": [],
                "stability_assessment": "insufficient_data"
            }
        
        # Create sliding windows
        window_wpm = []
        window_times = []
        
        for i in range(0, len(words) - 5, 5):  # Step by 5 words
            window_words = words[i:i+10]  # 10 word window
            if len(window_words) < 10:
                continue
            
            start_time = window_words[0]["start"]
            end_time = window_words[-1]["end"]
            duration = end_time - start_time
            
            if duration > 0:
                wpm = (len(window_words) / duration) * 60
                window_wpm.append(wpm)
                window_times.append((start_time + end_time) / 2)
        
        if len(window_wpm) < 3:
            return {
                "stability_score": 100.0,
                "wpm_variance": 0.0,
                "wpm_timeline": [],
                "stability_assessment": "insufficient_data"
            }
        
        # Calculate stability metrics
        wpm_variance = np.var(window_wpm)
        wpm_std = np.std(window_wpm)
        mean_wpm = np.mean(window_wpm)
        
        # Stability score (0-100, higher is more stable)
        cv = (wpm_std / mean_wpm) if mean_wpm > 0 else 0  # Coefficient of variation
        stability_score = max(0, 100 - (cv * 100))
        
        # Assessment
        if cv < 0.15:
            stability_assessment = "very_stable"
        elif cv < 0.25:
            stability_assessment = "stable"
        elif cv < 0.35:
            stability_assessment = "moderately_stable"
        else:
            stability_assessment = "unstable"
        
        return {
            "stability_score": round(stability_score, 1),
            "wpm_variance": round(wpm_variance, 1),
            "wpm_std": round(wpm_std, 1),
            "coefficient_variation": round(cv, 3),
            "wpm_timeline": list(zip(window_times, window_wpm)),
            "stability_assessment": stability_assessment
        }
    
    def _assess_pacing(self, wpm: float) -> str:
        """Assess overall pacing."""
        if wpm < self.slow_threshold:
            return "too_slow"
        elif wpm > self.fast_threshold:
            return "too_fast"
        elif self.target_range[0] <= wpm <= self.target_range[1]:
            return "optimal"
        elif wpm < self.target_range[0]:
            return "slightly_slow"
        else:
            return "slightly_fast"
    
    def _get_recommendations(self, wpm: float, stability_metrics: Dict[str, Any]) -> List[str]:
        """Generate pacing recommendations."""
        recommendations = []
        
        # WPM recommendations
        if wpm < self.slow_threshold:
            recommendations.append(
                f"Your speaking pace is quite slow ({wpm} WPM). Try to speak more briskly, "
                f"aiming for {self.target_range[0]}-{self.target_range[1]} WPM."
            )
        elif wpm > self.fast_threshold:
            recommendations.append(
                f"Your speaking pace is quite fast ({wpm} WPM). Slow down slightly to "
                f"improve clarity, aiming for {self.target_range[0]}-{self.target_range[1]} WPM."
            )
        elif wpm < self.target_range[0]:
            recommendations.append(
                f"Your pace is slightly slow ({wpm} WPM). A bit more energy would help reach "
                f"the optimal range of {self.target_range[0]}-{self.target_range[1]} WPM."
            )
        elif wpm > self.target_range[1]:
            recommendations.append(
                f"Your pace is slightly fast ({wpm} WPM). Slowing down just a bit will "
                f"improve clarity and keep you in the optimal {self.target_range[0]}-{self.target_range[1]} WPM range."
            )
        else:
            recommendations.append(
                f"Excellent pacing! Your {wpm} WPM is right in the optimal range."
            )
        
        # Stability recommendations
        stability_assessment = stability_metrics.get("stability_assessment", "unknown")
        if stability_assessment == "unstable":
            recommendations.append(
                "Your pacing varies significantly throughout your speech. Practice maintaining "
                "a more consistent rhythm."
            )
        elif stability_assessment == "moderately_stable":
            recommendations.append(
                "Work on maintaining more consistent pacing throughout your speech."
            )
        elif stability_assessment in ["stable", "very_stable"]:
            recommendations.append("Great job maintaining consistent pacing!")
        
        return recommendations
    
    def _is_filler_word(self, word: str) -> bool:
        """Check if a word is a filler word."""
        import re
        
        word_lower = word.lower().strip()
        filler_patterns = settings.fillers.get("patterns", [])
        
        for pattern in filler_patterns:
            if re.match(pattern, word_lower):
                return True
        
        return False


# Global pacing analyzer instance
pacing_analyzer = PacingAnalyzer()