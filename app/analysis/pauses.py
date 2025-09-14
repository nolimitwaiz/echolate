import logging
import numpy as np
import re
from typing import Dict, List, Any, Tuple, Optional

from ..models.asr import ASRResult
from ..models.vad import VADSegment
from ..analysis.fillers import filler_analyzer
from ..settings import settings

logger = logging.getLogger(__name__)


class PauseAnalyzer:
    """Analyze pause patterns and classify hesitation vs rhetorical pauses."""
    
    def __init__(self):
        self.config = settings.analysis.pauses
        self.min_pause_duration = self.config.get("min_pause_duration", 0.3)
        self.rhetorical_threshold = self.config.get("rhetorical_threshold", 1.0)
        self.punctuation_tolerance = self.config.get("punctuation_tolerance_ms", 250) / 1000  # Convert to seconds
        self.filler_tolerance = self.config.get("filler_tolerance_ms", 400) / 1000  # Convert to seconds
        
        # Filler patterns from settings
        self.filler_patterns = settings.fillers.get("patterns", [])
        self.compiled_filler_patterns = [(p, re.compile(p, re.IGNORECASE)) for p in self.filler_patterns]
    
    def analyze(self, vad_segments: List[VADSegment], asr_result: ASRResult) -> Dict[str, Any]:
        """Analyze pause patterns in speech."""
        try:
            # Extract pauses from VAD segments
            pauses = self._extract_pauses(vad_segments)
            
            # Classify each pause
            classified_pauses = self._classify_pauses(pauses, asr_result)
            
            # Calculate metrics
            metrics = self._calculate_metrics(classified_pauses)
            
            # Generate assessment and recommendations
            assessment = self._assess_pause_usage(metrics)
            recommendations = self._get_recommendations(metrics, classified_pauses)
            
            return {
                "pause_metrics": metrics,
                "classified_pauses": classified_pauses,
                "assessment": assessment,
                "recommendations": recommendations,
                "timeline_data": self._get_timeline_data(classified_pauses)
            }
            
        except Exception as e:
            logger.error(f"Error in pause analysis: {e}")
            return {
                "assessment": "error",
                "error": str(e)
            }
    
    def _extract_pauses(self, vad_segments: List[VADSegment]) -> List[Dict[str, Any]]:
        """Extract pauses from VAD segments."""
        pauses = []
        
        for segment in vad_segments:
            if not segment.is_speech and segment.duration >= self.min_pause_duration:
                pauses.append({
                    "start": segment.start,
                    "end": segment.end,
                    "duration": segment.duration
                })
        
        return pauses
    
    def _classify_pauses(self, pauses: List[Dict[str, Any]], asr_result: ASRResult) -> List[Dict[str, Any]]:
        """Classify pauses as hesitation or rhetorical."""
        if not pauses:
            return []
        
        # Build word timeline for context analysis
        word_timeline = self._build_word_timeline(asr_result)
        
        # Build punctuation timeline
        punctuation_timeline = self._build_punctuation_timeline(asr_result.text, word_timeline)
        
        # Find filler locations
        filler_locations = self._find_filler_locations(asr_result)
        
        classified_pauses = []
        
        for pause in pauses:
            classification = self._classify_single_pause(
                pause, word_timeline, punctuation_timeline, filler_locations
            )
            
            classified_pauses.append({
                **pause,
                "type": classification["type"],
                "confidence": classification["confidence"],
                "reason": classification["reason"],
                "context": classification.get("context", "")
            })
        
        return classified_pauses
    
    def _build_word_timeline(self, asr_result: ASRResult) -> List[Dict[str, Any]]:
        """Build timeline of words with positions."""
        timeline = []
        
        for i, word_info in enumerate(asr_result.words):
            timeline.append({
                "word": word_info["word"].strip(),
                "start": word_info.get("start", 0.0),
                "end": word_info.get("end", 0.0),
                "index": i,
                "confidence": word_info.get("confidence", 0.0)
            })
        
        return sorted(timeline, key=lambda x: x["start"])
    
    def _build_punctuation_timeline(self, transcript: str, word_timeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build timeline of punctuation marks."""
        punctuation_timeline = []
        
        # Find punctuation in transcript
        punctuation_marks = re.finditer(r'[,.;:!?]', transcript)
        
        for match in punctuation_marks:
            char_pos = match.start()
            punct_mark = match.group()
            
            # Estimate timing based on nearest words
            estimated_time = self._estimate_punctuation_time(char_pos, transcript, word_timeline)
            
            if estimated_time is not None:
                punctuation_timeline.append({
                    "mark": punct_mark,
                    "time": estimated_time,
                    "char_pos": char_pos
                })
        
        return sorted(punctuation_timeline, key=lambda x: x["time"])
    
    def _estimate_punctuation_time(self, char_pos: int, transcript: str, word_timeline: List[Dict[str, Any]]) -> Optional[float]:
        """Estimate timing for punctuation based on surrounding words."""
        if not word_timeline:
            return None
        
        # Find words before and after punctuation
        words_before = []
        words_after = []
        current_char_pos = 0
        
        for word_info in word_timeline:
            word = word_info["word"].strip()
            word_start = transcript.find(word, current_char_pos)
            
            if word_start >= 0:
                word_end = word_start + len(word)
                
                if word_end <= char_pos:
                    words_before.append(word_info)
                elif word_start > char_pos:
                    words_after.append(word_info)
                
                current_char_pos = word_end
        
        # Estimate time based on surrounding words
        if words_before and words_after:
            # Between two words
            before_time = words_before[-1]["end"]
            after_time = words_after[0]["start"]
            return (before_time + after_time) / 2
        elif words_before:
            # After last word
            return words_before[-1]["end"]
        elif words_after:
            # Before first word
            return words_after[0]["start"]
        
        return None
    
    def _find_filler_locations(self, asr_result: ASRResult) -> List[Dict[str, Any]]:
        """Find filler word locations in speech."""
        filler_locations = []
        
        for word_info in asr_result.words:
            word = word_info["word"].strip().lower()
            
            # Check if word matches filler patterns
            for pattern_str, pattern in self.compiled_filler_patterns:
                if pattern.match(word):
                    filler_locations.append({
                        "word": word,
                        "start": word_info.get("start", 0.0),
                        "end": word_info.get("end", 0.0),
                        "pattern": pattern_str
                    })
                    break
        
        return filler_locations
    
    def _classify_single_pause(self, 
                             pause: Dict[str, Any], 
                             word_timeline: List[Dict[str, Any]],
                             punctuation_timeline: List[Dict[str, Any]],
                             filler_locations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Classify a single pause as hesitation or rhetorical."""
        pause_start = pause["start"]
        pause_end = pause["end"]
        pause_duration = pause["duration"]
        
        # Check for filler proximity (hesitation indicator)
        near_filler = self._is_near_filler(pause_start, pause_end, filler_locations)
        if near_filler:
            return {
                "type": "hesitation",
                "confidence": 0.9,
                "reason": f"Near filler word: {near_filler['word']}",
                "context": near_filler
            }
        
        # Check for mid-clause position (hesitation indicator)
        mid_clause = self._is_mid_clause(pause_start, pause_end, word_timeline, punctuation_timeline)
        if mid_clause:
            return {
                "type": "hesitation", 
                "confidence": 0.8,
                "reason": "Mid-clause position",
                "context": mid_clause
            }
        
        # Check for punctuation alignment (rhetorical indicator)
        near_punctuation = self._is_near_punctuation(pause_start, pause_end, punctuation_timeline)
        if near_punctuation and pause_duration >= self.rhetorical_threshold:
            return {
                "type": "rhetorical",
                "confidence": 0.9,
                "reason": f"Aligned with punctuation: {near_punctuation['mark']}",
                "context": near_punctuation
            }
        
        # Default classification based on duration
        if pause_duration >= self.rhetorical_threshold:
            return {
                "type": "rhetorical",
                "confidence": 0.6,
                "reason": f"Long pause ({pause_duration:.1f}s)",
                "context": ""
            }
        else:
            return {
                "type": "hesitation",
                "confidence": 0.7,
                "reason": f"Short pause ({pause_duration:.1f}s)",
                "context": ""
            }
    
    def _is_near_filler(self, pause_start: float, pause_end: float, filler_locations: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Check if pause is near a filler word."""
        for filler in filler_locations:
            filler_start = filler["start"]
            filler_end = filler["end"]
            
            # Check if filler is within tolerance of pause
            if (abs(filler_end - pause_start) <= self.filler_tolerance or
                abs(pause_end - filler_start) <= self.filler_tolerance):
                return filler
        
        return None
    
    def _is_mid_clause(self, 
                      pause_start: float, 
                      pause_end: float,
                      word_timeline: List[Dict[str, Any]],
                      punctuation_timeline: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Check if pause occurs mid-clause (between grammatical units)."""
        # Find words immediately before and after pause
        word_before = None
        word_after = None
        
        for word in word_timeline:
            if word["end"] <= pause_start:
                word_before = word
            elif word["start"] >= pause_end and word_after is None:
                word_after = word
                break
        
        if not word_before or not word_after:
            return None
        
        # Check if there's punctuation nearby (indicates clause boundary)
        for punct in punctuation_timeline:
            if (abs(punct["time"] - pause_start) <= self.punctuation_tolerance or
                abs(punct["time"] - pause_end) <= self.punctuation_tolerance):
                return None  # Near punctuation, likely not mid-clause
        
        # Simple heuristic: check if words suggest mid-clause
        # This is a simplified version - a full parser would be more accurate
        word_before_text = word_before["word"].lower()
        word_after_text = word_after["word"].lower()
        
        # Common patterns that suggest mid-clause pauses
        mid_clause_patterns = [
            # Verb-object separation
            ("is", "the"), ("was", "a"), ("have", "been"),
            # Article-noun separation  
            ("a", "good"), ("the", "best"), ("an", "important"),
            # Preposition-object separation
            ("in", "the"), ("on", "a"), ("with", "their"),
            # Common mid-sentence breaks
            ("and", "then"), ("but", "i"), ("so", "we")
        ]
        
        for before_pattern, after_pattern in mid_clause_patterns:
            if before_pattern in word_before_text and after_pattern in word_after_text:
                return {
                    "word_before": word_before_text,
                    "word_after": word_after_text,
                    "pattern": f"{before_pattern}...{after_pattern}"
                }
        
        # Additional check: if next word starts with lowercase (not sentence start)
        if (word_after_text and word_after_text[0].islower() and 
            not word_after_text.startswith(("i", "i'm", "i'll", "i've", "i'd"))):
            return {
                "word_before": word_before_text,
                "word_after": word_after_text,
                "pattern": "mid_sentence_lowercase"
            }
        
        return None
    
    def _is_near_punctuation(self, pause_start: float, pause_end: float, punctuation_timeline: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Check if pause aligns with punctuation."""
        for punct in punctuation_timeline:
            punct_time = punct["time"]
            
            # Check if punctuation is within tolerance of pause
            if (abs(punct_time - pause_start) <= self.punctuation_tolerance or
                abs(punct_time - pause_end) <= self.punctuation_tolerance or
                (pause_start <= punct_time <= pause_end)):
                return punct
        
        return None
    
    def _calculate_metrics(self, classified_pauses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate pause usage metrics."""
        if not classified_pauses:
            return {
                "total_pauses": 0,
                "hesitation_pauses": 0,
                "rhetorical_pauses": 0,
                "hesitation_percentage": 0.0,
                "rhetorical_percentage": 0.0,
                "avg_pause_duration": 0.0,
                "avg_hesitation_duration": 0.0,
                "avg_rhetorical_duration": 0.0
            }
        
        hesitation_pauses = [p for p in classified_pauses if p["type"] == "hesitation"]
        rhetorical_pauses = [p for p in classified_pauses if p["type"] == "rhetorical"]
        
        total_pauses = len(classified_pauses)
        hesitation_count = len(hesitation_pauses)
        rhetorical_count = len(rhetorical_pauses)
        
        return {
            "total_pauses": total_pauses,
            "hesitation_pauses": hesitation_count,
            "rhetorical_pauses": rhetorical_count,
            "hesitation_percentage": round(hesitation_count / total_pauses * 100, 1) if total_pauses > 0 else 0,
            "rhetorical_percentage": round(rhetorical_count / total_pauses * 100, 1) if total_pauses > 0 else 0,
            "avg_pause_duration": round(np.mean([p["duration"] for p in classified_pauses]), 2),
            "avg_hesitation_duration": round(np.mean([p["duration"] for p in hesitation_pauses]), 2) if hesitation_pauses else 0,
            "avg_rhetorical_duration": round(np.mean([p["duration"] for p in rhetorical_pauses]), 2) if rhetorical_pauses else 0
        }
    
    def _assess_pause_usage(self, metrics: Dict[str, Any]) -> str:
        """Assess overall pause usage."""
        hesitation_percentage = metrics.get("hesitation_percentage", 0)
        rhetorical_percentage = metrics.get("rhetorical_percentage", 0)
        
        if hesitation_percentage <= 20 and rhetorical_percentage >= 70:
            return "excellent"
        elif hesitation_percentage <= 30 and rhetorical_percentage >= 60:
            return "good"
        elif hesitation_percentage <= 50:
            return "fair"
        else:
            return "needs_improvement"
    
    def _get_recommendations(self, metrics: Dict[str, Any], classified_pauses: List[Dict[str, Any]]) -> List[str]:
        """Generate pause usage recommendations."""
        recommendations = []
        
        hesitation_count = metrics.get("hesitation_pauses", 0)
        rhetorical_count = metrics.get("rhetorical_pauses", 0)
        hesitation_pct = metrics.get("hesitation_percentage", 0)
        
        # Hesitation pause recommendations
        if hesitation_count == 0:
            recommendations.append("Excellent! No hesitation pauses detected.")
        elif hesitation_count <= 2:
            recommendations.append(f"Very few hesitation pauses detected ({hesitation_count}). Good control!")
        elif hesitation_count <= 5:
            recommendations.append(f"Some hesitation pauses detected ({hesitation_count}). Work on maintaining flow.")
        else:
            recommendations.append(
                f"Many hesitation pauses detected ({hesitation_count}). "
                "Practice speaking more fluidly and prepare your thoughts in advance."
            )
        
        # Rhetorical pause recommendations
        if rhetorical_count > 0:
            recommendations.append(f"Good use of rhetorical pauses ({rhetorical_count}). These enhance your delivery!")
        else:
            recommendations.append("Consider adding purposeful pauses at punctuation marks for better emphasis.")
        
        # Specific pattern recommendations
        common_hesitation_reasons = {}
        for pause in classified_pauses:
            if pause["type"] == "hesitation":
                reason = pause["reason"].split(":")[0]  # Get general reason type
                common_hesitation_reasons[reason] = common_hesitation_reasons.get(reason, 0) + 1
        
        if "Near filler word" in common_hesitation_reasons:
            count = common_hesitation_reasons["Near filler word"]
            recommendations.append(
                f"You pause near filler words ({count} times). "
                "Try to eliminate fillers and replace with purposeful pauses."
            )
        
        if "Mid-clause position" in common_hesitation_reasons:
            count = common_hesitation_reasons["Mid-clause position"]
            recommendations.append(
                f"You pause mid-sentence ({count} times). "
                "Practice completing phrases without breaks."
            )
        
        return recommendations
    
    def _get_timeline_data(self, classified_pauses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate timeline data for visualization."""
        timeline = []
        
        for pause in classified_pauses:
            timeline.append({
                "start": pause["start"],
                "end": pause["end"],
                "duration": pause["duration"],
                "type": pause["type"],
                "confidence": pause["confidence"],
                "reason": pause["reason"]
            })
        
        return sorted(timeline, key=lambda x: x["start"])


# Global pause analyzer instance
pause_analyzer = PauseAnalyzer()