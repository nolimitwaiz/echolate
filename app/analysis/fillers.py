import re
import logging
from typing import Dict, List, Any, Tuple
from ..models.asr import ASRResult
from ..settings import settings

logger = logging.getLogger(__name__)


class FillerAnalyzer:
    """Analyze filler words and speech disfluencies."""
    
    def __init__(self):
        self.filler_patterns = settings.fillers.get("patterns", [])
        # Compile regex patterns for efficiency
        self.compiled_patterns = [(pattern, re.compile(pattern, re.IGNORECASE)) 
                                  for pattern in self.filler_patterns]
    
    def analyze(self, asr_result: ASRResult, speech_duration: float) -> Dict[str, Any]:
        """Analyze filler words in speech."""
        try:
            # Find all fillers in transcript
            fillers = self._find_fillers(asr_result.text, asr_result.words)
            
            # Calculate metrics
            metrics = self._calculate_metrics(fillers, speech_duration)
            
            # Get recommendations
            recommendations = self._get_recommendations(metrics)
            
            return {
                "filler_count": metrics["count"],
                "fillers_per_minute": metrics["per_minute"],
                "filler_percentage": metrics["percentage"],
                "filler_types": metrics["types"],
                "filler_instances": fillers,
                "assessment": self._assess_filler_usage(metrics["per_minute"]),
                "recommendations": recommendations
            }
            
        except Exception as e:
            logger.error(f"Error in filler analysis: {e}")
            return {
                "filler_count": 0,
                "fillers_per_minute": 0.0,
                "filler_percentage": 0.0,
                "filler_types": {},
                "filler_instances": [],
                "assessment": "unknown",
                "error": str(e)
            }
    
    def _find_fillers(self, transcript: str, words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find filler words with timestamps."""
        fillers = []
        
        # First pass: find fillers in full transcript using regex
        transcript_fillers = []
        for pattern_str, pattern in self.compiled_patterns:
            for match in pattern.finditer(transcript):
                transcript_fillers.append({
                    "text": match.group(),
                    "start_char": match.start(),
                    "end_char": match.end(),
                    "pattern": pattern_str
                })
        
        # Sort by position
        transcript_fillers.sort(key=lambda x: x["start_char"])
        
        # Second pass: match with word-level timestamps
        if words:
            # Build character position mapping for words
            word_positions = self._build_word_positions(transcript, words)
            
            for filler in transcript_fillers:
                # Find corresponding word(s) for timing
                word_info = self._find_word_timing(filler, word_positions)
                
                if word_info:
                    fillers.append({
                        "text": filler["text"],
                        "start": word_info["start"],
                        "end": word_info["end"],
                        "pattern": filler["pattern"],
                        "confidence": word_info.get("confidence", 0.0),
                        "type": self._classify_filler(filler["text"])
                    })
        else:
            # No word timing available, just use text matches
            for filler in transcript_fillers:
                fillers.append({
                    "text": filler["text"],
                    "start": 0.0,
                    "end": 0.0,
                    "pattern": filler["pattern"],
                    "confidence": 0.0,
                    "type": self._classify_filler(filler["text"])
                })
        
        return fillers
    
    def _build_word_positions(self, transcript: str, words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build mapping between character positions and word timestamps."""
        word_positions = []
        current_pos = 0
        
        for word in words:
            word_text = word["word"].strip()
            if not word_text:
                continue
            
            # Find word in transcript starting from current position
            word_start = transcript.lower().find(word_text.lower(), current_pos)
            
            if word_start >= 0:
                word_end = word_start + len(word_text)
                word_positions.append({
                    "text": word_text,
                    "char_start": word_start,
                    "char_end": word_end,
                    "start": word["start"],
                    "end": word["end"],
                    "confidence": word.get("confidence", 0.0)
                })
                current_pos = word_end
        
        return word_positions
    
    def _find_word_timing(self, filler: Dict[str, Any], word_positions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find timing information for a filler match."""
        filler_start = filler["start_char"]
        filler_end = filler["end_char"]
        
        # Find overlapping words
        overlapping_words = []
        for word_info in word_positions:
            word_start = word_info["char_start"]
            word_end = word_info["char_end"]
            
            # Check for overlap
            if (word_start <= filler_end and word_end >= filler_start):
                overlapping_words.append(word_info)
        
        if overlapping_words:
            # Use timing from first to last overlapping word
            start_time = min(word["start"] for word in overlapping_words)
            end_time = max(word["end"] for word in overlapping_words)
            avg_confidence = sum(word.get("confidence", 0.0) for word in overlapping_words) / len(overlapping_words)
            
            return {
                "start": start_time,
                "end": end_time,
                "confidence": avg_confidence
            }
        
        return None
    
    def _classify_filler(self, text: str) -> str:
        """Classify filler type."""
        text_lower = text.lower().strip()
        
        if text_lower in ["uh", "um", "uhh", "umm", "erm", "ahh", "ah"]:
            return "vocal_filler"
        elif text_lower in ["like", "you know", "sort of", "kind of", "basically", "actually"]:
            return "lexical_filler"
        elif text_lower in ["and", "so", "but", "well"]:
            return "discourse_marker"
        else:
            return "other"
    
    def _calculate_metrics(self, fillers: List[Dict[str, Any]], speech_duration: float) -> Dict[str, Any]:
        """Calculate filler usage metrics."""
        count = len(fillers)
        per_minute = (count / speech_duration * 60) if speech_duration > 0 else 0
        
        # Calculate filler types distribution
        type_counts = {}
        for filler in fillers:
            filler_type = filler["type"]
            type_counts[filler_type] = type_counts.get(filler_type, 0) + 1
        
        # Estimate word count for percentage (rough estimate: 150 WPM average)
        estimated_words = speech_duration * 2.5  # 150 WPM / 60 seconds
        percentage = (count / estimated_words * 100) if estimated_words > 0 else 0
        
        return {
            "count": count,
            "per_minute": round(per_minute, 1),
            "percentage": round(percentage, 1),
            "types": type_counts
        }
    
    def _assess_filler_usage(self, fillers_per_minute: float) -> str:
        """Assess filler usage level."""
        if fillers_per_minute <= 1:
            return "excellent"
        elif fillers_per_minute <= 2:
            return "good"
        elif fillers_per_minute <= 4:
            return "moderate"
        elif fillers_per_minute <= 6:
            return "high"
        else:
            return "excessive"
    
    def _get_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate recommendations for filler reduction."""
        recommendations = []
        
        per_minute = metrics["per_minute"]
        count = metrics["count"]
        types = metrics["types"]
        
        # Overall assessment
        if per_minute <= 1:
            recommendations.append("Excellent! You use very few filler words.")
        elif per_minute <= 2:
            recommendations.append("Good job! Your filler usage is quite low.")
        elif per_minute <= 4:
            recommendations.append(
                f"You used {count} filler words ({per_minute}/min). "
                "Try to pause instead of using fillers."
            )
        elif per_minute <= 6:
            recommendations.append(
                f"You used {count} filler words ({per_minute}/min). "
                "Practice pausing to gather your thoughts instead of using fillers."
            )
        else:
            recommendations.append(
                f"You used {count} filler words ({per_minute}/min). "
                "Focus on reducing filler words by slowing down and using purposeful pauses."
            )
        
        # Specific type recommendations
        if types.get("vocal_filler", 0) > 2:
            recommendations.append(
                "Focus on replacing vocal fillers (um, uh) with brief, silent pauses."
            )
        
        if types.get("lexical_filler", 0) > 2:
            recommendations.append(
                "Try to reduce words like 'like', 'you know', and 'basically'. "
                "Be more direct in your word choice."
            )
        
        if types.get("discourse_marker", 0) > 3:
            recommendations.append(
                "You use many connector words ('and', 'so', 'but'). "
                "Vary your sentence structure for better flow."
            )
        
        return recommendations
    
    def get_timeline_data(self, fillers: List[Dict[str, Any]], duration: float) -> List[Dict[str, Any]]:
        """Generate timeline data for visualization."""
        timeline = []
        
        for filler in fillers:
            timeline.append({
                "time": filler["start"],
                "duration": filler["end"] - filler["start"],
                "text": filler["text"],
                "type": filler["type"]
            })
        
        return timeline


# Global filler analyzer instance
filler_analyzer = FillerAnalyzer()