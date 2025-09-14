import logging
import numpy as np
from typing import Dict, List, Any, Set, Tuple
import re

try:
    from g2p_en import G2p
    G2P_AVAILABLE = True
except ImportError:
    try:
        from phonemizer import phonemize
        PHONEMIZER_AVAILABLE = True
        G2P_AVAILABLE = False
    except ImportError:
        G2P_AVAILABLE = False
        PHONEMIZER_AVAILABLE = False

from ..models.asr import ASRResult
from ..analysis.fillers import filler_analyzer
from ..settings import settings

logger = logging.getLogger(__name__)


class PhonemesAnalyzer:
    """Analyze phoneme pronunciation risks and accent patterns."""
    
    def __init__(self):
        self.risk_groups = settings.analysis.phonemes.get("risk_groups", {
            "TH": ["θ", "ð"],
            "RL": ["r", "l"],
            "WV": ["w", "v"], 
            "ZH": ["ʒ"],
            "NG": ["ŋ"]
        })
        
        self.confidence_weight = settings.analysis.phonemes.get("confidence_penalty_weight", 0.4)
        self.duration_weight = settings.analysis.phonemes.get("duration_penalty_weight", 0.3)
        self.filler_weight = settings.analysis.phonemes.get("filler_penalty_weight", 0.3)
        
        # Initialize G2P
        if G2P_AVAILABLE:
            try:
                self.g2p = G2p()
                logger.info("Initialized G2P for phoneme analysis")
            except Exception as e:
                logger.warning(f"Failed to initialize G2P: {e}")
                self.g2p = None
        else:
            self.g2p = None
            if not PHONEMIZER_AVAILABLE:
                logger.warning("Neither g2p_en nor phonemizer available for phoneme analysis")
    
    def analyze(self, asr_result: ASRResult) -> Dict[str, Any]:
        """Analyze phoneme risks in speech."""
        try:
            if not self.g2p and not PHONEMIZER_AVAILABLE:
                return {
                    "phoneme_risks": {},
                    "overall_risk": 0,
                    "risky_words": [],
                    "assessment": "unavailable",
                    "error": "No phoneme analysis backend available"
                }
            
            # Get word-phoneme mappings
            word_phonemes = self._get_word_phonemes(asr_result.words)
            
            # Analyze risk for each phoneme group
            group_risks = {}
            all_risky_words = []
            
            for group_name, phonemes in self.risk_groups.items():
                risk_analysis = self._analyze_phoneme_group(
                    group_name, phonemes, word_phonemes, asr_result.words
                )
                group_risks[group_name] = risk_analysis
                all_risky_words.extend(risk_analysis.get("risky_words", []))
            
            # Calculate overall risk
            overall_risk = self._calculate_overall_risk(group_risks)
            
            # Generate recommendations
            recommendations = self._get_recommendations(group_risks, overall_risk)
            
            return {
                "phoneme_risks": group_risks,
                "overall_risk": overall_risk,
                "risky_words": all_risky_words[:10],  # Top 10 most risky
                "assessment": self._assess_overall_risk(overall_risk),
                "recommendations": recommendations,
                "practice_words": self._get_practice_words(group_risks)
            }
            
        except Exception as e:
            logger.error(f"Error in phoneme analysis: {e}")
            return {
                "phoneme_risks": {},
                "overall_risk": 0,
                "risky_words": [],
                "assessment": "error",
                "error": str(e)
            }
    
    def analyze_drill(self, asr_result: ASRResult, target_phoneme_group: str) -> Dict[str, Any]:
        """Analyze single phoneme group for drill mode (faster)."""
        try:
            if target_phoneme_group not in self.risk_groups:
                raise ValueError(f"Unknown phoneme group: {target_phoneme_group}")
            
            if not self.g2p and not PHONEMIZER_AVAILABLE:
                return {"risk_score": 0, "error": "No phoneme analysis backend available"}
            
            # Get word-phoneme mappings
            word_phonemes = self._get_word_phonemes(asr_result.words)
            
            # Analyze only the target group
            target_phonemes = self.risk_groups[target_phoneme_group]
            risk_analysis = self._analyze_phoneme_group(
                target_phoneme_group, target_phonemes, word_phonemes, asr_result.words
            )
            
            return {
                "risk_score": risk_analysis.get("risk_score", 0),
                "word_count": risk_analysis.get("word_count", 0),
                "risky_words": risk_analysis.get("risky_words", [])[:3],  # Top 3
                "practice_suggestion": self._get_drill_suggestion(target_phoneme_group)
            }
            
        except Exception as e:
            logger.error(f"Error in phoneme drill analysis: {e}")
            return {"risk_score": 0, "error": str(e)}
    
    def _get_word_phonemes(self, words: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Get phoneme representation for each word."""
        word_phonemes = {}
        
        for word_info in words:
            word = word_info["word"].strip().lower()
            
            if word in word_phonemes:
                continue  # Skip duplicates
            
            try:
                if self.g2p:
                    # Use G2P
                    phonemes = self.g2p(word)
                else:
                    # Use phonemizer as fallback
                    phonemes = phonemize(word, language='en-us', backend='espeak', 
                                       strip=True, preserve_punctuation=False).split()
                
                word_phonemes[word] = phonemes
                
            except Exception as e:
                logger.debug(f"Failed to get phonemes for '{word}': {e}")
                word_phonemes[word] = []
        
        return word_phonemes
    
    def _analyze_phoneme_group(self, 
                             group_name: str,
                             target_phonemes: List[str], 
                             word_phonemes: Dict[str, List[str]],
                             words: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze risk for a specific phoneme group."""
        risky_words = []
        total_occurrences = 0
        
        # Find all words containing target phonemes
        for word_info in words:
            word = word_info["word"].strip().lower()
            phonemes = word_phonemes.get(word, [])
            
            if any(phoneme in phonemes for phoneme in target_phonemes):
                total_occurrences += 1
                
                # Calculate risk score for this word
                risk_score = self._calculate_word_risk(word_info, word, target_phonemes, phonemes)
                
                if risk_score > 30:  # Threshold for "risky"
                    risky_words.append({
                        "word": word,
                        "risk_score": round(risk_score, 1),
                        "confidence": word_info.get("confidence", 0.0),
                        "start": word_info.get("start", 0.0),
                        "phonemes": phonemes,
                        "target_phonemes_found": [p for p in phonemes if p in target_phonemes]
                    })
        
        # Sort by risk score
        risky_words.sort(key=lambda x: x["risk_score"], reverse=True)
        
        # Calculate group risk score
        if total_occurrences > 0:
            avg_risk = sum(w["risk_score"] for w in risky_words) / total_occurrences
            group_risk_score = min(100, avg_risk * (len(risky_words) / total_occurrences))
        else:
            group_risk_score = 0
        
        return {
            "risk_score": round(group_risk_score, 1),
            "word_count": total_occurrences,
            "risky_word_count": len(risky_words),
            "risky_words": risky_words,
            "example_words": self._get_example_words(group_name)
        }
    
    def _calculate_word_risk(self, 
                           word_info: Dict[str, Any],
                           word: str,
                           target_phonemes: List[str],
                           phonemes: List[str]) -> float:
        """Calculate risk score for a single word."""
        base_risk = 50.0  # Base risk for containing target phonemes
        
        # Confidence penalty (lower confidence = higher risk)
        confidence = word_info.get("confidence", 1.0)
        confidence_penalty = (1.0 - confidence) * 100 * self.confidence_weight
        
        # Duration penalty (unusually short/long pronunciations)
        duration_penalty = 0.0
        if "start" in word_info and "end" in word_info:
            duration = word_info["end"] - word_info["start"]
            expected_duration = len(word) * 0.08  # Rough estimate: 80ms per character
            
            if duration > 0:
                duration_ratio = abs(duration - expected_duration) / expected_duration
                duration_penalty = min(20, duration_ratio * 30) * self.duration_weight
        
        # Filler proximity penalty (words near fillers are often mispronounced)
        filler_penalty = 0.0
        # This would require filler detection, simplified for now
        
        total_risk = base_risk + confidence_penalty + duration_penalty + filler_penalty
        return min(100.0, total_risk)
    
    def _calculate_overall_risk(self, group_risks: Dict[str, Dict[str, Any]]) -> int:
        """Calculate overall phoneme risk score."""
        if not group_risks:
            return 0
        
        risk_scores = [group["risk_score"] for group in group_risks.values()]
        weighted_average = sum(risk_scores) / len(risk_scores)
        
        return round(weighted_average)
    
    def _assess_overall_risk(self, overall_risk: int) -> str:
        """Assess overall pronunciation risk level."""
        if overall_risk <= 20:
            return "low"
        elif overall_risk <= 40:
            return "moderate"
        elif overall_risk <= 60:
            return "high"
        else:
            return "very_high"
    
    def _get_example_words(self, phoneme_group: str) -> List[str]:
        """Get example words for practicing specific phoneme groups."""
        examples = {
            "TH": ["think", "this", "thirty", "three", "thank", "breath", "clothing"],
            "RL": ["really", "world", "color", "library", "February", "particular"],
            "WV": ["very", "wave", "video", "available", "every", "never"],
            "ZH": ["measure", "pleasure", "vision", "decision", "usual"],
            "NG": ["thinking", "working", "something", "anything", "running"]
        }
        return examples.get(phoneme_group, [])
    
    def _get_practice_words(self, group_risks: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
        """Get practice word suggestions for each risky phoneme group."""
        practice_words = {}
        
        for group_name, risk_info in group_risks.items():
            if risk_info.get("risk_score", 0) > 30:
                practice_words[group_name] = self._get_example_words(group_name)
        
        return practice_words
    
    def _get_drill_suggestion(self, phoneme_group: str) -> str:
        """Get drill suggestion for specific phoneme group."""
        suggestions = {
            "TH": "Practice 'The thirty-three thieves thought they thrilled the throne.'",
            "RL": "Practice 'Really red roses are rarely real.'",
            "WV": "Practice 'Very vivid videos were viewed by visitors.'",
            "ZH": "Practice 'Measure the pleasure of leisure and treasure.'",
            "NG": "Practice 'Running, jumping, thinking, working, singing.'"
        }
        return suggestions.get(phoneme_group, f"Practice words containing {phoneme_group} sounds.")
    
    def _get_recommendations(self, 
                           group_risks: Dict[str, Dict[str, Any]], 
                           overall_risk: int) -> List[str]:
        """Generate phoneme-specific recommendations."""
        recommendations = []
        
        # Overall assessment
        if overall_risk <= 20:
            recommendations.append("Excellent pronunciation! Your accent risks are minimal.")
        elif overall_risk <= 40:
            recommendations.append("Good pronunciation with moderate accent patterns.")
        elif overall_risk <= 60:
            recommendations.append("Some pronunciation challenges detected. Focus on problem sounds.")
        else:
            recommendations.append("Significant pronunciation challenges. Consider accent coaching.")
        
        # Specific group recommendations
        high_risk_groups = []
        for group_name, risk_info in group_risks.items():
            risk_score = risk_info.get("risk_score", 0)
            
            if risk_score > 50:
                high_risk_groups.append(group_name)
                risky_count = risk_info.get("risky_word_count", 0)
                total_count = risk_info.get("word_count", 0)
                
                if group_name == "TH":
                    recommendations.append(
                        f"TH sounds need work ({risky_count}/{total_count} words). "
                        "Place tongue between teeth for 'th' sounds."
                    )
                elif group_name == "RL":
                    recommendations.append(
                        f"R/L distinction needs work ({risky_count}/{total_count} words). "
                        "Practice tongue position differences."
                    )
                elif group_name == "WV":
                    recommendations.append(
                        f"W/V sounds need attention ({risky_count}/{total_count} words). "
                        "Use teeth for 'v', rounded lips for 'w'."
                    )
                elif group_name == "ZH":
                    recommendations.append(
                        f"ZH sounds need practice ({risky_count}/{total_count} words). "
                        "Similar to 'sh' but with voice."
                    )
                elif group_name == "NG":
                    recommendations.append(
                        f"NG endings need work ({risky_count}/{total_count} words). "
                        "Keep tongue back, don't add 'k' sound."
                    )
        
        # Practice recommendations
        if high_risk_groups:
            recommendations.append(
                f"Focus practice on: {', '.join(high_risk_groups)}. "
                "Use the Practice Mode for targeted drills."
            )
        
        return recommendations


# Global phonemes analyzer instance
phonemes_analyzer = PhonemesAnalyzer()