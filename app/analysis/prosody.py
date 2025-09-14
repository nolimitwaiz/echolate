import logging
import numpy as np
import librosa
from typing import Dict, List, Any, Tuple, Optional
from scipy import signal
import re

from ..models.asr import ASRResult
from ..analysis.preprocess import text_normalizer
from ..settings import settings

logger = logging.getLogger(__name__)


class ProsodyAnalyzer:
    """Analyze prosodic features including pitch, energy, and uptalk."""
    
    def __init__(self):
        self.config = settings.analysis.prosody
        self.uptalk_threshold = self.config.get("uptalk_threshold", 0.18)
        self.pitch_stability_window = self.config.get("pitch_stability_window", 2.0)
        self.energy_stability_window = self.config.get("energy_stability_window", 1.0)
    
    def analyze(self, audio: np.ndarray, sample_rate: int, asr_result: ASRResult) -> Dict[str, Any]:
        """Analyze prosodic features of speech."""
        try:
            # Extract fundamental frequency (F0) and energy
            f0_contour, energy_contour = self._extract_prosodic_features(audio, sample_rate)
            
            # Analyze pitch characteristics
            pitch_analysis = self._analyze_pitch(f0_contour, sample_rate)
            
            # Analyze energy characteristics
            energy_analysis = self._analyze_energy(energy_contour, sample_rate)
            
            # Detect uptalk patterns
            uptalk_analysis = self._detect_uptalk(f0_contour, sample_rate, asr_result)
            
            # Calculate stability metrics
            stability_metrics = self._calculate_stability(f0_contour, energy_contour, sample_rate)
            
            # Generate overall assessment
            assessment = self._assess_prosody(pitch_analysis, energy_analysis, uptalk_analysis, stability_metrics)
            
            # Generate recommendations
            recommendations = self._get_recommendations(pitch_analysis, energy_analysis, uptalk_analysis, stability_metrics)
            
            return {
                "pitch_analysis": pitch_analysis,
                "energy_analysis": energy_analysis,
                "uptalk_analysis": uptalk_analysis,
                "stability_metrics": stability_metrics,
                "assessment": assessment,
                "recommendations": recommendations
            }
            
        except Exception as e:
            logger.error(f"Error in prosody analysis: {e}")
            return {
                "assessment": "error",
                "error": str(e)
            }
    
    def _extract_prosodic_features(self, audio: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, np.ndarray]:
        """Extract F0 and energy contours from audio."""
        # Extract F0 using librosa's yin algorithm
        try:
            f0 = librosa.yin(audio, fmin=50, fmax=400, sr=sample_rate, frame_length=2048)
            # Remove unvoiced frames (f0 = 0)
            f0_clean = np.where(f0 > 0, f0, np.nan)
        except Exception as e:
            logger.warning(f"F0 extraction with YIN failed: {e}, using piptrack fallback")
            # Fallback to piptrack
            pitches, magnitudes = librosa.piptrack(y=audio, sr=sample_rate, fmin=50, fmax=400)
            f0_clean = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t] if magnitudes[index, t] > 0 else 0
                f0_clean.append(pitch if pitch > 0 else np.nan)
            f0_clean = np.array(f0_clean)
        
        # Extract energy (RMS)
        energy = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
        
        return f0_clean, energy
    
    def _analyze_pitch(self, f0_contour: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Analyze pitch characteristics."""
        voiced_f0 = f0_contour[~np.isnan(f0_contour)]
        
        if len(voiced_f0) == 0:
            return {
                "mean_f0": 0.0,
                "std_f0": 0.0,
                "f0_range": 0.0,
                "voiced_percentage": 0.0
            }
        
        mean_f0 = np.mean(voiced_f0)
        std_f0 = np.std(voiced_f0)
        f0_range = np.max(voiced_f0) - np.min(voiced_f0)
        voiced_percentage = len(voiced_f0) / len(f0_contour) * 100
        
        return {
            "mean_f0": round(mean_f0, 1),
            "std_f0": round(std_f0, 1),
            "f0_range": round(f0_range, 1),
            "voiced_percentage": round(voiced_percentage, 1),
            "min_f0": round(np.min(voiced_f0), 1),
            "max_f0": round(np.max(voiced_f0), 1)
        }
    
    def _analyze_energy(self, energy_contour: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Analyze energy characteristics."""
        if len(energy_contour) == 0:
            return {
                "mean_energy": 0.0,
                "std_energy": 0.0,
                "energy_range": 0.0
            }
        
        mean_energy = np.mean(energy_contour)
        std_energy = np.std(energy_contour)
        energy_range = np.max(energy_contour) - np.min(energy_contour)
        
        return {
            "mean_energy": round(float(mean_energy), 4),
            "std_energy": round(float(std_energy), 4),
            "energy_range": round(float(energy_range), 4),
            "min_energy": round(float(np.min(energy_contour)), 4),
            "max_energy": round(float(np.max(energy_contour)), 4)
        }
    
    def _detect_uptalk(self, f0_contour: np.ndarray, sample_rate: int, asr_result: ASRResult) -> Dict[str, Any]:
        """Detect uptalk patterns in declarative sentences."""
        if not asr_result.words or len(f0_contour) == 0:
            return {
                "uptalk_count": 0,
                "uptalk_instances": [],
                "uptalk_percentage": 0.0
            }
        
        # Split transcript into sentences
        sentences = self._split_into_sentences(asr_result.text, asr_result.words)
        
        uptalk_instances = []
        declarative_count = 0
        
        for sentence_info in sentences:
            if sentence_info["type"] == "declarative":
                declarative_count += 1
                
                # Check for uptalk in this sentence
                uptalk_detected = self._check_sentence_for_uptalk(
                    sentence_info, f0_contour, sample_rate, asr_result.words
                )
                
                if uptalk_detected:
                    uptalk_instances.append(sentence_info)
        
        uptalk_count = len(uptalk_instances)
        uptalk_percentage = (uptalk_count / declarative_count * 100) if declarative_count > 0 else 0
        
        return {
            "uptalk_count": uptalk_count,
            "uptalk_instances": uptalk_instances[:5],  # Top 5 examples
            "uptalk_percentage": round(uptalk_percentage, 1),
            "declarative_count": declarative_count
        }
    
    def _split_into_sentences(self, transcript: str, words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Split transcript into sentences with timing information."""
        sentences = []
        
        # Simple sentence splitting
        sentence_texts = re.split(r'([.!?]+)', transcript)
        
        current_word_idx = 0
        current_char_pos = 0
        
        for i in range(0, len(sentence_texts), 2):
            if i >= len(sentence_texts):
                break
                
            sentence_text = sentence_texts[i].strip()
            punctuation = sentence_texts[i + 1] if i + 1 < len(sentence_texts) else ""
            
            if not sentence_text:
                continue
            
            # Determine sentence type
            if "?" in punctuation:
                sentence_type = "question"
            elif "!" in punctuation:
                sentence_type = "exclamation"
            else:
                sentence_type = "declarative"
            
            # Find words in this sentence
            sentence_words = []
            sentence_start_char = current_char_pos
            sentence_end_char = current_char_pos + len(sentence_text) + len(punctuation)
            
            # Find corresponding words with timing
            while current_word_idx < len(words):
                word_info = words[current_word_idx]
                word_text = word_info["word"].strip()
                
                # Check if word is in current sentence (rough matching)
                if word_text.lower() in sentence_text.lower():
                    sentence_words.append(word_info)
                    current_word_idx += 1
                    if len(sentence_words) >= len(sentence_text.split()):
                        break
                else:
                    break
            
            if sentence_words:
                sentences.append({
                    "text": sentence_text + punctuation,
                    "type": sentence_type,
                    "words": sentence_words,
                    "start": sentence_words[0]["start"],
                    "end": sentence_words[-1]["end"],
                    "char_start": sentence_start_char,
                    "char_end": sentence_end_char
                })
            
            current_char_pos = sentence_end_char
        
        return sentences
    
    def _check_sentence_for_uptalk(self, 
                                 sentence_info: Dict[str, Any], 
                                 f0_contour: np.ndarray, 
                                 sample_rate: int,
                                 all_words: List[Dict[str, Any]]) -> bool:
        """Check if a sentence has uptalk (rising intonation at the end)."""
        sentence_words = sentence_info["words"]
        if not sentence_words:
            return False
        
        # Focus on the final word + a small tail (150-300ms)
        final_word = sentence_words[-1]
        final_word_start = final_word["start"]
        final_word_end = final_word["end"]
        
        # Extend by 200ms to catch terminal pitch
        analysis_start = final_word_start
        analysis_end = final_word_end + 0.2
        
        # Convert to F0 frame indices (assuming hop_length=512)
        hop_length = 512
        frame_rate = sample_rate / hop_length
        
        start_frame = int(analysis_start * frame_rate)
        end_frame = int(analysis_end * frame_rate)
        
        # Extract F0 for final segment
        if start_frame >= len(f0_contour) or end_frame <= start_frame:
            return False
        
        final_f0 = f0_contour[start_frame:min(end_frame, len(f0_contour))]
        voiced_f0 = final_f0[~np.isnan(final_f0)]
        
        if len(voiced_f0) < 3:  # Need minimum frames for trend analysis
            return False
        
        # Calculate pitch trend (linear slope)
        x = np.arange(len(voiced_f0))
        slope, _ = np.polyfit(x, voiced_f0, 1)
        
        # Calculate percentage change from start to end
        start_median = np.median(voiced_f0[:len(voiced_f0)//3]) if len(voiced_f0) >= 3 else voiced_f0[0]
        end_median = np.median(voiced_f0[-len(voiced_f0)//3:]) if len(voiced_f0) >= 3 else voiced_f0[-1]
        
        if start_median > 0:
            percent_change = (end_median - start_median) / start_median
        else:
            percent_change = 0
        
        # Detect uptalk if rising trend exceeds threshold
        return percent_change > self.uptalk_threshold or slope > 2.0
    
    def _calculate_stability(self, 
                           f0_contour: np.ndarray, 
                           energy_contour: np.ndarray, 
                           sample_rate: int) -> Dict[str, Any]:
        """Calculate pitch and energy stability metrics."""
        # Pitch stability
        voiced_f0 = f0_contour[~np.isnan(f0_contour)]
        if len(voiced_f0) > 0:
            pitch_cv = np.std(voiced_f0) / np.mean(voiced_f0) if np.mean(voiced_f0) > 0 else 0
            pitch_stability_score = max(0, 100 - (pitch_cv * 100))
        else:
            pitch_stability_score = 0
        
        # Energy stability
        if len(energy_contour) > 0:
            energy_cv = np.std(energy_contour) / np.mean(energy_contour) if np.mean(energy_contour) > 0 else 0
            energy_stability_score = max(0, 100 - (energy_cv * 100))
        else:
            energy_stability_score = 0
        
        return {
            "pitch_stability_score": round(pitch_stability_score, 1),
            "energy_stability_score": round(energy_stability_score, 1),
            "overall_stability_score": round((pitch_stability_score + energy_stability_score) / 2, 1)
        }
    
    def _assess_prosody(self, 
                       pitch_analysis: Dict[str, Any],
                       energy_analysis: Dict[str, Any],
                       uptalk_analysis: Dict[str, Any],
                       stability_metrics: Dict[str, Any]) -> str:
        """Generate overall prosody assessment."""
        stability_score = stability_metrics.get("overall_stability_score", 0)
        uptalk_percentage = uptalk_analysis.get("uptalk_percentage", 0)
        
        if stability_score >= 80 and uptalk_percentage <= 10:
            return "excellent"
        elif stability_score >= 70 and uptalk_percentage <= 20:
            return "good"
        elif stability_score >= 60 and uptalk_percentage <= 30:
            return "fair"
        else:
            return "needs_improvement"
    
    def _get_recommendations(self, 
                           pitch_analysis: Dict[str, Any],
                           energy_analysis: Dict[str, Any],
                           uptalk_analysis: Dict[str, Any],
                           stability_metrics: Dict[str, Any]) -> List[str]:
        """Generate prosody recommendations."""
        recommendations = []
        
        # Pitch stability recommendations
        pitch_stability = stability_metrics.get("pitch_stability_score", 0)
        if pitch_stability < 70:
            recommendations.append(
                "Your pitch varies significantly. Practice maintaining more consistent intonation patterns."
            )
        elif pitch_stability >= 90:
            recommendations.append("Excellent pitch control! Your intonation is very stable.")
        
        # Energy recommendations
        energy_stability = stability_metrics.get("energy_stability_score", 0)
        if energy_stability < 70:
            recommendations.append(
                "Your volume fluctuates quite a bit. Practice speaking with more consistent energy."
            )
        
        # Uptalk recommendations
        uptalk_count = uptalk_analysis.get("uptalk_count", 0)
        uptalk_percentage = uptalk_analysis.get("uptalk_percentage", 0)
        
        if uptalk_count > 0:
            if uptalk_percentage > 30:
                recommendations.append(
                    f"You used rising intonation on {uptalk_count} statements ({uptalk_percentage:.0f}%). "
                    "Practice ending declarative sentences with a downward pitch to sound more confident."
                )
            elif uptalk_percentage > 15:
                recommendations.append(
                    f"You used uptalk on {uptalk_count} statements. "
                    "Try landing with a downward pitch on declarative sentences."
                )
            else:
                recommendations.append(
                    f"Minimal uptalk detected ({uptalk_count} instances). Good control of statement intonation!"
                )
        else:
            recommendations.append("Excellent! No uptalk patterns detected in your statements.")
        
        # Overall prosody recommendations
        overall_stability = stability_metrics.get("overall_stability_score", 0)
        if overall_stability >= 85:
            recommendations.append("Excellent prosodic control overall!")
        elif overall_stability < 60:
            recommendations.append(
                "Focus on maintaining consistent pitch and energy throughout your speech."
            )
        
        return recommendations


# Global prosody analyzer instance
prosody_analyzer = ProsodyAnalyzer()