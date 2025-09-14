import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Tuple, Optional
import logging

from ..settings import settings

logger = logging.getLogger(__name__)


class AudioPreprocessor:
    """Audio preprocessing utilities."""
    
    def __init__(self):
        self.target_sr = settings.audio.get("sample_rate", 16000)
        self.max_duration = settings.audio.get("max_duration_seconds", 90)
    
    def load_and_preprocess(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Load and preprocess audio file."""
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=None)
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = librosa.to_mono(audio)
            
            # Resample if needed
            if sr != self.target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
                sr = self.target_sr
            
            # Trim silence from start and end
            audio = self._trim_silence(audio, sr)
            
            # Limit duration
            max_samples = int(self.max_duration * sr)
            if len(audio) > max_samples:
                logger.warning(f"Audio truncated to {self.max_duration}s")
                audio = audio[:max_samples]
            
            # Normalize volume
            audio = self._normalize_audio(audio)
            
            return audio, sr
            
        except Exception as e:
            logger.error(f"Error preprocessing audio {audio_path}: {e}")
            raise
    
    def _trim_silence(self, audio: np.ndarray, sr: int, threshold: float = 0.02) -> np.ndarray:
        """Trim silence from start and end of audio."""
        # Use librosa's trim function
        trimmed, _ = librosa.effects.trim(audio, top_db=20)
        return trimmed if len(trimmed) > 0 else audio
    
    def _normalize_audio(self, audio: np.ndarray, target_level: float = -20.0) -> np.ndarray:
        """Normalize audio to target RMS level in dB."""
        if len(audio) == 0:
            return audio
        
        # Calculate current RMS
        current_rms = np.sqrt(np.mean(audio**2))
        
        if current_rms == 0:
            return audio
        
        # Calculate target RMS from dB
        target_rms = 10**(target_level/20)
        
        # Apply normalization
        gain = target_rms / current_rms
        normalized = audio * gain
        
        # Prevent clipping
        max_val = np.max(np.abs(normalized))
        if max_val > 0.95:
            normalized = normalized * (0.95 / max_val)
        
        return normalized
    
    def save_preprocessed(self, audio: np.ndarray, sr: int, output_path: str) -> None:
        """Save preprocessed audio to file."""
        sf.write(output_path, audio, sr)
        logger.info(f"Saved preprocessed audio to {output_path}")


class TextNormalizer:
    """Text normalization utilities."""
    
    def __init__(self):
        self.filler_patterns = settings.fillers.get("patterns", [])
    
    def normalize_for_comparison(self, text: str) -> str:
        """Normalize text for comparison (remove fillers, punctuation, etc.)."""
        import re
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def remove_fillers(self, text: str) -> str:
        """Remove filler words from text."""
        import re
        
        for pattern in self.filler_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Clean up extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        import re
        
        # Simple sentence splitting on periods, exclamation marks, and question marks
        sentences = re.split(r'[.!?]+', text)
        
        # Clean and filter empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences


# Global instances
audio_preprocessor = AudioPreprocessor()
text_normalizer = TextNormalizer()