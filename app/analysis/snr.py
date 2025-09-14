import numpy as np
import librosa
import logging
from typing import Tuple, Dict, Any
from ..models.vad import VADProcessor, VADSegment
from ..settings import settings

logger = logging.getLogger(__name__)


class SNRAnalyzer:
    """Signal-to-Noise Ratio analyzer for audio quality assessment."""
    
    def __init__(self, vad_processor: VADProcessor = None):
        self.vad_processor = vad_processor or VADProcessor()
        self.snr_threshold = settings.analysis.snr_db_threshold
        self.epsilon = 1e-10  # Small constant to avoid division by zero
    
    def analyze_audio_file(self, audio_path: str) -> Dict[str, Any]:
        """Analyze SNR from audio file."""
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=None)
            return self.analyze_audio(audio, sr)
            
        except Exception as e:
            logger.error(f"Error analyzing SNR for {audio_path}: {e}")
            return {
                "snr_db": 0.0,
                "signal_rms": 0.0,
                "noise_rms": 0.0,
                "snr_ok": False,
                "error": str(e)
            }
    
    def analyze_audio(self, audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Analyze SNR from audio array."""
        try:
            # Get VAD segments
            vad_segments = self.vad_processor.process_audio(audio)
            
            if not vad_segments:
                return {
                    "snr_db": 0.0,
                    "signal_rms": 0.0,
                    "noise_rms": 0.0,
                    "snr_ok": False,
                    "error": "No VAD segments found"
                }
            
            # Separate speech and silence segments
            speech_segments = [seg for seg in vad_segments if seg.is_speech]
            silence_segments = [seg for seg in vad_segments if not seg.is_speech]
            
            if not speech_segments:
                return {
                    "snr_db": 0.0,
                    "signal_rms": 0.0,
                    "noise_rms": 0.0,
                    "snr_ok": False,
                    "error": "No speech segments detected"
                }
            
            # Calculate RMS for speech segments (signal)
            signal_rms = self._calculate_rms_for_segments(audio, speech_segments, sample_rate)
            
            # Calculate RMS for silence segments (noise)
            if silence_segments:
                noise_rms = self._calculate_rms_for_segments(audio, silence_segments, sample_rate)
            else:
                # If no silence segments, estimate noise from quietest 10% of speech
                noise_rms = self._estimate_noise_from_speech(audio, speech_segments, sample_rate)
            
            # Calculate SNR in dB
            snr_db = self._calculate_snr_db(signal_rms, noise_rms)
            snr_ok = snr_db >= self.snr_threshold
            
            return {
                "snr_db": float(snr_db),
                "signal_rms": float(signal_rms),
                "noise_rms": float(noise_rms),
                "snr_ok": snr_ok,
                "speech_segments_count": len(speech_segments),
                "silence_segments_count": len(silence_segments),
                "total_speech_duration": sum(seg.duration for seg in speech_segments),
                "total_silence_duration": sum(seg.duration for seg in silence_segments)
            }
            
        except Exception as e:
            logger.error(f"Error in SNR analysis: {e}")
            return {
                "snr_db": 0.0,
                "signal_rms": 0.0,
                "noise_rms": 0.0,
                "snr_ok": False,
                "error": str(e)
            }
    
    def _calculate_rms_for_segments(self, 
                                   audio: np.ndarray, 
                                   segments: list[VADSegment], 
                                   sample_rate: int) -> float:
        """Calculate RMS energy for given segments."""
        if not segments:
            return self.epsilon
        
        rms_values = []
        
        for segment in segments:
            start_sample = int(segment.start * sample_rate)
            end_sample = int(segment.end * sample_rate)
            
            # Ensure bounds are valid
            start_sample = max(0, start_sample)
            end_sample = min(len(audio), end_sample)
            
            if end_sample > start_sample:
                segment_audio = audio[start_sample:end_sample]
                rms = librosa.feature.rms(y=segment_audio, frame_length=2048, hop_length=512)[0]
                rms_values.extend(rms)
        
        if not rms_values:
            return self.epsilon
        
        return np.mean(rms_values)
    
    def _estimate_noise_from_speech(self, 
                                   audio: np.ndarray, 
                                   speech_segments: list[VADSegment], 
                                   sample_rate: int) -> float:
        """Estimate noise level from quietest parts of speech."""
        all_rms = []
        
        for segment in speech_segments:
            start_sample = int(segment.start * sample_rate)
            end_sample = int(segment.end * sample_rate)
            
            start_sample = max(0, start_sample)
            end_sample = min(len(audio), end_sample)
            
            if end_sample > start_sample:
                segment_audio = audio[start_sample:end_sample]
                rms = librosa.feature.rms(y=segment_audio, frame_length=2048, hop_length=512)[0]
                all_rms.extend(rms)
        
        if not all_rms:
            return self.epsilon
        
        # Use 10th percentile as noise estimate
        return np.percentile(all_rms, 10)
    
    def _calculate_snr_db(self, signal_rms: float, noise_rms: float) -> float:
        """Calculate SNR in decibels."""
        signal_rms = max(signal_rms, self.epsilon)
        noise_rms = max(noise_rms, self.epsilon)
        
        snr_linear = signal_rms / noise_rms
        snr_db = 20 * np.log10(snr_linear)
        
        return snr_db
    
    def get_snr_warning_message(self, snr_db: float) -> str:
        """Generate user-friendly warning message based on SNR."""
        if snr_db >= self.snr_threshold:
            return ""
        
        if snr_db < 10:
            return (
                "🚨 Very high background noise detected!\n\n"
                "Your audio has significant background noise that will severely impact "
                "the accuracy of the analysis. Please record in a much quieter environment."
            )
        elif snr_db < 15:
            return (
                "⚠️ High background noise detected!\n\n"
                "For accurate analysis, please record in a quieter space. "
                "Proceeding may result in low clarity scores."
            )
        else:
            return (
                "⚠️ Moderate background noise detected.\n\n"
                "For best results, consider recording in a quieter environment."
            )
    
    def should_block_analysis(self, snr_db: float) -> bool:
        """Determine if analysis should be blocked due to poor audio quality."""
        return snr_db < 10  # Block if extremely noisy


# Global SNR analyzer instance
snr_analyzer = SNRAnalyzer()