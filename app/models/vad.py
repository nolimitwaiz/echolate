import logging
import numpy as np
import soundfile as sf
from typing import List, Tuple, Optional
from pathlib import Path

try:
    import webrtcvad
    WEBRTCVAD_AVAILABLE = True
except ImportError:
    WEBRTCVAD_AVAILABLE = False

from ..settings import settings

logger = logging.getLogger(__name__)


class VADSegment:
    """Voice Activity Detection segment."""
    def __init__(self, start: float, end: float, is_speech: bool):
        self.start = start
        self.end = end
        self.is_speech = is_speech
        self.duration = end - start


class VADProcessor:
    """Voice Activity Detection using WebRTC VAD."""
    
    def __init__(self, 
                 aggressiveness: int = None,
                 frame_duration_ms: int = None):
        self.aggressiveness = aggressiveness or settings.vad.get("aggressiveness", 2)
        self.frame_duration_ms = frame_duration_ms or settings.vad.get("frame_duration_ms", 30)
        self.sample_rate = 16000  # WebRTC VAD requires 16kHz
        
        if not WEBRTCVAD_AVAILABLE:
            logger.warning("webrtcvad not available, using simple energy-based VAD")
            self.vad = None
        else:
            self.vad = webrtcvad.Vad(self.aggressiveness)
    
    def _energy_based_vad(self, audio: np.ndarray, frame_duration_ms: int) -> List[bool]:
        """Fallback energy-based VAD when webrtcvad is not available."""
        frame_size = int(self.sample_rate * frame_duration_ms / 1000)
        frames = []
        
        for i in range(0, len(audio) - frame_size + 1, frame_size):
            frame = audio[i:i + frame_size]
            energy = np.sum(frame ** 2)
            frames.append(energy)
        
        if not frames:
            return []
        
        # Simple threshold-based detection
        threshold = np.percentile(frames, 30)  # Bottom 30% as silence
        return [energy > threshold * 2 for energy in frames]
    
    def process_audio_file(self, audio_path: str) -> List[VADSegment]:
        """Process audio file and return VAD segments."""
        try:
            # Load audio
            audio, original_sr = sf.read(audio_path)
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = audio[:, 0]
            
            # Resample to 16kHz if needed
            if original_sr != self.sample_rate:
                from scipy import signal
                audio = signal.resample(audio, int(len(audio) * self.sample_rate / original_sr))
            
            return self.process_audio(audio)
            
        except Exception as e:
            logger.error(f"Error processing audio file {audio_path}: {e}")
            return []
    
    def process_audio(self, audio: np.ndarray) -> List[VADSegment]:
        """Process audio array and return VAD segments."""
        frame_size = int(self.sample_rate * self.frame_duration_ms / 1000)
        
        if self.vad is None:
            # Use energy-based VAD
            is_speech_frames = self._energy_based_vad(audio, self.frame_duration_ms)
        else:
            # Use WebRTC VAD
            is_speech_frames = []
            
            # Convert to int16 for WebRTC VAD
            audio_int16 = (audio * 32767).astype(np.int16)
            
            for i in range(0, len(audio_int16) - frame_size + 1, frame_size):
                frame = audio_int16[i:i + frame_size]
                
                # Ensure frame is exactly the right size
                if len(frame) == frame_size:
                    try:
                        is_speech = self.vad.is_speech(frame.tobytes(), self.sample_rate)
                        is_speech_frames.append(is_speech)
                    except Exception as e:
                        logger.warning(f"VAD frame processing error: {e}")
                        is_speech_frames.append(False)
        
        # Convert frame-level decisions to segments
        segments = self._frames_to_segments(is_speech_frames, self.frame_duration_ms / 1000)
        
        return segments
    
    def _frames_to_segments(self, is_speech_frames: List[bool], frame_duration: float) -> List[VADSegment]:
        """Convert frame-level VAD decisions to segments."""
        if not is_speech_frames:
            return []
        
        segments = []
        current_segment_start = 0
        current_is_speech = is_speech_frames[0]
        
        for i, is_speech in enumerate(is_speech_frames[1:], 1):
            if is_speech != current_is_speech:
                # Segment boundary
                start_time = current_segment_start * frame_duration
                end_time = i * frame_duration
                segments.append(VADSegment(start_time, end_time, current_is_speech))
                
                current_segment_start = i
                current_is_speech = is_speech
        
        # Add final segment
        start_time = current_segment_start * frame_duration
        end_time = len(is_speech_frames) * frame_duration
        segments.append(VADSegment(start_time, end_time, current_is_speech))
        
        return segments
    
    def get_speech_segments(self, segments: List[VADSegment]) -> List[VADSegment]:
        """Filter to get only speech segments."""
        return [seg for seg in segments if seg.is_speech]
    
    def get_silence_segments(self, segments: List[VADSegment]) -> List[VADSegment]:
        """Filter to get only silence segments."""
        return [seg for seg in segments if not seg.is_speech]
    
    def get_speech_ratio(self, segments: List[VADSegment]) -> float:
        """Calculate ratio of speech to total time."""
        if not segments:
            return 0.0
        
        total_time = segments[-1].end - segments[0].start
        speech_time = sum(seg.duration for seg in segments if seg.is_speech)
        
        return speech_time / total_time if total_time > 0 else 0.0
    
    def merge_short_segments(self, 
                           segments: List[VADSegment], 
                           min_segment_duration: float = 0.1,
                           max_gap_duration: float = 0.2) -> List[VADSegment]:
        """Merge short segments and gaps to reduce noise."""
        if not segments:
            return []
        
        merged = []
        current_seg = segments[0]
        
        for next_seg in segments[1:]:
            # If segments are both speech and close together, merge them
            if (current_seg.is_speech and next_seg.is_speech and 
                next_seg.start - current_seg.end <= max_gap_duration):
                current_seg = VADSegment(current_seg.start, next_seg.end, True)
            # If current segment is too short, extend it or merge with next
            elif current_seg.duration < min_segment_duration:
                if next_seg.is_speech == current_seg.is_speech:
                    current_seg = VADSegment(current_seg.start, next_seg.end, current_seg.is_speech)
                else:
                    # Change short segment to match the next one
                    merged.append(VADSegment(current_seg.start, current_seg.end, next_seg.is_speech))
                    current_seg = next_seg
            else:
                merged.append(current_seg)
                current_seg = next_seg
        
        merged.append(current_seg)
        return merged


# Global VAD processor instance
vad_processor = VADProcessor()