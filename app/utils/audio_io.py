import logging
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Tuple, Union
import soundfile as sf
import librosa
from pydub import AudioSegment
import numpy as np

from ..settings import settings

logger = logging.getLogger(__name__)


class AudioProcessor:
    """Handle audio file I/O and format conversion."""
    
    def __init__(self):
        self.supported_formats = settings.audio.get("supported_formats", ["wav", "mp3", "webm", "m4a", "flac"])
        self.target_sr = settings.audio.get("sample_rate", 16000)
        self.max_duration = settings.audio.get("max_duration_seconds", 90)
    
    def load_audio_file(self, file_path: Union[str, Path]) -> Tuple[np.ndarray, int]:
        """Load audio file and return numpy array with sample rate."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        file_extension = file_path.suffix.lower().lstrip('.')
        
        if file_extension not in self.supported_formats:
            raise ValueError(f"Unsupported audio format: {file_extension}")
        
        try:
            # Try loading with librosa first (handles most formats)
            audio, sr = librosa.load(str(file_path), sr=None)
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = librosa.to_mono(audio)
            
            logger.info(f"Loaded audio file: {file_path} ({audio.shape[0]} samples at {sr} Hz)")
            return audio, sr
            
        except Exception as e:
            logger.warning(f"Librosa failed to load {file_path}: {e}")
            
            # Fallback to pydub for problematic formats
            try:
                return self._load_with_pydub(file_path)
            except Exception as e2:
                logger.error(f"Failed to load audio file {file_path}: {e2}")
                raise
    
    def _load_with_pydub(self, file_path: Path) -> Tuple[np.ndarray, int]:
        """Load audio using pydub as fallback."""
        file_extension = file_path.suffix.lower().lstrip('.')
        
        if file_extension == "mp3":
            audio_segment = AudioSegment.from_mp3(str(file_path))
        elif file_extension == "wav":
            audio_segment = AudioSegment.from_wav(str(file_path))
        elif file_extension == "webm":
            audio_segment = AudioSegment.from_file(str(file_path), format="webm")
        elif file_extension == "m4a":
            audio_segment = AudioSegment.from_file(str(file_path), format="m4a")
        elif file_extension == "flac":
            audio_segment = AudioSegment.from_file(str(file_path), format="flac")
        elif file_extension == "aac":
            audio_segment = AudioSegment.from_file(str(file_path), format="aac")
        else:
            audio_segment = AudioSegment.from_file(str(file_path))
        
        # Convert to mono
        if audio_segment.channels > 1:
            audio_segment = audio_segment.set_channels(1)
        
        # Convert to numpy array
        samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
        samples = samples / (2**15)  # Normalize 16-bit to float32
        
        sr = audio_segment.frame_rate
        
        logger.info(f"Loaded audio with pydub: {file_path} ({len(samples)} samples at {sr} Hz)")
        return samples, sr
    
    def save_audio_file(self, audio: np.ndarray, sr: int, output_path: Union[str, Path], format: str = "wav") -> None:
        """Save audio array to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            sf.write(str(output_path), audio, sr, format=format.upper())
            logger.info(f"Saved audio file: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save audio file {output_path}: {e}")
            raise
    
    def convert_to_wav(self, input_file: Union[str, Path], output_dir: Optional[Path] = None) -> Path:
        """Convert audio file to WAV format."""
        input_path = Path(input_file)
        
        if output_dir is None:
            output_dir = input_path.parent
        
        output_path = output_dir / f"{input_path.stem}.wav"
        
        # Load and save as WAV
        audio, sr = self.load_audio_file(input_path)
        self.save_audio_file(audio, sr, output_path, format="wav")
        
        return output_path
    
    def create_temp_wav(self, audio: np.ndarray, sr: int) -> Path:
        """Create a temporary WAV file from audio array."""
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_path = Path(temp_file.name)
        temp_file.close()
        
        self.save_audio_file(audio, sr, temp_path)
        return temp_path
    
    def validate_duration(self, audio: np.ndarray, sr: int) -> bool:
        """Validate that audio duration is within limits."""
        duration = len(audio) / sr
        return duration <= self.max_duration
    
    def trim_to_max_duration(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Trim audio to maximum allowed duration."""
        max_samples = int(self.max_duration * sr)
        if len(audio) > max_samples:
            logger.warning(f"Audio trimmed from {len(audio)/sr:.1f}s to {self.max_duration}s")
            return audio[:max_samples]
        return audio
    
    def resample_audio(self, audio: np.ndarray, original_sr: int, target_sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
        """Resample audio to target sample rate."""
        if target_sr is None:
            target_sr = self.target_sr
        
        if original_sr == target_sr:
            return audio, original_sr
        
        try:
            resampled_audio = librosa.resample(audio, orig_sr=original_sr, target_sr=target_sr)
            logger.info(f"Resampled audio from {original_sr} Hz to {target_sr} Hz")
            return resampled_audio, target_sr
        except Exception as e:
            logger.error(f"Failed to resample audio: {e}")
            raise
    
    def normalize_volume(self, audio: np.ndarray, target_db: float = -20.0) -> np.ndarray:
        """Normalize audio volume to target dB level."""
        # Calculate current RMS
        rms = np.sqrt(np.mean(audio**2))
        
        if rms == 0:
            return audio
        
        # Convert target dB to linear scale
        target_rms = 10**(target_db/20)
        
        # Apply gain
        gain = target_rms / rms
        normalized = audio * gain
        
        # Prevent clipping
        max_val = np.max(np.abs(normalized))
        if max_val > 0.95:
            normalized = normalized * (0.95 / max_val)
        
        return normalized
    
    def detect_format_from_bytes(self, audio_bytes: bytes) -> str:
        """Detect audio format from byte header."""
        if audio_bytes.startswith(b'RIFF'):
            return 'wav'
        elif audio_bytes.startswith(b'ID3') or audio_bytes.startswith(b'\xff\xfb'):
            return 'mp3'
        elif audio_bytes.startswith(b'\x1a\x45\xdf\xa3'):
            return 'webm'
        elif audio_bytes.startswith(b'fLaC'):
            return 'flac'
        elif b'ftypM4A' in audio_bytes[:20]:
            return 'm4a'
        elif b'ftypmp42' in audio_bytes[:20] or b'ftypaac' in audio_bytes[:20]:
            return 'aac'
        else:
            return 'unknown'
    
    def save_uploaded_file(self, uploaded_file_data: bytes, filename: str) -> Path:
        """Save uploaded file data to temporary location."""
        # Create temporary file
        suffix = Path(filename).suffix if Path(filename).suffix else '.tmp'
        temp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        temp_path = Path(temp_file.name)
        
        # Write data
        temp_file.write(uploaded_file_data)
        temp_file.close()
        
        logger.info(f"Saved uploaded file to: {temp_path}")
        return temp_path
    
    def cleanup_temp_file(self, file_path: Path) -> None:
        """Clean up temporary file."""
        try:
            if file_path.exists():
                file_path.unlink()
                logger.debug(f"Cleaned up temp file: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to cleanup temp file {file_path}: {e}")


# Global audio processor instance
audio_processor = AudioProcessor()