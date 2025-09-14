import os
import logging
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
from abc import ABC, abstractmethod

try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False

try:
    import vosk
    import json
    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False

try:
    import whisperx
    WHISPERX_AVAILABLE = True
except ImportError:
    WHISPERX_AVAILABLE = False

from ..settings import settings

logger = logging.getLogger(__name__)


class ASRResult:
    """Container for ASR results."""
    def __init__(self, 
                 text: str,
                 words: List[Dict[str, Any]] = None,
                 segments: List[Dict[str, Any]] = None,
                 language: str = "en",
                 confidence: float = 0.0):
        self.text = text
        self.words = words or []
        self.segments = segments or []
        self.language = language
        self.confidence = confidence


class ASREngine(ABC):
    """Abstract base class for ASR engines."""
    
    @abstractmethod
    def transcribe(self, audio_path: str) -> ASRResult:
        """Transcribe audio file to text with timestamps."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the ASR engine is available."""
        pass


class FasterWhisperEngine(ASREngine):
    """Faster-Whisper ASR engine with CPU optimization."""
    
    def __init__(self):
        self.model = None
        self.config = settings.asr.faster_whisper
        
    def _load_model(self):
        """Lazy load the Faster-Whisper model."""
        if self.model is None and FASTER_WHISPER_AVAILABLE:
            try:
                self.model = WhisperModel(
                    model_size_or_path=self.config.get("model_size", "small.en"),
                    device=self.config.get("device", "cpu"),
                    compute_type=self.config.get("compute_type", "int8"),
                    cpu_threads=os.cpu_count(),
                    num_workers=1
                )
                logger.info(f"Loaded Faster-Whisper model: {self.config.get('model_size', 'small.en')}")
            except Exception as e:
                logger.error(f"Failed to load Faster-Whisper model: {e}")
                self.model = None
    
    def transcribe(self, audio_path: str) -> ASRResult:
        """Transcribe using Faster-Whisper."""
        self._load_model()
        if self.model is None:
            raise RuntimeError("Faster-Whisper model not available")
        
        try:
            segments, info = self.model.transcribe(
                audio_path,
                language=self.config.get("language", "en"),
                beam_size=self.config.get("beam_size", 5),
                word_timestamps=True,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )
            
            # Convert segments to our format
            words = []
            segments_list = []
            full_text = []
            
            for segment in segments:
                segment_dict = {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip(),
                    "confidence": segment.avg_logprob if hasattr(segment, 'avg_logprob') else 0.0
                }
                segments_list.append(segment_dict)
                full_text.append(segment.text.strip())
                
                # Extract word-level timestamps
                if hasattr(segment, 'words') and segment.words:
                    for word in segment.words:
                        words.append({
                            "word": word.word.strip(),
                            "start": word.start,
                            "end": word.end,
                            "confidence": word.probability if hasattr(word, 'probability') else 0.0
                        })
            
            # Calculate overall confidence
            overall_confidence = np.mean([w["confidence"] for w in words]) if words else 0.0
            
            return ASRResult(
                text=" ".join(full_text),
                words=words,
                segments=segments_list,
                language=info.language if hasattr(info, 'language') else "en",
                confidence=overall_confidence
            )
            
        except Exception as e:
            logger.error(f"Faster-Whisper transcription failed: {e}")
            raise
    
    def is_available(self) -> bool:
        """Check if Faster-Whisper is available."""
        return FASTER_WHISPER_AVAILABLE


class VoskEngine(ASREngine):
    """Vosk ASR engine as fallback."""
    
    def __init__(self):
        self.model = None
        self.rec = None
        self.config = settings.asr.vosk
        
    def _load_model(self):
        """Lazy load the Vosk model."""
        if self.model is None and VOSK_AVAILABLE:
            model_path = self.config.get("model_path", "models/vosk-model-small-en-us-0.15")
            if not os.path.exists(model_path):
                logger.warning(f"Vosk model not found at {model_path}")
                return
                
            try:
                vosk.SetLogLevel(-1)  # Suppress Vosk logging
                self.model = vosk.Model(model_path)
                logger.info(f"Loaded Vosk model from: {model_path}")
            except Exception as e:
                logger.error(f"Failed to load Vosk model: {e}")
                self.model = None
    
    def transcribe(self, audio_path: str) -> ASRResult:
        """Transcribe using Vosk."""
        self._load_model()
        if self.model is None:
            raise RuntimeError("Vosk model not available")
        
        try:
            import soundfile as sf
            
            # Load audio
            data, samplerate = sf.read(audio_path)
            if len(data.shape) > 1:
                data = data[:, 0]  # Take first channel if stereo
            
            # Resample to 16kHz if needed
            if samplerate != 16000:
                from scipy import signal
                data = signal.resample(data, int(len(data) * 16000 / samplerate))
            
            # Convert to int16
            audio_int16 = (data * 32767).astype(np.int16)
            
            # Create recognizer
            rec = vosk.KaldiRecognizer(self.model, 16000)
            rec.SetWords(True)
            
            # Process audio in chunks
            chunk_size = 4000
            results = []
            
            for i in range(0, len(audio_int16), chunk_size):
                chunk = audio_int16[i:i+chunk_size].tobytes()
                if rec.AcceptWaveform(chunk):
                    result = json.loads(rec.Result())
                    if result.get("text"):
                        results.append(result)
            
            # Final result
            final_result = json.loads(rec.FinalResult())
            if final_result.get("text"):
                results.append(final_result)
            
            # Combine results
            words = []
            full_text = []
            
            for result in results:
                if "result" in result:
                    for word_info in result["result"]:
                        words.append({
                            "word": word_info["word"],
                            "start": word_info["start"],
                            "end": word_info["end"],
                            "confidence": word_info.get("conf", 0.0)
                        })
                if "text" in result:
                    full_text.append(result["text"])
            
            overall_confidence = np.mean([w["confidence"] for w in words]) if words else 0.0
            
            return ASRResult(
                text=" ".join(full_text),
                words=words,
                segments=[],
                language="en",
                confidence=overall_confidence
            )
            
        except Exception as e:
            logger.error(f"Vosk transcription failed: {e}")
            raise
    
    def is_available(self) -> bool:
        """Check if Vosk is available."""
        return VOSK_AVAILABLE and os.path.exists(self.config.get("model_path", ""))


class WhisperXEngine(ASREngine):
    """WhisperX engine with forced alignment (optional)."""
    
    def __init__(self):
        self.model = None
        self.align_model = None
        self.config = settings.asr.whisperx
        
    def is_available(self) -> bool:
        """Check if WhisperX is available and enabled."""
        return WHISPERX_AVAILABLE and self.config.get("enabled", False)
    
    def transcribe(self, audio_path: str) -> ASRResult:
        """Transcribe using WhisperX with alignment."""
        if not self.is_available():
            raise RuntimeError("WhisperX not available or not enabled")
        
        try:
            # This is a placeholder - WhisperX implementation would go here
            # For now, fallback to other engines
            raise RuntimeError("WhisperX implementation not complete")
            
        except Exception as e:
            logger.error(f"WhisperX transcription failed: {e}")
            raise


class ASRManager:
    """Manager for ASR engines with fallback logic."""
    
    def __init__(self):
        self.engines = {
            "faster_whisper": FasterWhisperEngine(),
            "vosk": VoskEngine(),
            "whisperx": WhisperXEngine()
        }
        self.default_engine = settings.asr.default_model
        
    def get_available_engines(self) -> List[str]:
        """Get list of available ASR engines."""
        return [name for name, engine in self.engines.items() if engine.is_available()]
    
    def transcribe(self, audio_path: str, engine_name: Optional[str] = None) -> ASRResult:
        """Transcribe audio using specified or default engine with fallbacks."""
        if engine_name is None:
            engine_name = self.default_engine
        
        # Try specified engine first
        if engine_name in self.engines and self.engines[engine_name].is_available():
            try:
                logger.info(f"Trying ASR with {engine_name}")
                return self.engines[engine_name].transcribe(audio_path)
            except Exception as e:
                logger.warning(f"ASR engine {engine_name} failed: {e}")
        
        # Fallback to available engines
        available_engines = self.get_available_engines()
        for fallback_engine in available_engines:
            if fallback_engine != engine_name:
                try:
                    logger.info(f"Falling back to ASR engine: {fallback_engine}")
                    return self.engines[fallback_engine].transcribe(audio_path)
                except Exception as e:
                    logger.warning(f"Fallback ASR engine {fallback_engine} failed: {e}")
        
        raise RuntimeError("No ASR engines available or all failed")


# Global ASR manager instance
asr_manager = ASRManager()