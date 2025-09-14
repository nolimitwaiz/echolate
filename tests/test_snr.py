import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from app.analysis.snr import SNRAnalyzer
from app.models.vad import VADProcessor, VADSegment


class TestSNRAnalyzer:
    
    @pytest.fixture
    def snr_analyzer(self):
        return SNRAnalyzer()
    
    @pytest.fixture
    def vad_processor(self):
        return VADProcessor()
    
    def test_snr_analyzer_initialization(self, snr_analyzer):
        assert snr_analyzer.snr_threshold == 20.0
        assert snr_analyzer.epsilon == 1e-10
    
    def test_clean_audio_high_snr(self, snr_analyzer):
        # Generate clean speech-like signal
        sample_rate = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Create speech-like signal (varying amplitude sine wave)
        speech_signal = 0.5 * np.sin(2 * np.pi * 440 * t) * (1 + 0.5 * np.sin(2 * np.pi * 5 * t))
        
        # Add minimal noise
        noise = 0.01 * np.random.randn(len(speech_signal))
        audio = speech_signal + noise
        
        result = snr_analyzer.analyze_audio(audio, sample_rate)
        
        assert result["snr_ok"] is True
        assert result["snr_db"] > 15  # Should be high SNR
        assert result["signal_rms"] > result["noise_rms"]
    
    def test_noisy_audio_low_snr(self, snr_analyzer):
        # Generate noisy signal
        sample_rate = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Weak speech signal with loud noise
        speech_signal = 0.1 * np.sin(2 * np.pi * 440 * t)
        noise = 0.5 * np.random.randn(len(speech_signal))
        audio = speech_signal + noise
        
        result = snr_analyzer.analyze_audio(audio, sample_rate)
        
        assert result["snr_ok"] is False
        assert result["snr_db"] < 20  # Should be low SNR
    
    def test_silent_audio(self, snr_analyzer):
        # Test with very quiet audio
        sample_rate = 16000
        duration = 1.0
        audio = 0.001 * np.random.randn(int(sample_rate * duration))
        
        result = snr_analyzer.analyze_audio(audio, sample_rate)
        
        # Should handle gracefully
        assert "snr_db" in result
        assert "error" not in result
    
    def test_empty_audio(self, snr_analyzer):
        # Test with empty audio array
        audio = np.array([])
        sample_rate = 16000
        
        result = snr_analyzer.analyze_audio(audio, sample_rate)
        
        # Should return error or handle gracefully
        assert result["snr_ok"] is False
    
    def test_calculate_rms_for_segments(self, snr_analyzer):
        # Test RMS calculation for specific segments
        sample_rate = 16000
        duration = 2.0
        audio = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(sample_rate * duration)))
        
        # Create mock segments
        segments = [
            VADSegment(0.0, 1.0, True),  # Speech segment
            VADSegment(1.0, 2.0, False)  # Silence segment
        ]
        
        speech_rms = snr_analyzer._calculate_rms_for_segments(audio, [segments[0]], sample_rate)
        silence_rms = snr_analyzer._calculate_rms_for_segments(audio, [segments[1]], sample_rate)
        
        assert speech_rms > 0
        assert silence_rms >= 0
        assert speech_rms > silence_rms  # Speech should have higher RMS
    
    def test_snr_warning_messages(self, snr_analyzer):
        # Test warning message generation
        
        # High SNR - no warning
        message = snr_analyzer.get_snr_warning_message(25.0)
        assert message == ""
        
        # Moderate SNR
        message = snr_analyzer.get_snr_warning_message(18.0)
        assert "moderate" in message.lower() or "⚠️" in message
        
        # Low SNR
        message = snr_analyzer.get_snr_warning_message(8.0)
        assert "high" in message.lower() or "🚨" in message
    
    def test_should_block_analysis(self, snr_analyzer):
        # Test analysis blocking logic
        
        assert snr_analyzer.should_block_analysis(5.0) is True  # Very low SNR
        assert snr_analyzer.should_block_analysis(15.0) is False  # Acceptable SNR
        assert snr_analyzer.should_block_analysis(25.0) is False  # Good SNR
    
    @pytest.mark.parametrize("snr_threshold", [15.0, 20.0, 25.0])
    def test_different_thresholds(self, snr_threshold):
        analyzer = SNRAnalyzer()
        analyzer.snr_threshold = snr_threshold
        
        # Generate test signal
        sample_rate = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Signal with known SNR around 18 dB
        speech = 0.3 * np.sin(2 * np.pi * 440 * t)
        noise = 0.05 * np.random.randn(len(speech))
        audio = speech + noise
        
        result = analyzer.analyze_audio(audio, sample_rate)
        
        # Check threshold behavior
        expected_ok = result["snr_db"] >= snr_threshold
        assert result["snr_ok"] == expected_ok


if __name__ == "__main__":
    pytest.main([__file__])