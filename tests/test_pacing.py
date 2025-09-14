import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from app.analysis.pacing import PacingAnalyzer
from app.models.asr import ASRResult
from app.models.vad import VADSegment


class TestPacingAnalyzer:
    
    @pytest.fixture
    def pacing_analyzer(self):
        return PacingAnalyzer()
    
    @pytest.fixture
    def sample_asr_result_normal_pace(self):
        # Create ASR result with normal pacing (around 150 WPM)
        # For 150 WPM over 10 seconds = 25 words
        words = []
        word_list = ["hello", "this", "is", "a", "test", "of", "the", "speech", "recognition", "system",
                    "we", "are", "testing", "the", "pacing", "analysis", "component", "with", "normal", "speed",
                    "speaking", "at", "around", "one", "fifty"]
        
        current_time = 0.0
        for i, word in enumerate(word_list):
            # Average word duration for 150 WPM (including pauses)
            word_duration = 0.3 + (i % 3) * 0.1  # Vary word lengths
            words.append({
                "word": word,
                "start": current_time,
                "end": current_time + word_duration,
                "confidence": 0.9 + (i % 5) * 0.02
            })
            current_time += word_duration + 0.1  # Small gap between words
        
        text = " ".join(word_list)
        
        return ASRResult(
            text=text,
            words=words,
            segments=[],
            language="en",
            confidence=0.92
        )
    
    @pytest.fixture
    def sample_vad_segments_normal(self):
        # Create VAD segments for normal speech (10 seconds total, 8 seconds speech)
        return [
            VADSegment(0.0, 0.5, False),   # Initial silence
            VADSegment(0.5, 8.5, True),    # Main speech segment
            VADSegment(8.5, 10.0, False)   # Final silence
        ]
    
    @pytest.fixture
    def sample_asr_result_fast_pace(self):
        # Create ASR result with fast pacing (around 200 WPM)
        words = []
        word_list = ["this", "is", "very", "fast", "speech", "with", "quick", "pacing", "and", "rapid",
                    "delivery", "of", "words", "without", "many", "pauses", "between", "the", "individual", "words",
                    "making", "it", "sound", "rushed", "and", "hurried", "for", "the", "listener", "to", "follow"]
        
        current_time = 0.0
        for i, word in enumerate(word_list):
            word_duration = 0.2 + (i % 2) * 0.05  # Shorter words for fast speech
            words.append({
                "word": word,
                "start": current_time,
                "end": current_time + word_duration,
                "confidence": 0.85 + (i % 4) * 0.03
            })
            current_time += word_duration + 0.05  # Very short gaps
        
        text = " ".join(word_list)
        
        return ASRResult(
            text=text,
            words=words,
            segments=[],
            language="en",
            confidence=0.88
        )
    
    @pytest.fixture
    def sample_vad_segments_fast(self):
        # Create VAD segments for fast speech (mostly continuous)
        return [
            VADSegment(0.0, 0.2, False),
            VADSegment(0.2, 9.5, True),    # Long continuous speech
            VADSegment(9.5, 10.0, False)
        ]
    
    def test_pacing_analyzer_initialization(self, pacing_analyzer):
        assert pacing_analyzer.slow_threshold == 120
        assert pacing_analyzer.fast_threshold == 180
        assert pacing_analyzer.target_range == [140, 160]
    
    def test_normal_pacing_analysis(self, pacing_analyzer, sample_asr_result_normal_pace, sample_vad_segments_normal):
        result = pacing_analyzer.analyze(sample_asr_result_normal_pace, sample_vad_segments_normal)
        
        assert "overall_wpm" in result
        assert "clean_wpm" in result
        assert "speech_duration" in result
        assert "word_count" in result
        assert "assessment" in result
        
        # Should be in or near optimal range
        wpm = result["overall_wpm"]
        assert 120 < wpm < 200  # Reasonable range
        assert result["assessment"] in ["optimal", "slightly_slow", "slightly_fast"]
    
    def test_fast_pacing_analysis(self, pacing_analyzer, sample_asr_result_fast_pace, sample_vad_segments_fast):
        result = pacing_analyzer.analyze(sample_asr_result_fast_pace, sample_vad_segments_fast)
        
        wpm = result["overall_wpm"]
        assert wpm > 150  # Should be faster than normal
        
        # Assessment should reflect fast pace
        if wpm > 180:
            assert result["assessment"] == "too_fast"
        else:
            assert result["assessment"] in ["slightly_fast", "optimal"]
    
    def test_wpm_calculation(self, pacing_analyzer):
        # Test WPM calculation logic
        
        # Mock data: 10 words in 4 seconds = 150 WPM
        words = [{"word": f"word{i}", "start": i*0.4, "end": i*0.4+0.3, "confidence": 0.9} 
                for i in range(10)]
        asr_result = ASRResult(text=" ".join([w["word"] for w in words]), words=words)
        
        vad_segments = [VADSegment(0.0, 4.0, True)]  # 4 seconds of speech
        
        result = pacing_analyzer.analyze(asr_result, vad_segments)
        
        expected_wpm = (10 / 4.0) * 60  # 150 WPM
        assert abs(result["overall_wpm"] - expected_wpm) < 5  # Allow small variance
    
    def test_filler_word_detection(self, pacing_analyzer):
        # Test that filler words are correctly identified
        assert pacing_analyzer._is_filler_word("um") is True
        assert pacing_analyzer._is_filler_word("uh") is True
        assert pacing_analyzer._is_filler_word("like") is True
        assert pacing_analyzer._is_filler_word("hello") is False
        assert pacing_analyzer._is_filler_word("world") is False
    
    def test_clean_wpm_calculation(self, pacing_analyzer):
        # Test clean WPM (excluding fillers)
        words = [
            {"word": "hello", "start": 0.0, "end": 0.5, "confidence": 0.9},
            {"word": "um", "start": 0.6, "end": 0.8, "confidence": 0.7},
            {"word": "this", "start": 0.9, "end": 1.2, "confidence": 0.9},
            {"word": "is", "start": 1.3, "end": 1.5, "confidence": 0.9},
            {"word": "like", "start": 1.6, "end": 1.8, "confidence": 0.8},
            {"word": "a", "start": 1.9, "end": 2.0, "confidence": 0.9},
            {"word": "test", "start": 2.1, "end": 2.4, "confidence": 0.9}
        ]
        
        asr_result = ASRResult(text="hello um this is like a test", words=words)
        vad_segments = [VADSegment(0.0, 2.4, True)]  # 2.4 seconds
        
        result = pacing_analyzer.analyze(asr_result, vad_segments)
        
        # Should have different WPM counts for overall vs clean
        assert result["word_count"] == 7
        assert result["clean_word_count"] == 5  # Excluding "um" and "like"
        assert result["clean_wpm"] < result["overall_wpm"]
    
    def test_stability_analysis(self, pacing_analyzer):
        # Test pacing stability analysis with varying word timing
        
        # Create words with irregular timing (unstable pacing)
        words = []
        timings = [0.0, 0.3, 0.6, 0.9, 2.0, 2.2, 2.4, 2.6, 4.0, 4.1]  # Irregular gaps
        
        for i, start_time in enumerate(timings):
            words.append({
                "word": f"word{i}",
                "start": start_time,
                "end": start_time + 0.2,
                "confidence": 0.9
            })
        
        asr_result = ASRResult(
            text=" ".join([w["word"] for w in words]),
            words=words
        )
        
        result = pacing_analyzer._analyze_stability(asr_result)
        
        assert "stability_score" in result
        assert "wpm_variance" in result
        assert "stability_assessment" in result
        
        # With irregular timing, should have lower stability
        assert result["stability_score"] < 90
    
    def test_pacing_assessment(self, pacing_analyzer):
        # Test assessment categorization
        assert pacing_analyzer._assess_pacing(100) == "too_slow"
        assert pacing_analyzer._assess_pacing(130) == "slightly_slow"
        assert pacing_analyzer._assess_pacing(150) == "optimal"
        assert pacing_analyzer._assess_pacing(170) == "slightly_fast"
        assert pacing_analyzer._assess_pacing(200) == "too_fast"
    
    def test_recommendations_generation(self, pacing_analyzer):
        # Test recommendation generation for different scenarios
        
        # Fast pacing
        fast_metrics = {"overall_wpm": 200, "stability_assessment": "stable"}
        fast_recs = pacing_analyzer._get_recommendations(200, {"stability_assessment": "stable"})
        assert len(fast_recs) > 0
        assert any("slow" in rec.lower() for rec in fast_recs)
        
        # Slow pacing
        slow_recs = pacing_analyzer._get_recommendations(100, {"stability_assessment": "stable"})
        assert len(slow_recs) > 0
        assert any("faster" in rec.lower() or "energy" in rec.lower() for rec in slow_recs)
        
        # Optimal pacing
        optimal_recs = pacing_analyzer._get_recommendations(150, {"stability_assessment": "stable"})
        assert len(optimal_recs) > 0
        assert any("excellent" in rec.lower() or "perfect" in rec.lower() for rec in optimal_recs)
    
    def test_empty_words_handling(self, pacing_analyzer):
        # Test handling of empty word list
        empty_asr = ASRResult(text="", words=[], segments=[])
        vad_segments = [VADSegment(0.0, 1.0, True)]
        
        result = pacing_analyzer.analyze(empty_asr, vad_segments)
        
        assert result["overall_wpm"] == 0
        assert result["word_count"] == 0
        assert "error" not in result  # Should handle gracefully
    
    def test_no_speech_segments(self, pacing_analyzer, sample_asr_result_normal_pace):
        # Test with no speech segments (all silence)
        silence_segments = [VADSegment(0.0, 10.0, False)]
        
        result = pacing_analyzer.analyze(sample_asr_result_normal_pace, silence_segments)
        
        # Should handle gracefully when no speech detected
        assert result["overall_wpm"] == 0
        assert result["speech_duration"] == 0
    
    @pytest.mark.parametrize("wpm,expected_assessment", [
        (90, "too_slow"),
        (130, "slightly_slow"),
        (150, "optimal"),
        (170, "slightly_fast"),
        (200, "too_fast")
    ])
    def test_assessment_boundaries(self, pacing_analyzer, wpm, expected_assessment):
        assessment = pacing_analyzer._assess_pacing(wpm)
        assert assessment == expected_assessment


if __name__ == "__main__":
    pytest.main([__file__])