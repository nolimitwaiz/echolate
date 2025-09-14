import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from app.analysis.fillers import FillerAnalyzer
from app.models.asr import ASRResult


class TestFillerAnalyzer:
    
    @pytest.fixture
    def filler_analyzer(self):
        return FillerAnalyzer()
    
    @pytest.fixture
    def sample_asr_result_with_fillers(self):
        # Create mock ASR result with fillers
        words = [
            {"word": "hello", "start": 0.0, "end": 0.5, "confidence": 0.9},
            {"word": "um", "start": 0.6, "end": 0.8, "confidence": 0.7},
            {"word": "this", "start": 0.9, "end": 1.2, "confidence": 0.95},
            {"word": "is", "start": 1.3, "end": 1.5, "confidence": 0.98},
            {"word": "like", "start": 1.6, "end": 1.9, "confidence": 0.8},
            {"word": "a", "start": 2.0, "end": 2.1, "confidence": 0.99},
            {"word": "test", "start": 2.2, "end": 2.6, "confidence": 0.95},
            {"word": "you", "start": 2.7, "end": 2.9, "confidence": 0.9},
            {"word": "know", "start": 3.0, "end": 3.4, "confidence": 0.85}
        ]
        
        text = "hello um this is like a test you know"
        
        return ASRResult(
            text=text,
            words=words,
            segments=[],
            language="en",
            confidence=0.9
        )
    
    @pytest.fixture 
    def sample_asr_result_clean(self):
        # Create mock ASR result without fillers
        words = [
            {"word": "this", "start": 0.0, "end": 0.3, "confidence": 0.95},
            {"word": "is", "start": 0.4, "end": 0.6, "confidence": 0.98},
            {"word": "a", "start": 0.7, "end": 0.8, "confidence": 0.99},
            {"word": "clean", "start": 0.9, "end": 1.3, "confidence": 0.96},
            {"word": "sentence", "start": 1.4, "end": 1.9, "confidence": 0.94}
        ]
        
        text = "this is a clean sentence"
        
        return ASRResult(
            text=text,
            words=words,
            segments=[],
            language="en",
            confidence=0.95
        )
    
    def test_filler_analyzer_initialization(self, filler_analyzer):
        assert len(filler_analyzer.filler_patterns) > 0
        assert len(filler_analyzer.compiled_patterns) > 0
        assert len(filler_analyzer.compiled_patterns) == len(filler_analyzer.filler_patterns)
    
    def test_detect_vocal_fillers(self, filler_analyzer, sample_asr_result_with_fillers):
        speech_duration = 3.4
        result = filler_analyzer.analyze(sample_asr_result_with_fillers, speech_duration)
        
        # Should detect "um" and "like" and "you know"
        assert result["filler_count"] >= 2
        assert result["fillers_per_minute"] > 0
        
        # Check that specific fillers were found
        filler_texts = [f["text"] for f in result["filler_instances"]]
        assert any("um" in text.lower() for text in filler_texts)
        assert any("like" in text.lower() for text in filler_texts)
    
    def test_clean_speech_no_fillers(self, filler_analyzer, sample_asr_result_clean):
        speech_duration = 1.9
        result = filler_analyzer.analyze(sample_asr_result_clean, speech_duration)
        
        # Should detect no fillers
        assert result["filler_count"] == 0
        assert result["fillers_per_minute"] == 0
        assert result["assessment"] in ["excellent", "good"]
        assert len(result["filler_instances"]) == 0
    
    def test_filler_classification(self, filler_analyzer):
        # Test filler type classification
        assert filler_analyzer._classify_filler("um") == "vocal_filler"
        assert filler_analyzer._classify_filler("uh") == "vocal_filler"
        assert filler_analyzer._classify_filler("like") == "lexical_filler"
        assert filler_analyzer._classify_filler("you know") == "lexical_filler"
        assert filler_analyzer._classify_filler("and") == "discourse_marker"
        assert filler_analyzer._classify_filler("well") == "discourse_marker"
        assert filler_analyzer._classify_filler("random") == "other"
    
    def test_filler_assessment_levels(self, filler_analyzer):
        # Test assessment level determination
        assert filler_analyzer._assess_filler_usage(0.5) == "excellent"  # ≤1 per minute
        assert filler_analyzer._assess_filler_usage(1.5) == "good"       # ≤2 per minute
        assert filler_analyzer._assess_filler_usage(3.0) == "moderate"   # ≤4 per minute
        assert filler_analyzer._assess_filler_usage(5.0) == "high"       # ≤6 per minute
        assert filler_analyzer._assess_filler_usage(7.0) == "excessive"  # >6 per minute
    
    def test_metrics_calculation(self, filler_analyzer, sample_asr_result_with_fillers):
        speech_duration = 3.4
        
        # Create test fillers
        fillers = [
            {"text": "um", "type": "vocal_filler", "start": 0.6, "end": 0.8},
            {"text": "like", "type": "lexical_filler", "start": 1.6, "end": 1.9},
            {"text": "you know", "type": "lexical_filler", "start": 3.0, "end": 3.4}
        ]
        
        metrics = filler_analyzer._calculate_metrics(fillers, speech_duration)
        
        assert metrics["count"] == 3
        assert metrics["per_minute"] == pytest.approx(3 / speech_duration * 60, rel=1e-2)
        assert "vocal_filler" in metrics["types"]
        assert "lexical_filler" in metrics["types"]
        assert metrics["types"]["vocal_filler"] == 1
        assert metrics["types"]["lexical_filler"] == 2
    
    def test_recommendations_generation(self, filler_analyzer):
        # Test recommendation generation for different filler levels
        
        # Low filler usage
        low_metrics = {"per_minute": 1.0, "count": 2, "types": {"vocal_filler": 2}}
        low_recs = filler_analyzer._get_recommendations(low_metrics)
        assert len(low_recs) > 0
        assert any("good" in rec.lower() or "excellent" in rec.lower() for rec in low_recs)
        
        # High filler usage
        high_metrics = {"per_minute": 8.0, "count": 12, "types": {"vocal_filler": 8, "lexical_filler": 4}}
        high_recs = filler_analyzer._get_recommendations(high_metrics)
        assert len(high_recs) > 0
        assert any("reduce" in rec.lower() or "focus" in rec.lower() for rec in high_recs)
    
    def test_timeline_data_generation(self, filler_analyzer):
        # Test timeline data generation
        fillers = [
            {"text": "um", "type": "vocal_filler", "start": 0.6, "end": 0.8},
            {"text": "like", "type": "lexical_filler", "start": 1.6, "end": 1.9}
        ]
        duration = 3.0
        
        timeline = filler_analyzer.get_timeline_data(fillers, duration)
        
        assert len(timeline) == 2
        for item in timeline:
            assert "time" in item
            assert "duration" in item
            assert "text" in item
            assert "type" in item
    
    def test_empty_transcript(self, filler_analyzer):
        # Test with empty transcript
        empty_asr = ASRResult(text="", words=[], segments=[], language="en", confidence=0.0)
        result = filler_analyzer.analyze(empty_asr, 1.0)
        
        assert result["filler_count"] == 0
        assert result["fillers_per_minute"] == 0.0
        assert len(result["filler_instances"]) == 0
    
    def test_word_position_mapping(self, filler_analyzer, sample_asr_result_with_fillers):
        # Test building word position mapping
        transcript = sample_asr_result_with_fillers.text
        words = sample_asr_result_with_fillers.words
        
        word_positions = filler_analyzer._build_word_positions(transcript, words)
        
        assert len(word_positions) > 0
        
        # Check that positions have required fields
        for pos in word_positions:
            assert "text" in pos
            assert "char_start" in pos
            assert "char_end" in pos
            assert "start" in pos
            assert "end" in pos
    
    @pytest.mark.parametrize("filler_text,expected_type", [
        ("um", "vocal_filler"),
        ("uh", "vocal_filler"),
        ("like", "lexical_filler"),
        ("you know", "lexical_filler"),
        ("sort of", "lexical_filler"),
        ("and", "discourse_marker"),
        ("so", "discourse_marker"),
        ("unknown", "other")
    ])
    def test_filler_type_classification(self, filler_analyzer, filler_text, expected_type):
        result = filler_analyzer._classify_filler(filler_text)
        assert result == expected_type


if __name__ == "__main__":
    pytest.main([__file__])