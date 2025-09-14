import logging
import traceback
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional
import tempfile
import time
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field

# Import analysis modules
from .models.asr import asr_manager
from .models.vad import vad_processor
from .analysis.snr import snr_analyzer
from .analysis.pacing import pacing_analyzer
from .analysis.fillers import filler_analyzer
from .analysis.clarity import clarity_analyzer
from .analysis.phonemes import phonemes_analyzer
from .analysis.prosody import prosody_analyzer
from .analysis.pauses import pause_analyzer
from .analysis.tips import tips_generator
from .analysis.report import report_generator
from .analysis.preprocess import audio_preprocessor
from .utils.audio_io import audio_processor
from .utils.session_store import session_store
from .settings import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.logging.get("level", "INFO")),
    format=settings.logging.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger = logging.getLogger(__name__)


# Pydantic models for requests/responses
class SNRCheckResponse(BaseModel):
    snr_ok: bool
    snr_db: float
    message: str
    warning_level: str = Field(default="none")  # none, moderate, high, critical


class AnalysisRequest(BaseModel):
    audio_filename: str = Field(default="recording.wav")
    speaking_mode: str = Field(default="Presentation")
    save_session: bool = Field(default=True)
    session_name: Optional[str] = None


class AnalysisResponse(BaseModel):
    success: bool
    session_id: Optional[str] = None
    processing_time: float
    results: Dict[str, Any]
    tips: list[str]
    overall_score: int


class DrillRequest(BaseModel):
    phoneme_group: str
    audio_filename: str = Field(default="drill.wav")


class DrillResponse(BaseModel):
    success: bool
    phoneme_group: str
    risk_score: float
    improvement: Optional[float] = None
    suggestion: str
    processing_time: float


# Global state management
app_state = {
    "temp_files": set(),
    "processing_count": 0
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    logger.info("Starting Echo API server")
    logger.info(f"Available ASR engines: {asr_manager.get_available_engines()}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Echo API server")
    # Cleanup temp files
    for temp_file in app_state["temp_files"]:
        try:
            if Path(temp_file).exists():
                Path(temp_file).unlink()
        except Exception as e:
            logger.warning(f"Failed to cleanup temp file {temp_file}: {e}")


# Create FastAPI app
app = FastAPI(
    title="Echo Speech Analysis API",
    description="Real-time speech analysis for accent and clarity feedback",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def cleanup_temp_file(file_path: Path) -> None:
    """Background task to cleanup temp files."""
    try:
        if file_path.exists():
            file_path.unlink()
            app_state["temp_files"].discard(str(file_path))
            logger.debug(f"Cleaned up temp file: {file_path}")
    except Exception as e:
        logger.warning(f"Failed to cleanup temp file {file_path}: {e}")


@app.get("/")
async def root():
    """API health check."""
    return {
        "message": "Echo Speech Analysis API",
        "version": "1.0.0",
        "status": "healthy",
        "available_engines": asr_manager.get_available_engines()
    }


@app.post("/snr_check", response_model=SNRCheckResponse)
async def check_snr(
    background_tasks: BackgroundTasks,
    audio_file: UploadFile = File(...)
) -> SNRCheckResponse:
    """Check audio signal-to-noise ratio before analysis."""
    temp_file = None
    
    try:
        logger.info(f"SNR check requested for file: {audio_file.filename}")
        
        # Validate file
        if not audio_file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        # Save uploaded file
        file_data = await audio_file.read()
        temp_file = audio_processor.save_uploaded_file(file_data, audio_file.filename)
        app_state["temp_files"].add(str(temp_file))
        
        # Load and preprocess audio
        audio, sr = audio_processor.load_audio_file(temp_file)
        audio = audio_processor.trim_to_max_duration(audio, sr)
        
        # Analyze SNR
        snr_results = snr_analyzer.analyze_audio(audio, sr)
        
        # Generate response
        snr_db = snr_results.get("snr_db", 0)
        snr_ok = snr_results.get("snr_ok", False)
        
        if snr_db < 10:
            warning_level = "critical"
            message = "Very high background noise detected. Recording quality is poor."
        elif snr_db < 15:
            warning_level = "high"
            message = "High background noise detected. Consider recording in a quieter environment."
        elif snr_db < 20:
            warning_level = "moderate"
            message = "Moderate background noise detected. Quality is acceptable but could be improved."
        else:
            warning_level = "none"
            message = "Good audio quality detected."
        
        # Schedule cleanup
        background_tasks.add_task(cleanup_temp_file, temp_file)
        
        return SNRCheckResponse(
            snr_ok=snr_ok,
            snr_db=snr_db,
            message=message,
            warning_level=warning_level
        )
        
    except Exception as e:
        logger.error(f"SNR check failed: {e}")
        logger.error(traceback.format_exc())
        
        if temp_file and temp_file.exists():
            background_tasks.add_task(cleanup_temp_file, temp_file)
        
        raise HTTPException(status_code=500, detail=f"SNR analysis failed: {str(e)}")


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_speech(
    background_tasks: BackgroundTasks,
    audio_file: UploadFile = File(...),
    speaking_mode: str = "Presentation",
    save_session: bool = True,
    session_name: Optional[str] = None
) -> AnalysisResponse:
    """Perform comprehensive speech analysis."""
    temp_file = None
    start_time = time.time()
    
    try:
        app_state["processing_count"] += 1
        logger.info(f"Analysis requested for file: {audio_file.filename} (mode: {speaking_mode})")
        
        # Validate inputs
        if not audio_file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        if speaking_mode not in ["Presentation", "Interview", "Casual"]:
            speaking_mode = "Presentation"
        
        # Save uploaded file
        file_data = await audio_file.read()
        temp_file = audio_processor.save_uploaded_file(file_data, audio_file.filename)
        app_state["temp_files"].add(str(temp_file))
        
        # Load and preprocess audio
        audio, sr = audio_processor.load_audio_file(temp_file)
        audio = audio_processor.trim_to_max_duration(audio, sr)
        
        # Preprocess audio (normalize, etc.)
        audio = audio_processor.normalize_volume(audio)
        
        logger.info(f"Processing audio: {len(audio)/sr:.1f}s duration at {sr}Hz")
        
        # Run ASR
        asr_result = asr_manager.transcribe(str(temp_file))
        logger.info(f"ASR completed: {len(asr_result.words)} words recognized")
        
        if not asr_result.text.strip():
            raise HTTPException(status_code=400, detail="No speech detected in audio")
        
        # Run VAD
        vad_segments = vad_processor.process_audio(audio)
        speech_segments = [seg for seg in vad_segments if seg.is_speech]
        total_speech_duration = sum(seg.duration for seg in speech_segments)
        
        if total_speech_duration < 1.0:
            raise HTTPException(status_code=400, detail="Insufficient speech detected")
        
        logger.info(f"VAD completed: {len(speech_segments)} speech segments, {total_speech_duration:.1f}s total")
        
        # Run all analyses
        analysis_results = {}
        
        # SNR analysis
        analysis_results["snr_analysis"] = snr_analyzer.analyze_audio(audio, sr)
        
        # Pacing analysis
        analysis_results["pacing_analysis"] = pacing_analyzer.analyze(asr_result, vad_segments)
        
        # Filler analysis
        analysis_results["filler_analysis"] = filler_analyzer.analyze(asr_result, total_speech_duration)
        
        # Clarity analysis
        analysis_results["clarity_analysis"] = clarity_analyzer.analyze(asr_result)
        
        # Phoneme analysis
        analysis_results["phoneme_analysis"] = phonemes_analyzer.analyze(asr_result)
        
        # Prosody analysis
        analysis_results["prosody_analysis"] = prosody_analyzer.analyze(audio, sr, asr_result)
        
        # Pause analysis
        analysis_results["pause_analysis"] = pause_analyzer.analyze(vad_segments, asr_result)
        
        # Add metadata
        analysis_results["metadata"] = {
            "audio_filename": audio_file.filename,
            "audio_duration": total_speech_duration,
            "word_count": len(asr_result.words),
            "speaking_mode": speaking_mode,
            "processing_time": time.time() - start_time,
            "analysis_timestamp": time.time()
        }
        
        # Generate personalized tips
        tips = tips_generator.generate_personalized_tips(
            analysis_results, 
            max_tips=7, 
            speaking_mode=speaking_mode
        )
        
        # Calculate overall score
        overall_score = _calculate_overall_score(analysis_results)
        
        # Save session if requested
        session_id = None
        if save_session:
            try:
                session_id = session_store.save_session(
                    analysis_results,
                    audio_file.filename,
                    session_name
                )
            except Exception as e:
                logger.warning(f"Failed to save session: {e}")
        
        processing_time = time.time() - start_time
        logger.info(f"Analysis completed in {processing_time:.2f}s")
        
        # Schedule cleanup
        background_tasks.add_task(cleanup_temp_file, temp_file)
        
        return AnalysisResponse(
            success=True,
            session_id=session_id,
            processing_time=processing_time,
            results=analysis_results,
            tips=tips,
            overall_score=overall_score
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        logger.error(traceback.format_exc())
        
        if temp_file and temp_file.exists():
            background_tasks.add_task(cleanup_temp_file, temp_file)
        
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    
    finally:
        app_state["processing_count"] -= 1


@app.post("/drill_analyze", response_model=DrillResponse)
async def analyze_drill(
    background_tasks: BackgroundTasks,
    audio_file: UploadFile = File(...),
    phoneme_group: str = "TH"
) -> DrillResponse:
    """Analyze phoneme drill practice (faster, targeted analysis)."""
    temp_file = None
    start_time = time.time()
    
    try:
        logger.info(f"Drill analysis for {phoneme_group} requested: {audio_file.filename}")
        
        # Validate inputs
        if not audio_file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        valid_groups = ["TH", "RL", "WV", "ZH", "NG"]
        if phoneme_group not in valid_groups:
            raise HTTPException(status_code=400, detail=f"Invalid phoneme group. Must be one of: {valid_groups}")
        
        # Save uploaded file
        file_data = await audio_file.read()
        temp_file = audio_processor.save_uploaded_file(file_data, audio_file.filename)
        app_state["temp_files"].add(str(temp_file))
        
        # Load audio (no need for full preprocessing for drills)
        audio, sr = audio_processor.load_audio_file(temp_file)
        audio = audio_processor.trim_to_max_duration(audio, sr)
        
        # Quick ASR (use fastest available engine)
        asr_result = asr_manager.transcribe(str(temp_file))
        
        if not asr_result.text.strip():
            raise HTTPException(status_code=400, detail="No speech detected in drill")
        
        # Run targeted phoneme analysis
        drill_result = phonemes_analyzer.analyze_drill(asr_result, phoneme_group)
        
        if "error" in drill_result:
            raise HTTPException(status_code=500, detail=drill_result["error"])
        
        risk_score = drill_result.get("risk_score", 0)
        
        # Get previous scores for improvement calculation
        previous_history = session_store.get_drill_history(phoneme_group, limit=1)
        improvement = None
        
        if previous_history:
            previous_score = previous_history[0]["risk_score"]
            improvement = previous_score - risk_score  # Positive = improvement (risk reduction)
        
        # Save drill result
        session_store.save_drill_result(
            phoneme_group,
            risk_score,
            asr_result.text[:100]  # First 100 chars as drill text
        )
        
        # Get suggestion
        suggestion = tips_generator.get_drill_tip(phoneme_group, risk_score)
        
        processing_time = time.time() - start_time
        logger.info(f"Drill analysis completed in {processing_time:.2f}s")
        
        # Schedule cleanup
        background_tasks.add_task(cleanup_temp_file, temp_file)
        
        return DrillResponse(
            success=True,
            phoneme_group=phoneme_group,
            risk_score=risk_score,
            improvement=improvement,
            suggestion=suggestion,
            processing_time=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Drill analysis failed: {e}")
        
        if temp_file and temp_file.exists():
            background_tasks.add_task(cleanup_temp_file, temp_file)
        
        raise HTTPException(status_code=500, detail=f"Drill analysis failed: {str(e)}")


@app.get("/sessions")
async def get_recent_sessions(limit: int = 10):
    """Get list of recent analysis sessions."""
    try:
        sessions = session_store.get_recent_sessions(limit)
        return {"success": True, "sessions": sessions}
    except Exception as e:
        logger.error(f"Failed to retrieve sessions: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve sessions")


@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Get specific analysis session."""
    try:
        session = session_store.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"success": True, "session": session}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve session")


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete analysis session."""
    try:
        success = session_store.delete_session(session_id)
        if not success:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"success": True, "message": "Session deleted"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete session")


@app.post("/report/{session_id}")
async def generate_report(session_id: str, format: str = "pdf"):
    """Generate analysis report for a session."""
    try:
        # Get session data
        session = session_store.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Generate report
        report_result = report_generator.generate_report(
            session["analysis_results"],
            session["audio_filename"],
            format
        )
        
        if not report_result["success"]:
            raise HTTPException(status_code=500, detail=report_result.get("error", "Report generation failed"))
        
        # Return file
        report_path = Path(report_result["report_path"])
        if not report_path.exists():
            raise HTTPException(status_code=500, detail="Report file not found")
        
        return FileResponse(
            path=str(report_path),
            filename=report_path.name,
            media_type="application/pdf" if format == "pdf" else "application/json"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Report generation failed for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Report generation failed")


@app.get("/drill/history/{phoneme_group}")
async def get_drill_history(phoneme_group: str, limit: int = 10):
    """Get drill practice history for phoneme group."""
    try:
        history = session_store.get_drill_history(phoneme_group, limit)
        progress = session_store.get_drill_progress(phoneme_group)
        
        return {
            "success": True,
            "phoneme_group": phoneme_group,
            "history": history,
            "progress": progress
        }
    except Exception as e:
        logger.error(f"Failed to retrieve drill history for {phoneme_group}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve drill history")


@app.get("/status")
async def get_api_status():
    """Get API status and health information."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "processing_count": app_state["processing_count"],
        "temp_files": len(app_state["temp_files"]),
        "available_engines": asr_manager.get_available_engines(),
        "settings": {
            "max_audio_duration": settings.audio.get("max_duration_seconds", 90),
            "supported_formats": settings.audio.get("supported_formats", []),
            "snr_threshold": settings.analysis.snr_db_threshold
        }
    }


def _calculate_overall_score(analysis_results: Dict[str, Any]) -> int:
    """Calculate overall speech score (0-100)."""
    scores = []
    
    # Collect individual scores
    if "clarity_analysis" in analysis_results:
        scores.append(analysis_results["clarity_analysis"].get("clarity_score", 0))
    
    if "pacing_analysis" in analysis_results:
        wpm = analysis_results["pacing_analysis"].get("overall_wpm", 0)
        target_range = [140, 160]
        if target_range[0] <= wpm <= target_range[1]:
            pacing_score = 100
        elif wpm < target_range[0]:
            pacing_score = max(0, 100 - (target_range[0] - wpm) * 2)
        else:
            pacing_score = max(0, 100 - (wpm - target_range[1]) * 1.5)
        scores.append(pacing_score)
    
    if "filler_analysis" in analysis_results:
        fillers_per_min = analysis_results["filler_analysis"].get("fillers_per_minute", 0)
        filler_score = max(0, 100 - fillers_per_min * 15)
        scores.append(filler_score)
    
    if "prosody_analysis" in analysis_results:
        stability = analysis_results["prosody_analysis"].get("stability_metrics", {})
        prosody_score = stability.get("overall_stability_score", 50)
        scores.append(prosody_score)
    
    return round(sum(scores) / len(scores)) if scores else 0


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.app.get("host", "0.0.0.0"),
        port=settings.app.get("port", 8000),
        reload=settings.app.get("debug", False)
    )