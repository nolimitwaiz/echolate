#!/usr/bin/env python3
"""
Echo - Gradio Web Interface
Instant accent & clarity feedback for speech clips
"""

import gradio as gr
import asyncio
import logging
import traceback
import tempfile
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd

# Import Echo modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from app.models.asr import asr_manager
from app.models.vad import vad_processor
from app.analysis.snr import snr_analyzer
from app.analysis.pacing import pacing_analyzer
from app.analysis.fillers import filler_analyzer
from app.analysis.clarity import clarity_analyzer
from app.analysis.phonemes import phonemes_analyzer
from app.analysis.prosody import prosody_analyzer
from app.analysis.pauses import pause_analyzer
from app.analysis.tips import tips_generator
from app.analysis.report import report_generator
from app.utils.audio_io import audio_processor
from app.utils.plotting import plot_generator
from app.utils.session_store import session_store
from app.settings import settings

from webui.components import (
    create_score_display, create_metric_card, create_tips_display,
    create_phoneme_risk_display, create_timeline_events, format_session_summary,
    create_drill_progress_display, create_snr_warning_modal, get_phoneme_practice_text,
    create_transcript_display, CUSTOM_CSS
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global state
app_state = {
    "current_session_id": None,
    "demo_mode": False,
    "processing": False
}


def check_audio_quality(audio_file) -> Tuple[bool, str, float]:
    """Check audio quality before analysis."""
    if audio_file is None:
        return False, "No audio file provided", 0.0
    
    try:
        logger.info(f"Checking audio quality for: {audio_file}")
        
        # Load audio
        audio, sr = audio_processor.load_audio_file(audio_file)
        audio = audio_processor.trim_to_max_duration(audio, sr)
        
        # Run SNR analysis
        snr_results = snr_analyzer.analyze_audio(audio, sr)
        snr_db = snr_results.get("snr_db", 0)
        snr_ok = snr_results.get("snr_ok", False)
        
        if snr_db < 10:
            message = f"🚨 Very poor audio quality (SNR: {snr_db:.1f} dB). Please record in a much quieter environment."
        elif snr_db < 15:
            message = f"⚠️ Poor audio quality (SNR: {snr_db:.1f} dB). Consider recording in a quieter space."
        elif snr_db < 20:
            message = f"⚠️ Moderate audio quality (SNR: {snr_db:.1f} dB). Quality is acceptable but could be improved."
        else:
            message = f"✅ Good audio quality (SNR: {snr_db:.1f} dB). Ready for analysis."
        
        return snr_ok, message, snr_db
        
    except Exception as e:
        logger.error(f"Audio quality check failed: {e}")
        return False, f"❌ Error checking audio quality: {str(e)}", 0.0


def analyze_speech_full(audio_file, speaking_mode: str = "Presentation", 
                       session_name: str = None, save_session: bool = True) -> Tuple[str, str, str, str, str, str]:
    """Perform comprehensive speech analysis."""
    if app_state["processing"]:
        return "❌ Another analysis is in progress. Please wait.", "", "", "", ""
    
    if audio_file is None:
        return "❌ Please upload or record an audio file first.", "", "", "", "", ""
    
    try:
        app_state["processing"] = True
        logger.info(f"Starting comprehensive analysis for: {audio_file}")
        
        # Load and preprocess audio
        audio, sr = audio_processor.load_audio_file(audio_file)
        audio = audio_processor.trim_to_max_duration(audio, sr)
        audio = audio_processor.normalize_volume(audio)
        
        logger.info(f"Processing audio: {len(audio)/sr:.1f}s duration at {sr}Hz")
        
        # Run ASR
        asr_result = asr_manager.transcribe(audio_file)
        logger.info(f"ASR completed: {len(asr_result.words)} words recognized")
        
        if not asr_result.text.strip():
            return "❌ No speech detected in the audio. Please try recording again.", "", "", "", "", ""
        
        # Run VAD
        vad_segments = vad_processor.process_audio(audio)
        speech_segments = [seg for seg in vad_segments if seg.is_speech]
        total_speech_duration = sum(seg.duration for seg in speech_segments)
        
        if total_speech_duration < 1.0:
            return "❌ Insufficient speech detected (less than 1 second).", "", "", "", "", ""
        
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
            "audio_duration": total_speech_duration,
            "word_count": len(asr_result.words),
            "speaking_mode": speaking_mode,
            "transcript": asr_result.text
        }
        
        # Generate personalized tips
        tips = tips_generator.generate_personalized_tips(
            analysis_results, 
            max_tips=7, 
            speaking_mode=speaking_mode
        )
        
        # Calculate overall score
        overall_score = calculate_overall_score(analysis_results)
        
        # Save session if requested
        if save_session:
            try:
                session_id = session_store.save_session(
                    analysis_results,
                    Path(audio_file).name,
                    session_name
                )
                app_state["current_session_id"] = session_id
                logger.info(f"Session saved with ID: {session_id}")
            except Exception as e:
                logger.warning(f"Failed to save session: {e}")
        
        # Create visualizations
        overview_chart = None
        timeline_chart = None
        phoneme_chart = None
        
        try:
            if plot_generator:
                overview_chart = plot_generator.create_overview_radar(analysis_results)
                timeline_chart = plot_generator.create_timeline_chart(analysis_results)
                phoneme_chart = plot_generator.create_phoneme_risk_chart(analysis_results["phoneme_analysis"])
        except Exception as e:
            logger.warning(f"Chart generation failed: {e}")
        
        # Format results for display
        overall_display = create_score_display(overall_score, "Overall Score")
        
        # Create metrics cards
        metrics_html = create_analysis_metrics(analysis_results)
        
        # Create tips display
        tips_html = create_tips_display(tips)
        
        # Create detailed analysis
        detailed_html = create_detailed_analysis(analysis_results)
        
        # Create timeline
        timeline_html = create_timeline_events(analysis_results)
        
        # Create transcript display
        transcript_html = create_transcript_display(
            asr_result.text,
            len(asr_result.words),
            total_speech_duration
        )
        
        logger.info("Analysis completed successfully")
        
        return overall_display, metrics_html, tips_html, detailed_html, timeline_html, transcript_html
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        logger.error(traceback.format_exc())
        return f"❌ Analysis failed: {str(e)}", "", "", "", "", ""
    
    finally:
        app_state["processing"] = False


def analyze_drill(audio_file, phoneme_group: str) -> Tuple[str, str]:
    """Analyze phoneme drill practice."""
    if audio_file is None:
        return "❌ Please record audio for drill practice.", ""
    
    try:
        logger.info(f"Starting drill analysis for {phoneme_group}: {audio_file}")
        
        # Load audio
        audio, sr = audio_processor.load_audio_file(audio_file)
        audio = audio_processor.trim_to_max_duration(audio, sr)
        
        # Quick ASR
        asr_result = asr_manager.transcribe(audio_file)
        
        if not asr_result.text.strip():
            return "❌ No speech detected in the drill. Please try again.", ""
        
        # Run targeted phoneme analysis
        drill_result = phonemes_analyzer.analyze_drill(asr_result, phoneme_group)
        
        if "error" in drill_result:
            return f"❌ Drill analysis failed: {drill_result['error']}", ""
        
        risk_score = drill_result.get("risk_score", 0)
        
        # Get previous scores for improvement calculation
        previous_history = session_store.get_drill_history(phoneme_group, limit=1)
        improvement = None
        
        if previous_history:
            previous_score = previous_history[0]["risk_score"]
            improvement = previous_score - risk_score  # Positive = improvement
        
        # Save drill result
        session_store.save_drill_result(
            phoneme_group,
            risk_score,
            asr_result.text[:100]
        )
        
        # Get updated progress
        progress_data = session_store.get_drill_progress(phoneme_group)
        
        # Create result display
        result_html = create_drill_result_display(phoneme_group, risk_score, improvement)
        progress_html = create_drill_progress_display(phoneme_group, progress_data)
        
        logger.info(f"Drill analysis completed: {risk_score:.1f}% risk score")
        
        return result_html, progress_html
        
    except Exception as e:
        logger.error(f"Drill analysis failed: {e}")
        return f"❌ Drill analysis failed: {str(e)}", ""


def create_analysis_metrics(analysis_results: Dict[str, Any]) -> str:
    """Create metrics display from analysis results."""
    metrics = []
    
    # Clarity
    clarity_analysis = analysis_results.get("clarity_analysis", {})
    clarity_score = clarity_analysis.get("clarity_score", 0)
    clarity_assessment = clarity_analysis.get("assessment", "unknown").replace("_", " ").title()
    metrics.append(create_metric_card("Speech Clarity", f"{clarity_score}/100", clarity_assessment))
    
    # Pacing
    pacing_analysis = analysis_results.get("pacing_analysis", {})
    wpm = pacing_analysis.get("overall_wpm", 0)
    target_range = pacing_analysis.get("target_range", [140, 160])
    pacing_status = "Optimal" if target_range[0] <= wpm <= target_range[1] else "Needs Adjustment"
    metrics.append(create_metric_card("Speaking Pace", f"{wpm} WPM", f"Target: {target_range[0]}-{target_range[1]} WPM"))
    
    # Fillers
    filler_analysis = analysis_results.get("filler_analysis", {})
    filler_count = filler_analysis.get("filler_count", 0)
    fillers_per_min = filler_analysis.get("fillers_per_minute", 0)
    filler_assessment = filler_analysis.get("assessment", "unknown").replace("_", " ").title()
    metrics.append(create_metric_card("Filler Words", f"{filler_count} total", f"{fillers_per_min:.1f} per minute - {filler_assessment}"))
    
    # Prosody
    prosody_analysis = analysis_results.get("prosody_analysis", {})
    uptalk_count = prosody_analysis.get("uptalk_analysis", {}).get("uptalk_count", 0)
    stability_score = prosody_analysis.get("stability_metrics", {}).get("overall_stability_score", 0)
    metrics.append(create_metric_card("Prosody & Intonation", f"{stability_score:.0f}/100", f"Uptalk instances: {uptalk_count}"))
    
    return f"""
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; margin: 20px 0;">
        {''.join(metrics)}
    </div>
    """


def create_detailed_analysis(analysis_results: Dict[str, Any]) -> str:
    """Create detailed analysis display."""
    html_parts = []
    
    # Phoneme analysis
    phoneme_analysis = analysis_results.get("phoneme_analysis", {})
    if phoneme_analysis:
        html_parts.append(create_phoneme_risk_display(phoneme_analysis.get("phoneme_risks", {})))
    
    # Pause analysis
    pause_analysis = analysis_results.get("pause_analysis", {})
    if pause_analysis:
        metrics = pause_analysis.get("pause_metrics", {})
        pause_html = f"""
        <div>
            <h3 style="color: #1f2937; margin-bottom: 15px;">⏸️ Pause Analysis</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px;">
                <div style="text-align: center; padding: 10px; background: #f0f9ff; border-radius: 6px;">
                    <div style="font-weight: bold; color: #0891b2;">Total Pauses</div>
                    <div style="font-size: 1.5em; color: #0f172a;">{metrics.get('total_pauses', 0)}</div>
                </div>
                <div style="text-align: center; padding: 10px; background: #fef2f2; border-radius: 6px;">
                    <div style="font-weight: bold; color: #dc2626;">Hesitation</div>
                    <div style="font-size: 1.5em; color: #0f172a;">{metrics.get('hesitation_pauses', 0)}</div>
                </div>
                <div style="text-align: center; padding: 10px; background: #f0fdf4; border-radius: 6px;">
                    <div style="font-weight: bold; color: #059669;">Rhetorical</div>
                    <div style="font-size: 1.5em; color: #0f172a;">{metrics.get('rhetorical_pauses', 0)}</div>
                </div>
            </div>
        </div>
        """
        html_parts.append(pause_html)
    
    return f"""
    <div style="space-y: 20px;">
        {'<div style="margin-bottom: 30px;"></div>'.join(html_parts)}
    </div>
    """


def create_drill_result_display(phoneme_group: str, risk_score: float, improvement: Optional[float]) -> str:
    """Create drill result display."""
    # Risk level
    if risk_score <= 20:
        level = "Excellent"
        color = "#059669"
        icon = "🎉"
    elif risk_score <= 40:
        level = "Good"
        color = "#0891b2"
        icon = "👍"
    elif risk_score <= 60:
        level = "Fair"
        color = "#d97706"
        icon = "⚠️"
    else:
        level = "Needs Work"
        color = "#dc2626"
        icon = "📚"
    
    # Improvement text
    improvement_html = ""
    if improvement is not None:
        if improvement > 0:
            improvement_html = f"""
            <div style="margin-top: 15px; padding: 10px; background: #f0fdf4; border-radius: 6px; text-align: center;">
                <span style="color: #059669; font-weight: bold;">🎯 Improvement: {improvement:.1f}% better!</span>
            </div>
            """
        elif improvement < 0:
            improvement_html = f"""
            <div style="margin-top: 15px; padding: 10px; background: #fef2f2; border-radius: 6px; text-align: center;">
                <span style="color: #dc2626; font-weight: bold;">📈 Risk increased by {abs(improvement):.1f}%</span>
            </div>
            """
    
    return f"""
    <div style="padding: 20px; border-radius: 10px; background: {color}11; border: 2px solid {color};">
        <div style="text-align: center; margin-bottom: 15px;">
            <span style="font-size: 3em;">{icon}</span>
        </div>
        <h3 style="text-align: center; color: {color}; margin-bottom: 10px;">{phoneme_group} Sounds Analysis</h3>
        <div style="text-align: center; margin-bottom: 15px;">
            <div style="font-size: 2.5em; font-weight: bold; color: {color};">{risk_score:.1f}%</div>
            <div style="font-size: 1.2em; color: {color};">Risk Score - {level}</div>
        </div>
        {improvement_html}
    </div>
    """


def calculate_overall_score(analysis_results: Dict[str, Any]) -> int:
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


def load_demo_session() -> Tuple[str, str, str, str, str, str]:
    """Load demo session for demonstration."""
    try:
        # Create mock analysis results for demo
        demo_results = {
            "clarity_analysis": {"clarity_score": 87, "assessment": "good"},
            "pacing_analysis": {"overall_wpm": 155, "assessment": "optimal", "target_range": [140, 160]},
            "filler_analysis": {"filler_count": 3, "fillers_per_minute": 2.1, "assessment": "good"},
            "prosody_analysis": {
                "stability_metrics": {"overall_stability_score": 82},
                "uptalk_analysis": {"uptalk_count": 1},
                "assessment": "good"
            },
            "phoneme_analysis": {
                "phoneme_risks": {
                    "TH": {"risk_score": 25, "word_count": 4},
                    "RL": {"risk_score": 45, "word_count": 6},
                    "WV": {"risk_score": 15, "word_count": 3}
                }
            },
            "pause_analysis": {
                "pause_metrics": {"total_pauses": 5, "hesitation_pauses": 1, "rhetorical_pauses": 4}
            },
            "metadata": {"audio_duration": 18.5, "word_count": 47}
        }
        
        tips = [
            "🎯 Great overall delivery! Your pace of 155 WPM is in the optimal range.",
            "💡 Nice control of filler words with only 3 instances. Keep it up!",
            "📝 Consider working on R/L sound distinction - some words showed moderate risk.",
            "🎵 Good prosodic control with stable pitch and energy patterns.",
            "⏸️ Excellent use of rhetorical pauses to emphasize key points."
        ]
        
        overall_score = 85
        overall_display = create_score_display(overall_score, "Demo Score")
        metrics_html = create_analysis_metrics(demo_results)
        tips_html = create_tips_display(tips)
        detailed_html = create_detailed_analysis(demo_results)
        timeline_html = "<p style='color: #6b7280;'>Timeline events would appear here in a real analysis.</p>"
        
        # Demo transcript
        demo_transcript = "Welcome to Echo speech analysis. This demo shows how your transcript will appear after analysis. The system captures every word you speak and provides detailed insights about your speaking patterns, pace, and clarity."
        transcript_html = create_transcript_display(demo_transcript, 30, 18.5)
        
        return overall_display, metrics_html, tips_html, detailed_html, timeline_html, transcript_html
        
    except Exception as e:
        logger.error(f"Demo session failed: {e}")
        return "❌ Demo failed to load", "", "", "", "", ""


def get_session_list() -> List[List[str]]:
    """Get formatted session list for display."""
    try:
        sessions = session_store.get_recent_sessions(limit=20)
        
        session_data = []
        for session in sessions:
            summary = session.get("summary", {})
            
            # Format timestamp
            from datetime import datetime
            try:
                timestamp = datetime.fromisoformat(session["timestamp"])
                formatted_time = timestamp.strftime("%m/%d %H:%M")
            except:
                formatted_time = "Unknown"
            
            session_data.append([
                session.get("session_name", "Session"),
                formatted_time,
                str(summary.get("overall_score", 0)),
                str(summary.get("clarity_score", 0)),
                f"{summary.get('wpm', 0)} WPM",
                str(summary.get("filler_count", 0)),
                session["session_id"]
            ])
        
        return session_data
        
    except Exception as e:
        logger.error(f"Failed to get session list: {e}")
        return []


def generate_session_report(session_id: str, format_type: str = "pdf") -> str:
    """Generate report for a session."""
    try:
        if not session_id:
            return "❌ No session selected"
        
        session = session_store.get_session(session_id)
        if not session:
            return "❌ Session not found"
        
        report_result = report_generator.generate_report(
            session["analysis_results"],
            session["audio_filename"],
            format_type
        )
        
        if report_result["success"]:
            return f"✅ Report generated: {report_result['report_path']}"
        else:
            return f"❌ Report generation failed: {report_result.get('error', 'Unknown error')}"
    
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        return f"❌ Report generation failed: {str(e)}"


def create_echo_interface():
    """Create the main Echo Gradio interface."""
    
    with gr.Blocks(
        title="Echo - Speech Analysis",
        theme=gr.themes.Soft(),
        css=CUSTOM_CSS
    ) as demo:
        
        gr.Markdown("""
        # 🎤 Echo - Speech Analysis
        **Instant accent & clarity feedback for your speech**
        
        Get real-time analysis of your speaking pace, clarity, pronunciation, and fluency patterns.
        Perfect for presentations, interviews, and daily communication improvement.
        """)
        
        # Main tabs
        with gr.Tabs():
            
            # Quick Check Tab
            with gr.TabItem("🚀 Quick Check"):
                gr.Markdown("### Upload or record audio for comprehensive speech analysis")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        # Audio input
                        audio_input = gr.Audio(
                            label="📹 Record or Upload Audio (max 90 seconds)",
                            type="filepath",
                            elem_classes=["audio-input"]
                        )
                        
                        with gr.Row():
                            speaking_mode = gr.Dropdown(
                                choices=["Presentation", "Interview", "Casual"],
                                value="Presentation",
                                label="Speaking Context"
                            )
                            
                        with gr.Row():
                            session_name = gr.Textbox(
                                placeholder="Optional: Name this session",
                                label="Session Name"
                            )
                        
                        with gr.Row():
                            check_quality_btn = gr.Button(
                                "🔍 Check Audio Quality",
                                variant="secondary",
                                elem_classes=["secondary-button"]
                            )
                            analyze_btn = gr.Button(
                                "🎯 Analyze Speech",
                                variant="primary",
                                elem_classes=["primary-button"]
                            )
                            demo_btn = gr.Button(
                                "🎪 Try Demo",
                                variant="secondary",
                                elem_classes=["secondary-button"]
                            )
                        
                        # Audio quality check result
                        quality_status = gr.Markdown(value="", visible=True)
                    
                    with gr.Column(scale=2):
                        # Results display
                        with gr.Row():
                            overall_score_display = gr.HTML(value="", label="Overall Score")
                        
                        with gr.Row():
                            metrics_display = gr.HTML(value="", label="Key Metrics")
                        
                        with gr.Row():
                            tips_display = gr.HTML(value="", label="Recommendations")
                        
                        with gr.Row():
                            transcript_display = gr.HTML(value="", label="Speech Transcript")
            
            # Detailed Analysis Tab
            with gr.TabItem("📊 Detailed Analysis"):
                gr.Markdown("### In-depth breakdown of your speech patterns")
                
                with gr.Row():
                    with gr.Column():
                        detailed_analysis = gr.HTML(value="", label="Detailed Analysis")
                    
                    with gr.Column():
                        timeline_display = gr.HTML(value="", label="Speech Timeline")
            
            # Practice Mode Tab
            with gr.TabItem("🎯 Practice Mode"):
                gr.Markdown("### Target specific sounds for improvement")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        phoneme_group = gr.Dropdown(
                            choices=["TH", "RL", "WV", "ZH", "NG"],
                            value="TH",
                            label="Sound Group to Practice"
                        )
                        
                        practice_text_display = gr.Markdown(
                            value=get_phoneme_practice_text("TH"),
                            label="Practice Text"
                        )
                        
                        drill_audio_input = gr.Audio(
                            label="📹 Record Your Practice (5-10 seconds)",
                            type="filepath"
                        )
                        
                        drill_analyze_btn = gr.Button(
                            "🎯 Analyze Practice",
                            variant="primary",
                            elem_classes=["primary-button"]
                        )
                    
                    with gr.Column(scale=1):
                        drill_result_display = gr.HTML(value="", label="Practice Result")
                        drill_progress_display = gr.HTML(value="", label="Progress Tracking")
            
            # Reports Tab
            with gr.TabItem("📋 Reports"):
                gr.Markdown("### View and download your session reports")
                
                with gr.Row():
                    refresh_sessions_btn = gr.Button("🔄 Refresh Sessions", variant="secondary")
                
                with gr.Row():
                    sessions_table = gr.Dataframe(
                        headers=["Session Name", "Date", "Score", "Clarity", "Pace", "Fillers", "ID"],
                        value=get_session_list(),
                        interactive=True,
                        label="Recent Sessions"
                    )
                
                with gr.Row():
                    selected_session = gr.Textbox(
                        placeholder="Select a session from the table above",
                        label="Selected Session ID"
                    )
                    
                    generate_report_btn = gr.Button(
                        "📄 Generate PDF Report",
                        variant="primary"
                    )
                
                report_status = gr.Markdown(value="", label="Report Status")
            
            # Settings Tab
            with gr.TabItem("⚙️ Settings"):
                gr.Markdown("### Configure your analysis preferences")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Analysis Settings")
                        
                        snr_threshold = gr.Slider(
                            minimum=10,
                            maximum=30,
                            value=20,
                            step=1,
                            label="Audio Quality Threshold (SNR dB)"
                        )
                        
                        save_sessions_toggle = gr.Checkbox(
                            value=True,
                            label="Save analysis sessions automatically"
                        )
                        
                        high_contrast_toggle = gr.Checkbox(
                            value=False,
                            label="High contrast mode"
                        )
                    
                    with gr.Column():
                        gr.Markdown("#### System Information")
                        
                        system_info = gr.Markdown(f"""
                        **Echo Version:** 1.0.0  
                        **Available ASR Engines:** {', '.join(asr_manager.get_available_engines())}  
                        **Max Audio Duration:** {settings.audio.get('max_duration_seconds', 90)} seconds  
                        **Supported Formats:** {', '.join(settings.audio.get('supported_formats', []))}  
                        **Processing:** On-device (offline-first)  
                        """)
        
        # Event handlers
        check_quality_btn.click(
            fn=check_audio_quality,
            inputs=[audio_input],
            outputs=[quality_status],
        )
        
        analyze_btn.click(
            fn=analyze_speech_full,
            inputs=[audio_input, speaking_mode, session_name, save_sessions_toggle],
            outputs=[overall_score_display, metrics_display, tips_display, detailed_analysis, timeline_display, transcript_display],
        )
        
        demo_btn.click(
            fn=load_demo_session,
            outputs=[overall_score_display, metrics_display, tips_display, detailed_analysis, timeline_display, transcript_display],
        )
        
        # Update practice text when phoneme group changes
        phoneme_group.change(
            fn=get_phoneme_practice_text,
            inputs=[phoneme_group],
            outputs=[practice_text_display]
        )
        
        drill_analyze_btn.click(
            fn=analyze_drill,
            inputs=[drill_audio_input, phoneme_group],
            outputs=[drill_result_display, drill_progress_display],
        )
        
        refresh_sessions_btn.click(
            fn=get_session_list,
            outputs=[sessions_table],
        )
        
        generate_report_btn.click(
            fn=generate_session_report,
            inputs=[selected_session],
            outputs=[report_status],
        )
        
        # Auto-populate selected session when table is clicked
        sessions_table.select(
            fn=lambda evt: evt.index[0][6] if evt.index and len(evt.index[0]) > 6 else "",
            inputs=[sessions_table],
            outputs=[selected_session],
        )
    
    return demo


def main():
    """Main entry point for the Gradio app."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Echo Speech Analysis Web Interface")
    parser.add_argument("--host", default="127.0.0.1", help="Host to run the server on")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the server on")
    parser.add_argument("--demo", action="store_true", help="Run in demo mode")
    parser.add_argument("--share", action="store_true", help="Create a public link")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.demo:
        app_state["demo_mode"] = True
    
    logger.info("Starting Echo Web Interface")
    logger.info(f"Available ASR engines: {asr_manager.get_available_engines()}")
    
    # Create and launch the interface
    demo = create_echo_interface()
    
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
        quiet=not args.debug
    )


if __name__ == "__main__":
    main()