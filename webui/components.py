import gradio as gr
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import plotly.graph_objects as go
import json
import logging

from app.utils.plotting import plot_generator
from app.utils.session_store import session_store
from app.settings import settings

logger = logging.getLogger(__name__)


def create_score_display(score: int, label: str = "Score") -> str:
    """Create HTML for score display with color coding."""
    if score >= 90:
        color = "#059669"  # Green
        level = "Excellent"
    elif score >= 80:
        color = "#0891b2"  # Blue
        level = "Good"
    elif score >= 70:
        color = "#d97706"  # Amber
        level = "Fair"
    elif score >= 60:
        color = "#dc2626"  # Red
        level = "Poor"
    else:
        color = "#6b7280"  # Gray
        level = "Very Poor"
    
    return f"""
    <div style="text-align: center; padding: 20px; border-radius: 10px; background: linear-gradient(135deg, {color}22, {color}11);">
        <div style="font-size: 2.5em; font-weight: bold; color: {color};">{score}</div>
        <div style="font-size: 1.2em; color: {color}; margin-top: 5px;">{level}</div>
        <div style="font-size: 0.9em; color: #666; margin-top: 5px;">{label}</div>
    </div>
    """


def create_metric_card(title: str, value: str, description: str = "", trend: Optional[str] = None) -> str:
    """Create HTML for metric display card."""
    trend_html = ""
    if trend:
        trend_color = "#059669" if trend.startswith("↑") else "#dc2626" if trend.startswith("↓") else "#6b7280"
        trend_html = f'<div style="font-size: 0.9em; color: {trend_color}; margin-top: 5px;">{trend}</div>'
    
    return f"""
    <div style="padding: 15px; border-radius: 8px; border: 1px solid #e5e7eb; background: white;">
        <div style="font-weight: bold; color: #374151; margin-bottom: 8px;">{title}</div>
        <div style="font-size: 1.8em; font-weight: bold; color: #1f2937; margin-bottom: 5px;">{value}</div>
        <div style="font-size: 0.85em; color: #6b7280;">{description}</div>
        {trend_html}
    </div>
    """


def create_tips_display(tips: List[str]) -> str:
    """Create HTML for tips display."""
    if not tips:
        return "<p style='color: #6b7280;'>No specific recommendations at this time.</p>"
    
    tips_html = ""
    for i, tip in enumerate(tips, 1):
        icon = "🎯" if i == 1 else "💡" if i <= 3 else "📝"
        tips_html += f"""
        <div style="display: flex; align-items: flex-start; margin-bottom: 12px; padding: 10px; 
                    background: #f8fafc; border-radius: 6px; border-left: 3px solid #2563eb;">
            <span style="margin-right: 10px; font-size: 1.2em;">{icon}</span>
            <span style="flex: 1; color: #374151;">{tip}</span>
        </div>
        """
    
    return f"""
    <div style="max-height: 400px; overflow-y: auto;">
        <h3 style="color: #1f2937; margin-bottom: 15px;">📚 Personalized Recommendations</h3>
        {tips_html}
    </div>
    """


def create_progress_bar(value: float, max_value: float = 100, color: str = "#2563eb") -> str:
    """Create HTML progress bar."""
    percentage = (value / max_value) * 100
    
    return f"""
    <div style="width: 100%; background-color: #e5e7eb; border-radius: 4px; height: 20px;">
        <div style="width: {percentage}%; background-color: {color}; height: 100%; 
                    border-radius: 4px; transition: width 0.3s ease;"></div>
    </div>
    <div style="text-align: center; font-size: 0.9em; color: #6b7280; margin-top: 5px;">
        {value:.1f} / {max_value}
    </div>
    """


def create_phoneme_risk_display(phoneme_risks: Dict[str, Any]) -> str:
    """Create HTML for phoneme risk visualization."""
    if not phoneme_risks:
        return "<p style='color: #6b7280;'>No pronunciation analysis available.</p>"
    
    risk_html = ""
    for group, data in phoneme_risks.items():
        risk_score = data.get("risk_score", 0)
        word_count = data.get("word_count", 0)
        
        # Color based on risk level
        if risk_score > 60:
            color = "#dc2626"  # Red
            level = "High Risk"
        elif risk_score > 30:
            color = "#d97706"  # Amber
            level = "Moderate Risk"
        else:
            color = "#059669"  # Green
            level = "Low Risk"
        
        risk_html += f"""
        <div style="margin-bottom: 15px; padding: 12px; border-radius: 8px; 
                    background: {color}11; border-left: 4px solid {color};">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="font-weight: bold; color: #1f2937;">{group} Sounds</span>
                    <span style="margin-left: 10px; font-size: 0.9em; color: #6b7280;">
                        ({word_count} words found)
                    </span>
                </div>
                <div>
                    <span style="font-weight: bold; color: {color};">{risk_score:.0f}%</span>
                    <span style="margin-left: 8px; font-size: 0.8em; color: {color};">{level}</span>
                </div>
            </div>
            <div style="margin-top: 8px;">
                {create_progress_bar(risk_score, 100, color)}
            </div>
        </div>
        """
    
    return f"""
    <div>
        <h3 style="color: #1f2937; margin-bottom: 15px;">🔤 Pronunciation Analysis</h3>
        {risk_html}
    </div>
    """


def create_timeline_events(analysis_results: Dict[str, Any]) -> str:
    """Create timeline display for speech events."""
    events_html = ""
    
    # Collect events from different analyses
    events = []
    
    # Fillers
    if "filler_analysis" in analysis_results:
        fillers = analysis_results["filler_analysis"].get("filler_instances", [])
        for filler in fillers[:10]:  # Limit to first 10
            events.append({
                "time": filler.get("start", 0),
                "type": "Filler",
                "text": filler.get("text", ""),
                "color": "#d97706"
            })
    
    # Pauses
    if "pause_analysis" in analysis_results:
        pauses = analysis_results["pause_analysis"].get("timeline_data", [])
        for pause in pauses[:10]:  # Limit to first 10
            pause_type = pause.get("type", "").replace("_", " ").title()
            color = "#dc2626" if pause_type == "Hesitation" else "#059669"
            events.append({
                "time": pause.get("start", 0),
                "type": pause_type + " Pause",
                "text": f"{pause.get('duration', 0):.1f}s",
                "color": color
            })
    
    # Uptalk
    if "prosody_analysis" in analysis_results:
        uptalk_instances = analysis_results["prosody_analysis"].get("uptalk_analysis", {}).get("uptalk_instances", [])
        for uptalk in uptalk_instances[:5]:  # Limit to first 5
            events.append({
                "time": uptalk.get("start", 0),
                "type": "Uptalk",
                "text": uptalk.get("text", "statement")[:20] + "...",
                "color": "#0891b2"
            })
    
    # Sort by time
    events.sort(key=lambda x: x["time"])
    
    if not events:
        return "<p style='color: #6b7280;'>No significant events detected.</p>"
    
    for event in events:
        events_html += f"""
        <div style="display: flex; align-items: center; margin-bottom: 8px; padding: 8px; 
                    background: white; border-radius: 6px; border-left: 3px solid {event['color']};">
            <div style="min-width: 60px; font-weight: bold; color: #374151;">
                {event['time']:.1f}s
            </div>
            <div style="min-width: 100px; font-size: 0.9em; color: {event['color']}; font-weight: bold;">
                {event['type']}
            </div>
            <div style="flex: 1; color: #6b7280;">
                {event['text']}
            </div>
        </div>
        """
    
    return f"""
    <div style="max-height: 300px; overflow-y: auto; background: #f8fafc; padding: 10px; border-radius: 8px;">
        <h4 style="color: #1f2937; margin-bottom: 10px;">📋 Speech Events Timeline</h4>
        {events_html}
    </div>
    """


def format_session_summary(session: Dict[str, Any]) -> str:
    """Format session summary for display."""
    summary = session.get("summary", {})
    
    # Format timestamp
    from datetime import datetime
    try:
        timestamp = datetime.fromisoformat(session["timestamp"])
        formatted_time = timestamp.strftime("%Y-%m-%d %H:%M")
    except:
        formatted_time = session.get("timestamp", "Unknown")
    
    overall_score = summary.get("overall_score", 0)
    clarity_score = summary.get("clarity_score", 0)
    wpm = summary.get("wpm", 0)
    filler_count = summary.get("filler_count", 0)
    
    score_color = "#059669" if overall_score >= 80 else "#d97706" if overall_score >= 60 else "#dc2626"
    
    return f"""
    <div style="padding: 15px; margin: 10px 0; border-radius: 8px; border: 1px solid #e5e7eb; background: white;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
            <h3 style="margin: 0; color: #1f2937;">{session.get('session_name', 'Session')}</h3>
            <div style="font-size: 1.5em; font-weight: bold; color: {score_color};">{overall_score}/100</div>
        </div>
        <div style="font-size: 0.9em; color: #6b7280; margin-bottom: 10px;">
            📁 {session.get('audio_filename', 'Unknown file')} • 📅 {formatted_time}
        </div>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 10px;">
            <div style="text-align: center;">
                <div style="font-weight: bold; color: #374151;">Clarity</div>
                <div style="color: #0891b2;">{clarity_score}/100</div>
            </div>
            <div style="text-align: center;">
                <div style="font-weight: bold; color: #374151;">Pace</div>
                <div style="color: #0891b2;">{wpm} WPM</div>
            </div>
            <div style="text-align: center;">
                <div style="font-weight: bold; color: #374151;">Fillers</div>
                <div style="color: #0891b2;">{filler_count}</div>
            </div>
        </div>
    </div>
    """


def create_drill_progress_display(phoneme_group: str, progress_data: Dict[str, Any]) -> str:
    """Create drill progress visualization."""
    if not progress_data:
        return f"<p style='color: #6b7280;'>No practice history for {phoneme_group} sounds yet.</p>"
    
    total_attempts = progress_data.get("total_attempts", 0)
    latest_score = progress_data.get("latest_score", 0)
    best_score = progress_data.get("best_score", 0)
    improvement = progress_data.get("improvement", 0)
    trend = progress_data.get("trend", [])
    
    # Progress indicators
    improvement_text = ""
    if improvement > 0:
        improvement_text = f"↓ {improvement:.1f}% improvement"
        improvement_color = "#059669"
    elif improvement < 0:
        improvement_text = f"↑ {abs(improvement):.1f}% increase in risk"
        improvement_color = "#dc2626"
    else:
        improvement_text = "No change from last attempt"
        improvement_color = "#6b7280"
    
    # Trend sparkline (simplified)
    trend_html = ""
    if len(trend) > 1:
        trend_points = ", ".join([str(t) for t in trend])
        trend_html = f"""
        <div style="margin-top: 10px;">
            <div style="font-size: 0.9em; color: #6b7280;">Practice Trend:</div>
            <div style="font-family: monospace; font-size: 0.8em; color: #374151;">{trend_points}</div>
        </div>
        """
    
    return f"""
    <div style="padding: 15px; border-radius: 8px; background: white; border: 1px solid #e5e7eb;">
        <h3 style="color: #1f2937; margin-bottom: 15px;">📈 {phoneme_group} Progress</h3>
        
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; margin-bottom: 15px;">
            <div style="text-align: center;">
                <div style="font-size: 1.8em; font-weight: bold; color: #2563eb;">{total_attempts}</div>
                <div style="font-size: 0.9em; color: #6b7280;">Total Attempts</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 1.8em; font-weight: bold; color: #059669;">{best_score:.1f}%</div>
                <div style="font-size: 0.9em; color: #6b7280;">Best Score (Low Risk)</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 1.8em; font-weight: bold; color: #0891b2;">{latest_score:.1f}%</div>
                <div style="font-size: 0.9em; color: #6b7280;">Latest Score</div>
            </div>
        </div>
        
        <div style="padding: 10px; background: #f8fafc; border-radius: 6px; text-align: center;">
            <span style="color: {improvement_color}; font-weight: bold;">{improvement_text}</span>
        </div>
        
        {trend_html}
    </div>
    """


def create_snr_warning_modal(snr_db: float, message: str) -> str:
    """Create SNR warning modal content."""
    if snr_db >= 20:
        return ""  # No warning needed
    
    icon = "🚨" if snr_db < 10 else "⚠️"
    color = "#dc2626" if snr_db < 10 else "#d97706"
    
    return f"""
    <div style="padding: 20px; border-radius: 10px; background: {color}11; border: 2px solid {color};">
        <div style="text-align: center; margin-bottom: 15px;">
            <span style="font-size: 3em;">{icon}</span>
        </div>
        <h3 style="text-align: center; color: {color}; margin-bottom: 15px;">Audio Quality Warning</h3>
        <p style="text-align: center; color: #374151; margin-bottom: 20px;">{message}</p>
        <div style="text-align: center; font-size: 0.9em; color: #6b7280;">
            Signal-to-Noise Ratio: {snr_db:.1f} dB (Minimum recommended: 20 dB)
        </div>
    </div>
    """


def create_transcript_display(transcript: str, word_count: int, duration: float) -> str:
    """Create HTML for transcript display with metadata."""
    if not transcript or not transcript.strip():
        return """
        <div style="padding: 20px; text-align: center; color: #6b7280; background: #f9fafb; border-radius: 8px; border: 1px solid #e5e7eb;">
            <span style="font-size: 1.2em;">📝</span>
            <p>No transcript available</p>
        </div>
        """
    
    # Calculate reading stats
    wpm_estimate = round((word_count / duration) * 60) if duration > 0 else 0
    char_count = len(transcript)
    
    # Clean and format transcript
    formatted_transcript = transcript.strip()
    if len(formatted_transcript) > 500:
        # Show preview with expand option for long transcripts
        preview = formatted_transcript[:500] + "..."
        transcript_html = f"""
        <div id="transcript-preview" style="display: block;">
            <p style="line-height: 1.6; color: #374151; margin-bottom: 10px; font-size: 1.05em;">{preview}</p>
            <button onclick="document.getElementById('transcript-preview').style.display='none'; document.getElementById('transcript-full').style.display='block';" 
                    style="background: #2563eb; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; font-size: 0.9em;">
                Show Full Transcript
            </button>
        </div>
        <div id="transcript-full" style="display: none;">
            <p style="line-height: 1.6; color: #374151; margin-bottom: 10px; font-size: 1.05em;">{formatted_transcript}</p>
            <button onclick="document.getElementById('transcript-preview').style.display='block'; document.getElementById('transcript-full').style.display='none';" 
                    style="background: #6b7280; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; font-size: 0.9em;">
                Show Less
            </button>
        </div>
        """
    else:
        transcript_html = f'<p style="line-height: 1.6; color: #374151; margin-bottom: 10px; font-size: 1.05em;">{formatted_transcript}</p>'
    
    return f"""
    <div style="background: white; border-radius: 10px; border: 1px solid #e5e7eb; overflow: hidden;">
        <!-- Header -->
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 15px; color: white;">
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <h3 style="margin: 0; display: flex; align-items: center;">
                    <span style="margin-right: 10px; font-size: 1.3em;">📝</span>
                    Speech Transcript
                </h3>
                <div style="display: flex; gap: 15px; font-size: 0.9em;">
                    <span>📊 {word_count} words</span>
                    <span>⏱️ {duration:.1f}s</span>
                    <span>🗣️ {wpm_estimate} WPM</span>
                </div>
            </div>
        </div>
        
        <!-- Content -->
        <div style="padding: 20px;">
            <!-- Stats Bar -->
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 15px; margin-bottom: 20px; padding: 15px; background: #f8fafc; border-radius: 8px;">
                <div style="text-align: center;">
                    <div style="font-weight: bold; color: #374151; margin-bottom: 4px;">Words</div>
                    <div style="font-size: 1.4em; color: #2563eb; font-weight: bold;">{word_count}</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-weight: bold; color: #374151; margin-bottom: 4px;">Characters</div>
                    <div style="font-size: 1.4em; color: #059669; font-weight: bold;">{char_count}</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-weight: bold; color: #374151; margin-bottom: 4px;">Duration</div>
                    <div style="font-size: 1.4em; color: #d97706; font-weight: bold;">{duration:.1f}s</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-weight: bold; color: #374151; margin-bottom: 4px;">Pace</div>
                    <div style="font-size: 1.4em; color: #dc2626; font-weight: bold;">{wpm_estimate} WPM</div>
                </div>
            </div>
            
            <!-- Transcript Text -->
            <div style="background: #fafbfc; padding: 20px; border-radius: 8px; border: 1px solid #e5e7eb; min-height: 120px;">
                {transcript_html}
            </div>
            
            <!-- Actions -->
            <div style="margin-top: 15px; display: flex; gap: 10px; justify-content: flex-end;">
                <button onclick="navigator.clipboard.writeText(`{formatted_transcript.replace('`', '\\`')}`); this.innerHTML='✅ Copied!'; setTimeout(() => this.innerHTML='📋 Copy Text', 2000);" 
                        style="background: #f3f4f6; color: #374151; border: 1px solid #d1d5db; padding: 8px 16px; border-radius: 6px; cursor: pointer; font-size: 0.9em;">
                    📋 Copy Text
                </button>
            </div>
        </div>
    </div>
    """


def get_phoneme_practice_text(phoneme_group: str) -> str:
    """Get practice text for specific phoneme group."""
    practice_texts = {
        "TH": "The thirty-three thieves thought they thrilled the throne throughout the thick thunderstorm.",
        "RL": "Really red roses are rarely real. The library regularly collects colorful literature.",
        "WV": "Very vivid videos were viewed by visitors. The waves were vast and violent in the valley.",
        "ZH": "Measure the pleasure of leisure and treasure. The vision of fusion brings precision to the decision.",
        "NG": "Running, jumping, thinking, working, singing brings meaning to living and learning."
    }
    
    return practice_texts.get(phoneme_group, f"Practice words containing {phoneme_group} sounds carefully and clearly.")


# CSS styles for the interface
CUSTOM_CSS = """
/* Main container styling */
.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto;
}

/* Tab styling */
.tab-nav {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border-radius: 8px 8px 0 0;
}

/* Button styling */
.primary-button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    border-radius: 8px !important;
    color: white !important;
    font-weight: bold !important;
}

.secondary-button {
    background: #f8fafc !important;
    border: 2px solid #e5e7eb !important;
    border-radius: 8px !important;
    color: #374151 !important;
}

/* Card styling */
.metric-card {
    background: white !important;
    border-radius: 10px !important;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1) !important;
    padding: 20px !important;
    margin: 10px 0 !important;
}

/* Input styling */
.audio-input {
    border: 2px dashed #d1d5db !important;
    border-radius: 10px !important;
    background: #f9fafb !important;
}

/* Progress bar */
.progress-bar {
    background: linear-gradient(90deg, #10b981 0%, #059669 100%) !important;
    border-radius: 4px !important;
    height: 8px !important;
}

/* Transcript styling */
.transcript-container {
    background: white !important;
    border-radius: 10px !important;
    border: 1px solid #e5e7eb !important;
    margin: 15px 0 !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08) !important;
}

.transcript-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    padding: 15px !important;
    border-radius: 10px 10px 0 0 !important;
}

.transcript-text {
    line-height: 1.6 !important;
    font-size: 1.05em !important;
    color: #374151 !important;
    padding: 20px !important;
    background: #fafbfc !important;
    border-radius: 8px !important;
    border: 1px solid #e5e7eb !important;
    min-height: 120px !important;
}

.transcript-stats {
    display: grid !important;
    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)) !important;
    gap: 15px !important;
    padding: 15px !important;
    background: #f8fafc !important;
    border-radius: 8px !important;
    margin-bottom: 20px !important;
}

.transcript-stat-item {
    text-align: center !important;
}

.transcript-stat-label {
    font-weight: bold !important;
    color: #374151 !important;
    margin-bottom: 4px !important;
    font-size: 0.9em !important;
}

.transcript-stat-value {
    font-size: 1.4em !important;
    font-weight: bold !important;
}

/* Copy button styling */
.transcript-copy-btn {
    background: #f3f4f6 !important;
    color: #374151 !important;
    border: 1px solid #d1d5db !important;
    padding: 8px 16px !important;
    border-radius: 6px !important;
    cursor: pointer !important;
    font-size: 0.9em !important;
    transition: all 0.2s ease !important;
}

.transcript-copy-btn:hover {
    background: #e5e7eb !important;
    color: #111827 !important;
}

/* Expand/collapse buttons */
.transcript-toggle-btn {
    background: #2563eb !important;
    color: white !important;
    border: none !important;
    padding: 8px 16px !important;
    border-radius: 4px !important;
    cursor: pointer !important;
    font-size: 0.9em !important;
    transition: background 0.2s ease !important;
}

.transcript-toggle-btn:hover {
    background: #1d4ed8 !important;
}

.transcript-toggle-btn.secondary {
    background: #6b7280 !important;
}

.transcript-toggle-btn.secondary:hover {
    background: #4b5563 !important;
}
"""