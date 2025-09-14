import logging
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import base64
from io import BytesIO

try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.colors import HexColor, black, darkgray
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.platypus import PageBreak, Image
    from reportlab.lib.units import inch
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.offline import plot
    import plotly.io as pio
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from ..settings import settings
from .tips import tips_generator

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate comprehensive reports from speech analysis results."""
    
    def __init__(self):
        self.reports_dir = Path(settings.storage.reports_dir)
        self.reports_dir.mkdir(exist_ok=True)
        
        # Colors for consistent theming
        self.colors = {
            "primary": "#2563eb",      # Blue
            "secondary": "#64748b",    # Slate
            "success": "#059669",      # Green
            "warning": "#d97706",      # Amber
            "error": "#dc2626",        # Red
            "background": "#f8fafc",   # Light gray
            "text": "#1e293b"          # Dark slate
        }
    
    def generate_report(self, 
                       analysis_results: Dict[str, Any],
                       audio_filename: str = "audio.wav",
                       report_format: str = "pdf") -> Dict[str, Any]:
        """Generate comprehensive analysis report."""
        try:
            timestamp = datetime.now()
            
            # Prepare report data
            report_data = self._prepare_report_data(analysis_results, audio_filename, timestamp)
            
            # Generate visualizations
            charts = self._generate_charts(analysis_results) if PLOTLY_AVAILABLE else {}
            
            if report_format.lower() == "pdf" and REPORTLAB_AVAILABLE:
                report_path = self._generate_pdf_report(report_data, charts, timestamp)
                report_type = "pdf"
            else:
                report_path = self._generate_json_report(report_data, timestamp)
                report_type = "json"
            
            return {
                "success": True,
                "report_path": str(report_path),
                "report_type": report_type,
                "timestamp": timestamp.isoformat(),
                "summary": report_data["summary"]
            }
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _prepare_report_data(self, 
                           analysis_results: Dict[str, Any],
                           audio_filename: str,
                           timestamp: datetime) -> Dict[str, Any]:
        """Prepare structured data for report generation."""
        
        # Extract key metrics
        summary = {
            "overall_score": self._calculate_overall_score(analysis_results),
            "audio_filename": audio_filename,
            "analysis_timestamp": timestamp.isoformat(),
            "duration": analysis_results.get("audio_duration", 0),
            "word_count": analysis_results.get("pacing_analysis", {}).get("word_count", 0)
        }
        
        # Process each analysis category
        sections = {}
        
        if "snr_analysis" in analysis_results:
            sections["audio_quality"] = self._process_snr_section(analysis_results["snr_analysis"])
        
        if "pacing_analysis" in analysis_results:
            sections["pacing"] = self._process_pacing_section(analysis_results["pacing_analysis"])
        
        if "clarity_analysis" in analysis_results:
            sections["clarity"] = self._process_clarity_section(analysis_results["clarity_analysis"])
        
        if "filler_analysis" in analysis_results:
            sections["fillers"] = self._process_filler_section(analysis_results["filler_analysis"])
        
        if "prosody_analysis" in analysis_results:
            sections["prosody"] = self._process_prosody_section(analysis_results["prosody_analysis"])
        
        if "pause_analysis" in analysis_results:
            sections["pauses"] = self._process_pause_section(analysis_results["pause_analysis"])
        
        if "phoneme_analysis" in analysis_results:
            sections["pronunciation"] = self._process_phoneme_section(analysis_results["phoneme_analysis"])
        
        # Generate personalized tips
        tips = tips_generator.generate_personalized_tips(analysis_results, max_tips=7)
        
        return {
            "summary": summary,
            "sections": sections,
            "tips": tips,
            "metadata": {
                "generated_by": "Echo v1.0.0",
                "analysis_type": "comprehensive",
                "on_device": True
            }
        }
    
    def _calculate_overall_score(self, analysis_results: Dict[str, Any]) -> int:
        """Calculate overall speech score (0-100)."""
        scores = []
        
        # Collect individual scores
        if "clarity_analysis" in analysis_results:
            scores.append(analysis_results["clarity_analysis"].get("clarity_score", 0))
        
        if "pacing_analysis" in analysis_results:
            wpm = analysis_results["pacing_analysis"].get("overall_wpm", 0)
            # Convert WPM to score (0-100)
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
            filler_score = max(0, 100 - fillers_per_min * 15)  # Penalty per filler/min
            scores.append(filler_score)
        
        if "prosody_analysis" in analysis_results:
            stability = analysis_results["prosody_analysis"].get("stability_metrics", {})
            prosody_score = stability.get("overall_stability_score", 50)
            scores.append(prosody_score)
        
        # Average the scores
        return round(sum(scores) / len(scores)) if scores else 0
    
    def _process_snr_section(self, snr_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Process SNR analysis for report."""
        return {
            "title": "Audio Quality",
            "score": 100 if snr_analysis.get("snr_ok", True) else 60,
            "metrics": {
                "SNR (dB)": f"{snr_analysis.get('snr_db', 0):.1f}",
                "Quality": "Good" if snr_analysis.get("snr_ok", True) else "Poor"
            },
            "description": "Signal-to-noise ratio indicates audio recording quality."
        }
    
    def _process_pacing_section(self, pacing_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Process pacing analysis for report."""
        wpm = pacing_analysis.get("overall_wpm", 0)
        assessment = pacing_analysis.get("assessment", "unknown")
        
        return {
            "title": "Speaking Pace",
            "score": self._assessment_to_score(assessment),
            "metrics": {
                "Words per Minute": f"{wpm}",
                "Target Range": f"{pacing_analysis.get('target_range', [140, 160])[0]}-{pacing_analysis.get('target_range', [140, 160])[1]}",
                "Assessment": assessment.replace("_", " ").title()
            },
            "description": "Your speaking rate and rhythm patterns."
        }
    
    def _process_clarity_section(self, clarity_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Process clarity analysis for report."""
        return {
            "title": "Speech Clarity",
            "score": clarity_analysis.get("clarity_score", 0),
            "metrics": {
                "Clarity Score": f"{clarity_analysis.get('clarity_score', 0)}/100",
                "Assessment": clarity_analysis.get("assessment", "unknown").replace("_", " ").title()
            },
            "description": "How clearly your words are pronounced and understood."
        }
    
    def _process_filler_section(self, filler_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Process filler analysis for report."""
        count = filler_analysis.get("filler_count", 0)
        per_minute = filler_analysis.get("fillers_per_minute", 0)
        assessment = filler_analysis.get("assessment", "unknown")
        
        return {
            "title": "Filler Words",
            "score": self._assessment_to_score(assessment),
            "metrics": {
                "Total Fillers": f"{count}",
                "Per Minute": f"{per_minute:.1f}",
                "Assessment": assessment.replace("_", " ").title()
            },
            "description": "Usage of 'um', 'uh', 'like' and other filler words."
        }
    
    def _process_prosody_section(self, prosody_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Process prosody analysis for report."""
        stability_metrics = prosody_analysis.get("stability_metrics", {})
        uptalk_analysis = prosody_analysis.get("uptalk_analysis", {})
        
        return {
            "title": "Intonation & Prosody",
            "score": stability_metrics.get("overall_stability_score", 0),
            "metrics": {
                "Stability Score": f"{stability_metrics.get('overall_stability_score', 0)}/100",
                "Uptalk Instances": f"{uptalk_analysis.get('uptalk_count', 0)}",
                "Assessment": prosody_analysis.get("assessment", "unknown").replace("_", " ").title()
            },
            "description": "Your pitch patterns, rhythm, and intonation control."
        }
    
    def _process_pause_section(self, pause_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Process pause analysis for report."""
        metrics = pause_analysis.get("pause_metrics", {})
        assessment = pause_analysis.get("assessment", "unknown")
        
        return {
            "title": "Pause Patterns",
            "score": self._assessment_to_score(assessment),
            "metrics": {
                "Total Pauses": f"{metrics.get('total_pauses', 0)}",
                "Hesitation Pauses": f"{metrics.get('hesitation_pauses', 0)}",
                "Rhetorical Pauses": f"{metrics.get('rhetorical_pauses', 0)}",
                "Assessment": assessment.replace("_", " ").title()
            },
            "description": "Analysis of pause timing and purpose in your speech."
        }
    
    def _process_phoneme_section(self, phoneme_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Process phoneme analysis for report."""
        overall_risk = phoneme_analysis.get("overall_risk", 0)
        assessment = phoneme_analysis.get("assessment", "unknown")
        
        # Find highest risk phoneme groups
        phoneme_risks = phoneme_analysis.get("phoneme_risks", {})
        top_risks = sorted(phoneme_risks.items(), 
                          key=lambda x: x[1].get("risk_score", 0), 
                          reverse=True)[:3]
        
        risk_summary = ", ".join([f"{group}: {data.get('risk_score', 0):.0f}%" 
                                 for group, data in top_risks]) if top_risks else "None"
        
        return {
            "title": "Pronunciation",
            "score": max(0, 100 - overall_risk),
            "metrics": {
                "Overall Risk": f"{overall_risk}/100",
                "Top Risk Areas": risk_summary,
                "Assessment": assessment.replace("_", " ").title()
            },
            "description": "Analysis of challenging sound groups and pronunciation patterns."
        }
    
    def _assessment_to_score(self, assessment: str) -> int:
        """Convert assessment string to numeric score."""
        score_map = {
            "excellent": 95,
            "good": 85,
            "fair": 70,
            "moderate": 65,
            "poor": 40,
            "very_poor": 20,
            "needs_improvement": 45,
            "unknown": 50
        }
        return score_map.get(assessment, 50)
    
    def _generate_charts(self, analysis_results: Dict[str, Any]) -> Dict[str, str]:
        """Generate visualization charts as base64 images."""
        charts = {}
        
        try:
            # Overview radar chart
            charts["radar"] = self._create_radar_chart(analysis_results)
            
            # Timeline chart (if applicable)
            if "filler_analysis" in analysis_results or "pause_analysis" in analysis_results:
                charts["timeline"] = self._create_timeline_chart(analysis_results)
            
            # Phoneme risk chart
            if "phoneme_analysis" in analysis_results:
                charts["phonemes"] = self._create_phoneme_chart(analysis_results["phoneme_analysis"])
                
        except Exception as e:
            logger.warning(f"Error generating charts: {e}")
        
        return charts
    
    def _create_radar_chart(self, analysis_results: Dict[str, Any]) -> str:
        """Create radar chart of all metrics."""
        categories = []
        scores = []
        
        # Collect scores for radar chart
        if "clarity_analysis" in analysis_results:
            categories.append("Clarity")
            scores.append(analysis_results["clarity_analysis"].get("clarity_score", 0))
        
        if "pacing_analysis" in analysis_results:
            categories.append("Pacing")
            wpm = analysis_results["pacing_analysis"].get("overall_wpm", 0)
            # Convert WPM to score
            target_range = [140, 160]
            if target_range[0] <= wpm <= target_range[1]:
                pacing_score = 100
            else:
                pacing_score = max(0, 100 - abs(wpm - 150) * 2)
            scores.append(pacing_score)
        
        if "filler_analysis" in analysis_results:
            categories.append("Fluency")
            fillers_per_min = analysis_results["filler_analysis"].get("fillers_per_minute", 0)
            fluency_score = max(0, 100 - fillers_per_min * 15)
            scores.append(fluency_score)
        
        if "prosody_analysis" in analysis_results:
            categories.append("Prosody")
            stability = analysis_results["prosody_analysis"].get("stability_metrics", {})
            prosody_score = stability.get("overall_stability_score", 50)
            scores.append(prosody_score)
        
        if not categories:
            return ""
        
        # Add first category again to close the radar chart
        categories.append(categories[0])
        scores.append(scores[0])
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=scores,
            theta=categories,
            fill='toself',
            name='Your Score',
            fillcolor=f'rgba{self._hex_to_rgba(self.colors["primary"], 0.3)}',
            line=dict(color=self.colors["primary"])
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100]),
            ),
            showlegend=False,
            width=400,
            height=400,
            margin=dict(t=20, b=20, l=20, r=20)
        )
        
        # Convert to base64
        img_bytes = fig.to_image(format="png", width=400, height=400)
        return base64.b64encode(img_bytes).decode()
    
    def _create_timeline_chart(self, analysis_results: Dict[str, Any]) -> str:
        """Create timeline chart showing fillers and pauses."""
        fig = go.Figure()
        
        # Add filler events
        if "filler_analysis" in analysis_results:
            fillers = analysis_results["filler_analysis"].get("filler_instances", [])
            if fillers:
                filler_times = [f.get("start", 0) for f in fillers]
                filler_texts = [f.get("text", "") for f in fillers]
                
                fig.add_trace(go.Scatter(
                    x=filler_times,
                    y=[1] * len(filler_times),
                    mode='markers',
                    name='Fillers',
                    text=filler_texts,
                    marker=dict(color=self.colors["warning"], size=8)
                ))
        
        # Add pause events  
        if "pause_analysis" in analysis_results:
            pauses = analysis_results["pause_analysis"].get("timeline_data", [])
            if pauses:
                hesitation_pauses = [p for p in pauses if p.get("type") == "hesitation"]
                rhetorical_pauses = [p for p in pauses if p.get("type") == "rhetorical"]
                
                if hesitation_pauses:
                    fig.add_trace(go.Scatter(
                        x=[p["start"] for p in hesitation_pauses],
                        y=[0.5] * len(hesitation_pauses),
                        mode='markers',
                        name='Hesitation Pauses',
                        marker=dict(color=self.colors["error"], size=6)
                    ))
                
                if rhetorical_pauses:
                    fig.add_trace(go.Scatter(
                        x=[p["start"] for p in rhetorical_pauses],
                        y=[0] * len(rhetorical_pauses),
                        mode='markers',
                        name='Rhetorical Pauses',
                        marker=dict(color=self.colors["success"], size=6)
                    ))
        
        fig.update_layout(
            title="Speech Timeline",
            xaxis_title="Time (seconds)",
            yaxis_title="Event Type",
            width=600,
            height=300,
            margin=dict(t=40, b=40, l=40, r=40)
        )
        
        # Convert to base64
        img_bytes = fig.to_image(format="png", width=600, height=300)
        return base64.b64encode(img_bytes).decode()
    
    def _create_phoneme_chart(self, phoneme_analysis: Dict[str, Any]) -> str:
        """Create phoneme risk bar chart."""
        phoneme_risks = phoneme_analysis.get("phoneme_risks", {})
        
        if not phoneme_risks:
            return ""
        
        groups = list(phoneme_risks.keys())
        scores = [data.get("risk_score", 0) for data in phoneme_risks.values()]
        
        # Color bars based on risk level
        colors = []
        for score in scores:
            if score > 60:
                colors.append(self.colors["error"])
            elif score > 30:
                colors.append(self.colors["warning"])
            else:
                colors.append(self.colors["success"])
        
        fig = go.Figure(data=go.Bar(
            x=groups,
            y=scores,
            marker_color=colors,
            text=[f"{score:.0f}%" for score in scores],
            textposition='auto',
        ))
        
        fig.update_layout(
            title="Pronunciation Risk by Sound Group",
            xaxis_title="Sound Groups",
            yaxis_title="Risk Score (%)",
            width=500,
            height=300,
            margin=dict(t=40, b=40, l=40, r=40)
        )
        
        # Convert to base64
        img_bytes = fig.to_image(format="png", width=500, height=300)
        return base64.b64encode(img_bytes).decode()
    
    def _hex_to_rgba(self, hex_color: str, alpha: float = 1.0) -> tuple:
        """Convert hex color to RGBA tuple."""
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return (r, g, b, alpha)
    
    def _generate_pdf_report(self, 
                           report_data: Dict[str, Any], 
                           charts: Dict[str, str],
                           timestamp: datetime) -> Path:
        """Generate PDF report using ReportLab."""
        filename = f"echo_report_{timestamp.strftime('%Y%m%d_%H%M%S')}.pdf"
        file_path = self.reports_dir / filename
        
        # Create PDF document
        doc = SimpleDocTemplate(str(file_path), pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            textColor=HexColor(self.colors["primary"]),
            alignment=TA_CENTER
        )
        
        header_style = ParagraphStyle(
            'CustomHeader',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            textColor=HexColor(self.colors["text"])
        )
        
        # Title and metadata
        story.append(Paragraph("Echo Speech Analysis Report", title_style))
        story.append(Spacer(1, 12))
        
        # Summary section
        summary = report_data["summary"]
        summary_data = [
            ["Overall Score", f"{summary['overall_score']}/100"],
            ["Audio File", summary["audio_filename"]],
            ["Duration", f"{summary['duration']:.1f}s"],
            ["Words", str(summary["word_count"])],
            ["Analysis Date", timestamp.strftime("%Y-%m-%d %H:%M")],
            ["Generated By", "Echo v1.0.0 (On-device)"]
        ]
        
        summary_table = Table(summary_data, colWidths=[2*inch, 3*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor(self.colors["background"])),
            ('TEXTCOLOR', (0, 0), (-1, 0), HexColor(self.colors["text"])),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, HexColor(self.colors["secondary"]))
        ]))
        
        story.append(summary_table)
        story.append(Spacer(1, 20))
        
        # Add radar chart if available
        if "radar" in charts:
            story.append(Paragraph("Performance Overview", header_style))
            try:
                img_data = base64.b64decode(charts["radar"])
                img_buffer = BytesIO(img_data)
                img = Image(img_buffer, width=4*inch, height=4*inch)
                story.append(img)
                story.append(Spacer(1, 20))
            except Exception as e:
                logger.warning(f"Error adding radar chart to PDF: {e}")
        
        # Detailed sections
        sections = report_data["sections"]
        for section_key, section_data in sections.items():
            story.append(Paragraph(section_data["title"], header_style))
            
            # Section metrics table
            metrics_data = [[k, v] for k, v in section_data["metrics"].items()]
            metrics_table = Table(metrics_data, colWidths=[2*inch, 3*inch])
            metrics_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('VALIGN', (0, 0), (-1, -1), 'TOP')
            ]))
            
            story.append(metrics_table)
            story.append(Spacer(1, 12))
        
        # Tips section
        story.append(Paragraph("Personalized Recommendations", header_style))
        for i, tip in enumerate(report_data["tips"], 1):
            tip_text = f"{i}. {tip}"
            story.append(Paragraph(tip_text, styles['Normal']))
            story.append(Spacer(1, 6))
        
        # Build PDF
        doc.build(story)
        logger.info(f"Generated PDF report: {file_path}")
        
        return file_path
    
    def _generate_json_report(self, report_data: Dict[str, Any], timestamp: datetime) -> Path:
        """Generate JSON report as fallback."""
        filename = f"echo_report_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        file_path = self.reports_dir / filename
        
        with open(file_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        logger.info(f"Generated JSON report: {file_path}")
        return file_path


# Global report generator instance
report_generator = ReportGenerator()