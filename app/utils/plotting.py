import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

logger = logging.getLogger(__name__)


class PlotGenerator:
    """Generate interactive plots for speech analysis visualization."""
    
    def __init__(self):
        self.colors = {
            "primary": "#2563eb",
            "secondary": "#64748b", 
            "success": "#059669",
            "warning": "#d97706",
            "error": "#dc2626",
            "info": "#0891b2",
            "background": "#f8fafc"
        }
        
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available - plotting functionality disabled")
    
    def create_overview_radar(self, analysis_results: Dict[str, Any]) -> Optional[go.Figure]:
        """Create radar chart showing overall performance metrics."""
        if not PLOTLY_AVAILABLE:
            return None
        
        categories = []
        scores = []
        
        # Collect scores from different analyses
        if "clarity_analysis" in analysis_results:
            categories.append("Clarity")
            scores.append(analysis_results["clarity_analysis"].get("clarity_score", 0))
        
        if "pacing_analysis" in analysis_results:
            categories.append("Pacing")
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
            categories.append("Fluency")
            fillers_per_min = analysis_results["filler_analysis"].get("fillers_per_minute", 0)
            fluency_score = max(0, 100 - fillers_per_min * 15)
            scores.append(fluency_score)
        
        if "prosody_analysis" in analysis_results:
            categories.append("Prosody") 
            stability = analysis_results["prosody_analysis"].get("stability_metrics", {})
            prosody_score = stability.get("overall_stability_score", 50)
            scores.append(prosody_score)
        
        if "phoneme_analysis" in analysis_results:
            categories.append("Pronunciation")
            overall_risk = analysis_results["phoneme_analysis"].get("overall_risk", 50)
            pronunciation_score = max(0, 100 - overall_risk)
            scores.append(pronunciation_score)
        
        if len(categories) < 3:
            return None  # Need at least 3 categories for meaningful radar
        
        # Close the radar chart
        categories.append(categories[0])
        scores.append(scores[0])
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=scores,
            theta=categories,
            fill='toself',
            name='Your Performance',
            fillcolor=f'rgba(37, 99, 235, 0.3)',
            line=dict(color=self.colors["primary"], width=3)
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    showticklabels=True,
                    tickmode="linear",
                    tick0=0,
                    dtick=20
                ),
                angularaxis=dict(
                    tickfont_size=12,
                    rotation=90,
                    direction="counterclockwise"
                )
            ),
            showlegend=False,
            title=dict(
                text="Speech Analysis Overview",
                x=0.5,
                font=dict(size=16)
            ),
            font=dict(size=11),
            margin=dict(t=60, b=20, l=20, r=20)
        )
        
        return fig
    
    def create_timeline_chart(self, analysis_results: Dict[str, Any]) -> Optional[go.Figure]:
        """Create timeline showing fillers, pauses, and other events."""
        if not PLOTLY_AVAILABLE:
            return None
        
        fig = go.Figure()
        
        # Add filler markers
        if "filler_analysis" in analysis_results:
            fillers = analysis_results["filler_analysis"].get("filler_instances", [])
            if fillers:
                filler_times = [f.get("start", 0) for f in fillers]
                filler_texts = [f.get("text", "") for f in fillers]
                filler_types = [f.get("type", "vocal_filler") for f in fillers]
                
                fig.add_trace(go.Scatter(
                    x=filler_times,
                    y=[2] * len(filler_times),
                    mode='markers+text',
                    name='Filler Words',
                    text=filler_texts,
                    textposition="top center",
                    marker=dict(
                        color=self.colors["warning"],
                        size=10,
                        symbol="diamond"
                    ),
                    hovertemplate="<b>Filler:</b> %{text}<br><b>Time:</b> %{x:.1f}s<extra></extra>"
                ))
        
        # Add pause markers
        if "pause_analysis" in analysis_results:
            timeline_data = analysis_results["pause_analysis"].get("timeline_data", [])
            
            hesitation_pauses = [p for p in timeline_data if p.get("type") == "hesitation"]
            rhetorical_pauses = [p for p in timeline_data if p.get("type") == "rhetorical"]
            
            if hesitation_pauses:
                fig.add_trace(go.Scatter(
                    x=[p["start"] for p in hesitation_pauses],
                    y=[1] * len(hesitation_pauses),
                    mode='markers',
                    name='Hesitation Pauses',
                    marker=dict(
                        color=self.colors["error"],
                        size=8,
                        symbol="x"
                    ),
                    hovertemplate="<b>Hesitation Pause</b><br><b>Time:</b> %{x:.1f}s<br><b>Duration:</b> %{customdata:.2f}s<extra></extra>",
                    customdata=[p["duration"] for p in hesitation_pauses]
                ))
            
            if rhetorical_pauses:
                fig.add_trace(go.Scatter(
                    x=[p["start"] for p in rhetorical_pauses],
                    y=[0] * len(rhetorical_pauses),
                    mode='markers',
                    name='Rhetorical Pauses',
                    marker=dict(
                        color=self.colors["success"],
                        size=8,
                        symbol="circle"
                    ),
                    hovertemplate="<b>Rhetorical Pause</b><br><b>Time:</b> %{x:.1f}s<br><b>Duration:</b> %{customdata:.2f}s<extra></extra>",
                    customdata=[p["duration"] for p in rhetorical_pauses]
                ))
        
        # Add uptalk markers
        if "prosody_analysis" in analysis_results:
            uptalk_instances = analysis_results["prosody_analysis"].get("uptalk_analysis", {}).get("uptalk_instances", [])
            if uptalk_instances:
                fig.add_trace(go.Scatter(
                    x=[u.get("start", 0) for u in uptalk_instances],
                    y=[3] * len(uptalk_instances),
                    mode='markers',
                    name='Uptalk',
                    marker=dict(
                        color=self.colors["info"],
                        size=8,
                        symbol="triangle-up"
                    ),
                    hovertemplate="<b>Uptalk:</b> %{customdata}<br><b>Time:</b> %{x:.1f}s<extra></extra>",
                    customdata=[u.get("text", "statement") for u in uptalk_instances]
                ))
        
        fig.update_layout(
            title="Speech Events Timeline",
            xaxis_title="Time (seconds)",
            yaxis=dict(
                title="Event Type",
                tickmode="array",
                tickvals=[0, 1, 2, 3],
                ticktext=["Rhetorical Pauses", "Hesitation Pauses", "Fillers", "Uptalk"]
            ),
            hovermode="closest",
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(t=60, b=40, l=60, r=40)
        )
        
        return fig
    
    def create_phoneme_risk_chart(self, phoneme_analysis: Dict[str, Any]) -> Optional[go.Figure]:
        """Create bar chart showing phoneme risk scores."""
        if not PLOTLY_AVAILABLE:
            return None
        
        phoneme_risks = phoneme_analysis.get("phoneme_risks", {})
        if not phoneme_risks:
            return None
        
        groups = list(phoneme_risks.keys())
        scores = [data.get("risk_score", 0) for data in phoneme_risks.values()]
        word_counts = [data.get("word_count", 0) for data in phoneme_risks.values()]
        
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
            textposition='outside',
            hovertemplate="<b>%{x}</b><br><b>Risk Score:</b> %{y:.1f}%<br><b>Words Found:</b> %{customdata}<extra></extra>",
            customdata=word_counts
        ))
        
        fig.update_layout(
            title="Pronunciation Risk by Sound Group",
            xaxis_title="Phoneme Groups",
            yaxis_title="Risk Score (%)",
            yaxis=dict(range=[0, 100]),
            showlegend=False,
            margin=dict(t=40, b=40, l=40, r=40)
        )
        
        # Add risk level annotations
        fig.add_hline(y=30, line_dash="dash", line_color="orange", 
                     annotation_text="Moderate Risk", annotation_position="right")
        fig.add_hline(y=60, line_dash="dash", line_color="red",
                     annotation_text="High Risk", annotation_position="right")
        
        return fig
    
    def create_pace_stability_chart(self, pacing_analysis: Dict[str, Any]) -> Optional[go.Figure]:
        """Create chart showing pace stability over time."""
        if not PLOTLY_AVAILABLE:
            return None
        
        wpm_timeline = pacing_analysis.get("wpm_timeline", [])
        if not wpm_timeline:
            return None
        
        times = [point[0] for point in wpm_timeline]
        wpms = [point[1] for point in wpm_timeline]
        
        target_range = pacing_analysis.get("target_range", [140, 160])
        overall_wpm = pacing_analysis.get("overall_wpm", 0)
        
        fig = go.Figure()
        
        # Add target range shading
        fig.add_shape(
            type="rect",
            x0=min(times) if times else 0,
            x1=max(times) if times else 60,
            y0=target_range[0],
            y1=target_range[1],
            fillcolor="rgba(34, 197, 94, 0.2)",
            line=dict(width=0),
            layer="below"
        )
        
        # Add WPM over time
        fig.add_trace(go.Scatter(
            x=times,
            y=wpms,
            mode='lines+markers',
            name='Speaking Pace',
            line=dict(color=self.colors["primary"], width=2),
            marker=dict(size=6),
            hovertemplate="<b>Time:</b> %{x:.1f}s<br><b>Pace:</b> %{y:.0f} WPM<extra></extra>"
        ))
        
        # Add overall average line
        fig.add_hline(
            y=overall_wpm,
            line_dash="dot",
            line_color=self.colors["secondary"],
            annotation_text=f"Overall: {overall_wpm:.0f} WPM"
        )
        
        fig.update_layout(
            title="Speaking Pace Over Time",
            xaxis_title="Time (seconds)",
            yaxis_title="Words Per Minute",
            showlegend=False,
            margin=dict(t=40, b=40, l=40, r=40)
        )
        
        return fig
    
    def create_clarity_timeline(self, clarity_analysis: Dict[str, Any]) -> Optional[go.Figure]:
        """Create timeline showing clarity scores over time."""
        if not PLOTLY_AVAILABLE:
            return None
        
        clarity_timeline = clarity_analysis.get("clarity_timeline", [])
        if not clarity_timeline:
            return None
        
        times = [(point["start"] + point["end"]) / 2 for point in clarity_timeline]
        confidences = [point["confidence"] * 100 for point in clarity_timeline]  # Convert to percentage
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=times,
            y=confidences,
            mode='lines+markers',
            name='Clarity Score',
            line=dict(color=self.colors["info"], width=2),
            marker=dict(size=6),
            fill='tonexty',
            fillcolor='rgba(8, 145, 178, 0.1)',
            hovertemplate="<b>Time:</b> %{x:.1f}s<br><b>Clarity:</b> %{y:.1f}%<extra></extra>"
        ))
        
        # Add clarity threshold line
        fig.add_hline(
            y=80,
            line_dash="dash",
            line_color=self.colors["warning"],
            annotation_text="Target: 80%"
        )
        
        fig.update_layout(
            title="Speech Clarity Over Time",
            xaxis_title="Time (seconds)",
            yaxis_title="Clarity Score (%)",
            yaxis=dict(range=[0, 100]),
            showlegend=False,
            margin=dict(t=40, b=40, l=40, r=40)
        )
        
        return fig
    
    def create_combined_dashboard(self, analysis_results: Dict[str, Any]) -> Optional[go.Figure]:
        """Create combined dashboard with multiple subplots."""
        if not PLOTLY_AVAILABLE:
            return None
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=["Performance Overview", "Timeline", "Pronunciation Risks", "Pace Stability"],
            specs=[[{"type": "polar"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # Add radar chart (simplified for subplot)
        if "clarity_analysis" in analysis_results:
            categories = ["Clarity", "Pacing", "Fluency", "Prosody"]
            scores = [
                analysis_results.get("clarity_analysis", {}).get("clarity_score", 50),
                80,  # Simplified pacing score
                max(0, 100 - analysis_results.get("filler_analysis", {}).get("fillers_per_minute", 0) * 15),
                analysis_results.get("prosody_analysis", {}).get("stability_metrics", {}).get("overall_stability_score", 50)
            ]
            categories.append(categories[0])  # Close the radar
            scores.append(scores[0])
            
            fig.add_trace(go.Scatterpolar(
                r=scores,
                theta=categories,
                fill='toself',
                name='Performance',
                showlegend=False
            ), row=1, col=1)
        
        # Add timeline (simplified)
        if "filler_analysis" in analysis_results:
            fillers = analysis_results["filler_analysis"].get("filler_instances", [])
            if fillers:
                fig.add_trace(go.Scatter(
                    x=[f.get("start", 0) for f in fillers[:10]],  # Limit to 10 for clarity
                    y=[1] * min(10, len(fillers)),
                    mode='markers',
                    name='Fillers',
                    showlegend=False,
                    marker=dict(color=self.colors["warning"])
                ), row=1, col=2)
        
        return fig


# Global plot generator instance
plot_generator = PlotGenerator()