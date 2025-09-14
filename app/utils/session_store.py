import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from uuid import uuid4

from ..settings import settings

logger = logging.getLogger(__name__)


class SessionStore:
    """Handle session storage and history management."""
    
    def __init__(self):
        self.storage_dir = Path(settings.storage.cache_dir) / "sessions"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.session_limit = settings.storage.session_history_limit
        self.drill_limit = settings.storage.drill_history_limit
        
        self.sessions_file = self.storage_dir / "sessions.json"
        self.drill_history_file = self.storage_dir / "drill_history.json"
        
        # Initialize files if they don't exist
        if not self.sessions_file.exists():
            self._save_sessions([])
        
        if not self.drill_history_file.exists():
            self._save_drill_history({})
    
    def save_session(self, 
                    analysis_results: Dict[str, Any],
                    audio_filename: str = "recording.wav",
                    session_name: Optional[str] = None) -> str:
        """Save analysis session and return session ID."""
        try:
            session_id = str(uuid4())
            timestamp = datetime.now()
            
            session_data = {
                "session_id": session_id,
                "timestamp": timestamp.isoformat(),
                "audio_filename": audio_filename,
                "session_name": session_name or f"Session {timestamp.strftime('%Y-%m-%d %H:%M')}",
                "analysis_results": analysis_results,
                "summary": self._extract_summary(analysis_results)
            }
            
            # Load existing sessions
            sessions = self._load_sessions()
            
            # Add new session at the beginning
            sessions.insert(0, session_data)
            
            # Trim to limit
            if len(sessions) > self.session_limit:
                sessions = sessions[:self.session_limit]
            
            # Save back
            self._save_sessions(sessions)
            
            logger.info(f"Saved session: {session_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Error saving session: {e}")
            raise
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific session by ID."""
        try:
            sessions = self._load_sessions()
            for session in sessions:
                if session["session_id"] == session_id:
                    return session
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving session {session_id}: {e}")
            return None
    
    def get_recent_sessions(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get list of recent sessions."""
        try:
            sessions = self._load_sessions()
            if limit:
                sessions = sessions[:limit]
            
            # Return only summary data, not full analysis results
            return [
                {
                    "session_id": s["session_id"],
                    "timestamp": s["timestamp"],
                    "audio_filename": s["audio_filename"],
                    "session_name": s["session_name"],
                    "summary": s["summary"]
                }
                for s in sessions
            ]
            
        except Exception as e:
            logger.error(f"Error retrieving recent sessions: {e}")
            return []
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a specific session."""
        try:
            sessions = self._load_sessions()
            original_count = len(sessions)
            
            sessions = [s for s in sessions if s["session_id"] != session_id]
            
            if len(sessions) < original_count:
                self._save_sessions(sessions)
                logger.info(f"Deleted session: {session_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error deleting session {session_id}: {e}")
            return False
    
    def save_drill_result(self, 
                         phoneme_group: str,
                         risk_score: float,
                         drill_text: str = "") -> None:
        """Save drill practice result."""
        try:
            drill_history = self._load_drill_history()
            
            if phoneme_group not in drill_history:
                drill_history[phoneme_group] = []
            
            drill_entry = {
                "timestamp": datetime.now().isoformat(),
                "risk_score": risk_score,
                "drill_text": drill_text
            }
            
            # Add to beginning of list
            drill_history[phoneme_group].insert(0, drill_entry)
            
            # Trim to limit
            if len(drill_history[phoneme_group]) > self.drill_limit:
                drill_history[phoneme_group] = drill_history[phoneme_group][:self.drill_limit]
            
            self._save_drill_history(drill_history)
            logger.debug(f"Saved drill result for {phoneme_group}: {risk_score}")
            
        except Exception as e:
            logger.error(f"Error saving drill result: {e}")
    
    def get_drill_history(self, phoneme_group: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get drill history for a specific phoneme group."""
        try:
            drill_history = self._load_drill_history()
            group_history = drill_history.get(phoneme_group, [])
            
            if limit:
                group_history = group_history[:limit]
            
            return group_history
            
        except Exception as e:
            logger.error(f"Error retrieving drill history for {phoneme_group}: {e}")
            return []
    
    def get_all_drill_history(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get all drill history."""
        try:
            return self._load_drill_history()
        except Exception as e:
            logger.error(f"Error retrieving all drill history: {e}")
            return {}
    
    def get_drill_progress(self, phoneme_group: str) -> Dict[str, Any]:
        """Get progress metrics for a phoneme group."""
        try:
            history = self.get_drill_history(phoneme_group)
            
            if not history:
                return {
                    "phoneme_group": phoneme_group,
                    "total_attempts": 0,
                    "best_score": 0,
                    "latest_score": 0,
                    "improvement": 0,
                    "trend": []
                }
            
            scores = [entry["risk_score"] for entry in history]
            latest_score = scores[0] if scores else 0
            best_score = min(scores) if scores else 0  # Lower risk score is better
            
            # Calculate improvement (first vs latest)
            if len(scores) > 1:
                improvement = scores[-1] - latest_score  # Positive = improvement (risk reduction)
            else:
                improvement = 0
            
            # Create trend data (last 10 attempts)
            trend = scores[:10] if len(scores) >= 10 else scores
            trend.reverse()  # Chronological order for trend
            
            return {
                "phoneme_group": phoneme_group,
                "total_attempts": len(history),
                "best_score": best_score,
                "latest_score": latest_score,
                "improvement": improvement,
                "trend": trend
            }
            
        except Exception as e:
            logger.error(f"Error calculating drill progress for {phoneme_group}: {e}")
            return {}
    
    def _load_sessions(self) -> List[Dict[str, Any]]:
        """Load sessions from file."""
        try:
            with open(self.sessions_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []
    
    def _save_sessions(self, sessions: List[Dict[str, Any]]) -> None:
        """Save sessions to file."""
        with open(self.sessions_file, 'w') as f:
            json.dump(sessions, f, indent=2)
    
    def _load_drill_history(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load drill history from file."""
        try:
            with open(self.drill_history_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
    
    def _save_drill_history(self, drill_history: Dict[str, List[Dict[str, Any]]]) -> None:
        """Save drill history to file."""
        with open(self.drill_history_file, 'w') as f:
            json.dump(drill_history, f, indent=2)
    
    def _extract_summary(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract summary metrics from analysis results."""
        summary = {}
        
        try:
            # Overall score
            summary["overall_score"] = self._calculate_overall_score(analysis_results)
            
            # Key metrics
            if "clarity_analysis" in analysis_results:
                summary["clarity_score"] = analysis_results["clarity_analysis"].get("clarity_score", 0)
            
            if "pacing_analysis" in analysis_results:
                summary["wpm"] = analysis_results["pacing_analysis"].get("overall_wpm", 0)
            
            if "filler_analysis" in analysis_results:
                summary["filler_count"] = analysis_results["filler_analysis"].get("filler_count", 0)
            
            if "prosody_analysis" in analysis_results:
                uptalk_count = analysis_results["prosody_analysis"].get("uptalk_analysis", {}).get("uptalk_count", 0)
                summary["uptalk_count"] = uptalk_count
            
            if "phoneme_analysis" in analysis_results:
                summary["pronunciation_risk"] = analysis_results["phoneme_analysis"].get("overall_risk", 0)
            
        except Exception as e:
            logger.error(f"Error extracting summary: {e}")
        
        return summary
    
    def _calculate_overall_score(self, analysis_results: Dict[str, Any]) -> int:
        """Calculate overall score for session summary."""
        scores = []
        
        # Collect individual scores
        if "clarity_analysis" in analysis_results:
            scores.append(analysis_results["clarity_analysis"].get("clarity_score", 0))
        
        if "pacing_analysis" in analysis_results:
            wpm = analysis_results["pacing_analysis"].get("overall_wpm", 0)
            # Convert WPM to score
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
        
        return round(sum(scores) / len(scores)) if scores else 0
    
    def cleanup_old_sessions(self, days_old: int = 30) -> int:
        """Clean up sessions older than specified days."""
        try:
            from datetime import timedelta
            
            cutoff_date = datetime.now() - timedelta(days=days_old)
            sessions = self._load_sessions()
            
            original_count = len(sessions)
            sessions = [
                s for s in sessions 
                if datetime.fromisoformat(s["timestamp"]) > cutoff_date
            ]
            
            deleted_count = original_count - len(sessions)
            
            if deleted_count > 0:
                self._save_sessions(sessions)
                logger.info(f"Cleaned up {deleted_count} old sessions")
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error cleaning up old sessions: {e}")
            return 0


# Global session store instance
session_store = SessionStore()