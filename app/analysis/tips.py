import logging
from typing import Dict, List, Any, Tuple
import random

from ..settings import settings

logger = logging.getLogger(__name__)


class TipsGenerator:
    """Generate personalized coaching tips based on analysis results."""
    
    def __init__(self):
        self.speaking_mode = "Presentation"  # Default, can be overridden
        
        # Tip templates organized by category
        self.tip_templates = {
            "pacing": {
                "too_slow": [
                    "Your speaking pace is {wpm} WPM, which is slower than ideal. Try to speak with more energy and momentum to reach {target_min}-{target_max} WPM.",
                    "At {wpm} WPM, you're speaking quite slowly. Practice reading aloud with a timer to build up your natural pace.",
                    "Speed up your delivery slightly. Aim for {target_min}-{target_max} WPM to keep your audience engaged."
                ],
                "too_fast": [
                    "You're speaking at {wpm} WPM, which may be too fast for clarity. Slow down to {target_min}-{target_max} WPM for better comprehension.",
                    "At {wpm} WPM, you might be rushing. Take deliberate pauses between ideas to give listeners time to process.",
                    "Your rapid pace of {wpm} WPM could reduce clarity. Practice speaking more deliberately."
                ],
                "optimal": [
                    "Perfect pacing at {wpm} WPM! You're in the sweet spot for {speaking_mode} delivery.",
                    "Excellent pace control at {wpm} WPM. Your audience can easily follow along.",
                    "Great job maintaining {wpm} WPM - this is ideal for {speaking_mode} speaking."
                ]
            },
            "clarity": {
                "excellent": [
                    "Outstanding clarity! Your pronunciation is crisp and easy to understand.",
                    "Excellent articulation. Your words come through crystal clear.",
                    "Perfect clarity score. Your speech is highly intelligible."
                ],
                "good": [
                    "Good clarity overall. Minor improvements in articulation could make you even clearer.",
                    "Nice clear delivery. Consider slowing down slightly on complex words.",
                    "Good pronunciation. Focus on consonant clarity for even better results."
                ],
                "fair": [
                    "Your clarity needs some attention. Slow down and focus on enunciating clearly.",
                    "Practice articulating consonants more distinctly, especially at word endings.",
                    "Work on speaking more deliberately to improve your clarity score."
                ],
                "poor": [
                    "Clarity needs significant improvement. Focus on speaking slowly and distinctly.",
                    "Practice pronunciation exercises, especially for consonant sounds.",
                    "Consider recording yourself to identify which sounds need the most work."
                ]
            },
            "fillers": {
                "excellent": [
                    "Excellent control! You used very few filler words.",
                    "Outstanding fluency with minimal fillers. Keep it up!",
                    "Perfect filler control - your speech flows naturally."
                ],
                "good": [
                    "Good job keeping fillers to a minimum. Your speech flows well.",
                    "Nice control of filler words. Just a few minor instances.",
                    "Good fluency overall with low filler usage."
                ],
                "moderate": [
                    "You used {count} filler words. Practice pausing instead of saying 'um' or 'uh'.",
                    "Try to replace fillers with brief, purposeful pauses to gather your thoughts.",
                    "Work on reducing 'like', 'you know', and other verbal crutches."
                ],
                "high": [
                    "High filler usage detected ({count} words). Slow down and think before speaking.",
                    "Practice eliminating 'um', 'uh', and 'like' by using silent pauses instead.",
                    "Prepare your thoughts in advance to reduce dependency on filler words."
                ],
                "excessive": [
                    "Excessive filler usage ({count} words) disrupts your message. Focus on this as priority #1.",
                    "Your message gets lost in filler words. Practice speaking with intentional pauses.",
                    "Record yourself daily and count fillers to build awareness and improvement."
                ]
            },
            "prosody": {
                "excellent": [
                    "Excellent prosodic control! Your intonation patterns are engaging and natural.",
                    "Outstanding pitch and energy variation. Your delivery is very dynamic.",
                    "Perfect prosody - your speech has excellent rhythm and musicality."
                ],
                "good": [
                    "Good prosodic patterns. Your speech has nice variation in pitch and energy.",
                    "Nice intonation control. Your delivery keeps listeners engaged.",
                    "Good use of pitch variation to emphasize key points."
                ],
                "fair": [
                    "Your prosody could use some work. Try varying your pitch more for emphasis.",
                    "Add more energy variation to make your delivery more engaging.",
                    "Practice emphasizing important words with pitch and volume changes."
                ],
                "needs_improvement": [
                    "Your prosody needs attention. Work on pitch variation and energy control.",
                    "Practice reading with expression to develop better prosodic patterns.",
                    "Your delivery sounds monotone. Focus on adding musical variation to your speech."
                ]
            },
            "uptalk": [
                "You used rising intonation on {count} statements. End declarative sentences with falling pitch to sound more confident.",
                "Avoid uptalk on statements - it can make you sound uncertain. Practice ending with downward pitch.",
                "Your uptalk pattern ({count} instances) undermines authority. Work on confident statement delivery.",
                "Rising intonation on statements makes you sound hesitant. Practice assertive endings."
            ],
            "pauses": {
                "excellent": [
                    "Excellent pause control! You use rhetorical pauses effectively with minimal hesitation.",
                    "Perfect pause patterns - you use silence as a powerful speaking tool.",
                    "Outstanding control of pauses. Your hesitation pauses are minimal."
                ],
                "good": [
                    "Good use of pauses overall. Nice balance between rhetorical and hesitation pauses.",
                    "You use pauses effectively, with just a few hesitation instances.",
                    "Good pause discipline with effective use of rhetorical pauses."
                ],
                "fair": [
                    "Your pause patterns need work. Focus on reducing hesitation pauses.",
                    "Try to convert hesitation pauses into purposeful, rhetorical ones.",
                    "Work on smoother transitions between ideas to reduce hesitation pauses."
                ],
                "needs_improvement": [
                    "Many hesitation pauses detected. Practice speaking with better preparation.",
                    "Your pause patterns suggest uncertainty. Work on confident, purposeful pausing.",
                    "Reduce hesitation pauses by thinking through your ideas before speaking."
                ]
            },
            "phonemes": {
                "low": [
                    "Excellent pronunciation! Your accent risks are minimal.",
                    "Outstanding articulation of challenging sound groups.",
                    "Perfect pronunciation control across all phoneme categories."
                ],
                "moderate": [
                    "Good pronunciation with some minor accent patterns to address.",
                    "Most sounds are clear, but focus on {risk_groups} for improvement.",
                    "Nice pronunciation overall. Target practice on {risk_groups} sounds."
                ],
                "high": [
                    "Several pronunciation challenges detected. Focus on {risk_groups} sounds.",
                    "Your accent patterns need attention, particularly {risk_groups}.",
                    "Work on {risk_groups} pronunciation using the practice drills."
                ],
                "very_high": [
                    "Significant pronunciation challenges. Consider accent coaching for {risk_groups}.",
                    "Multiple sound groups need work: {risk_groups}. Use targeted practice drills.",
                    "Strong accent patterns detected. Focus intensive practice on {risk_groups}."
                ]
            },
            "snr": [
                "High background noise detected. For accurate analysis, please record in a quieter environment.",
                "Audio quality affects analysis accuracy. Consider using a better microphone or quieter space.",
                "Background noise is interfering with the analysis. Try recording closer to your microphone."
            ],
            "general_encouragement": [
                "Keep practicing! Consistent effort leads to noticeable improvement in speech clarity.",
                "Great job analyzing your speech! Awareness is the first step to improvement.",
                "You're on the right track! Focus on one area at a time for best results.",
                "Excellent work! Small improvements in each area add up to big gains.",
                "Nice progress! Regular practice with Echo will help you continue improving."
            ]
        }
        
        # Context-specific modifiers
        self.context_modifiers = {
            "Presentation": {
                "pace_preference": "slightly slower for emphasis",
                "pause_preference": "longer rhetorical pauses",
                "energy_preference": "consistent, confident energy"
            },
            "Interview": {
                "pace_preference": "moderate pace for clarity",
                "pause_preference": "brief, thoughtful pauses",
                "energy_preference": "calm, steady energy"
            },
            "Casual": {
                "pace_preference": "natural conversational pace",
                "pause_preference": "natural pause patterns",
                "energy_preference": "relaxed, authentic energy"
            }
        }
    
    def generate_personalized_tips(self, analysis_results: Dict[str, Any], 
                                 max_tips: int = 5,
                                 speaking_mode: str = "Presentation") -> List[str]:
        """Generate personalized tips based on comprehensive analysis results."""
        self.speaking_mode = speaking_mode
        tips = []
        
        try:
            # Process each analysis category
            if "pacing_analysis" in analysis_results:
                tips.extend(self._get_pacing_tips(analysis_results["pacing_analysis"]))
            
            if "clarity_analysis" in analysis_results:
                tips.extend(self._get_clarity_tips(analysis_results["clarity_analysis"]))
            
            if "filler_analysis" in analysis_results:
                tips.extend(self._get_filler_tips(analysis_results["filler_analysis"]))
            
            if "prosody_analysis" in analysis_results:
                tips.extend(self._get_prosody_tips(analysis_results["prosody_analysis"]))
            
            if "pause_analysis" in analysis_results:
                tips.extend(self._get_pause_tips(analysis_results["pause_analysis"]))
            
            if "phoneme_analysis" in analysis_results:
                tips.extend(self._get_phoneme_tips(analysis_results["phoneme_analysis"]))
            
            if "snr_analysis" in analysis_results:
                tips.extend(self._get_snr_tips(analysis_results["snr_analysis"]))
            
            # Prioritize tips by impact and add encouraging message
            prioritized_tips = self._prioritize_tips(tips, analysis_results)
            
            # Limit to max_tips and add encouragement
            final_tips = prioritized_tips[:max_tips-1]
            final_tips.append(random.choice(self.tip_templates["general_encouragement"]))
            
            return final_tips
            
        except Exception as e:
            logger.error(f"Error generating tips: {e}")
            return ["Keep practicing to improve your speech clarity and confidence!"]
    
    def _get_pacing_tips(self, pacing_analysis: Dict[str, Any]) -> List[str]:
        """Generate pacing-specific tips."""
        tips = []
        
        assessment = pacing_analysis.get("assessment", "unknown")
        wpm = pacing_analysis.get("overall_wpm", 0)
        target_range = pacing_analysis.get("target_range", [140, 160])
        
        if assessment in self.tip_templates["pacing"]:
            template = random.choice(self.tip_templates["pacing"][assessment])
            tip = template.format(
                wpm=wpm,
                target_min=target_range[0],
                target_max=target_range[1],
                speaking_mode=self.speaking_mode
            )
            tips.append(tip)
        
        return tips
    
    def _get_clarity_tips(self, clarity_analysis: Dict[str, Any]) -> List[str]:
        """Generate clarity-specific tips."""
        tips = []
        
        assessment = clarity_analysis.get("assessment", "unknown")
        
        if assessment in self.tip_templates["clarity"]:
            tip = random.choice(self.tip_templates["clarity"][assessment])
            tips.append(tip)
        
        return tips
    
    def _get_filler_tips(self, filler_analysis: Dict[str, Any]) -> List[str]:
        """Generate filler-specific tips."""
        tips = []
        
        assessment = filler_analysis.get("assessment", "unknown")
        count = filler_analysis.get("filler_count", 0)
        
        if assessment in self.tip_templates["fillers"]:
            template = random.choice(self.tip_templates["fillers"][assessment])
            tip = template.format(count=count)
            tips.append(tip)
        
        return tips
    
    def _get_prosody_tips(self, prosody_analysis: Dict[str, Any]) -> List[str]:
        """Generate prosody-specific tips."""
        tips = []
        
        assessment = prosody_analysis.get("assessment", "unknown")
        
        if assessment in self.tip_templates["prosody"]:
            tip = random.choice(self.tip_templates["prosody"][assessment])
            tips.append(tip)
        
        # Uptalk specific tips
        uptalk_analysis = prosody_analysis.get("uptalk_analysis", {})
        uptalk_count = uptalk_analysis.get("uptalk_count", 0)
        
        if uptalk_count > 0:
            template = random.choice(self.tip_templates["uptalk"])
            tip = template.format(count=uptalk_count)
            tips.append(tip)
        
        return tips
    
    def _get_pause_tips(self, pause_analysis: Dict[str, Any]) -> List[str]:
        """Generate pause-specific tips."""
        tips = []
        
        assessment = pause_analysis.get("assessment", "unknown")
        
        if assessment in self.tip_templates["pauses"]:
            tip = random.choice(self.tip_templates["pauses"][assessment])
            tips.append(tip)
        
        return tips
    
    def _get_phoneme_tips(self, phoneme_analysis: Dict[str, Any]) -> List[str]:
        """Generate phoneme-specific tips."""
        tips = []
        
        assessment = phoneme_analysis.get("assessment", "unknown")
        phoneme_risks = phoneme_analysis.get("phoneme_risks", {})
        
        # Find high-risk groups
        risk_groups = []
        for group, data in phoneme_risks.items():
            if data.get("risk_score", 0) > 40:
                risk_groups.append(group)
        
        if assessment in self.tip_templates["phonemes"]:
            template = random.choice(self.tip_templates["phonemes"][assessment])
            risk_groups_str = ", ".join(risk_groups) if risk_groups else "key sounds"
            tip = template.format(risk_groups=risk_groups_str)
            tips.append(tip)
        
        return tips
    
    def _get_snr_tips(self, snr_analysis: Dict[str, Any]) -> List[str]:
        """Generate SNR/audio quality tips."""
        tips = []
        
        if not snr_analysis.get("snr_ok", True):
            tip = random.choice(self.tip_templates["snr"])
            tips.append(tip)
        
        return tips
    
    def _prioritize_tips(self, tips: List[str], analysis_results: Dict[str, Any]) -> List[str]:
        """Prioritize tips based on impact potential."""
        if len(tips) <= 5:
            return tips
        
        # Simple prioritization based on severity
        priority_scores = {}
        
        for i, tip in enumerate(tips):
            score = 1.0  # Base score
            
            # Boost priority for critical issues
            if any(word in tip.lower() for word in ["excessive", "significant", "poor", "needs attention"]):
                score += 2.0
            elif any(word in tip.lower() for word in ["high", "many", "work on"]):
                score += 1.0
            elif any(word in tip.lower() for word in ["excellent", "perfect", "outstanding"]):
                score -= 0.5  # Lower priority for positive feedback
            
            priority_scores[i] = score
        
        # Sort by priority score
        sorted_indices = sorted(priority_scores.keys(), key=lambda x: priority_scores[x], reverse=True)
        return [tips[i] for i in sorted_indices]
    
    def get_drill_tip(self, phoneme_group: str, risk_score: float) -> str:
        """Generate specific tip for phoneme drill practice."""
        drill_tips = {
            "TH": [
                "Place your tongue lightly between your teeth for 'th' sounds.",
                "Practice minimal pairs: 'think' vs 'sink', 'this' vs 'dis'.",
                "Feel the air flow over your tongue for voiceless 'th' sounds."
            ],
            "RL": [
                "For 'R', curl your tongue back without touching the roof of your mouth.",
                "For 'L', touch the tip of your tongue to the ridge behind your upper teeth.",
                "Practice 'red' vs 'led' and 'right' vs 'light' to feel the difference."
            ],
            "WV": [
                "For 'V', gently touch your upper teeth to your lower lip.",
                "For 'W', round your lips and don't let teeth touch lips.",
                "Practice 'very' vs 'wery' and 'vest' vs 'west'."
            ],
            "ZH": [
                "Make a 'sh' sound but add voice - your vocal cords should vibrate.",
                "Practice words like 'measure', 'pleasure', 'vision'.",
                "Feel the vibration in your throat for the /ʒ/ sound."
            ],
            "NG": [
                "Keep your tongue back and don't add a 'k' sound at the end.",
                "Practice 'running' vs 'run-king' - no extra consonant.",
                "The sound comes from the back of your mouth, not the front."
            ]
        }
        
        base_tips = drill_tips.get(phoneme_group, ["Practice this sound group carefully."])
        
        if risk_score > 70:
            return f"High risk detected! {random.choice(base_tips)} Take your time with each repetition."
        elif risk_score > 40:
            return f"Moderate risk. {random.choice(base_tips)}"
        else:
            return f"Low risk - great job! {random.choice(base_tips)} Keep practicing for consistency."


# Global tips generator instance
tips_generator = TipsGenerator()