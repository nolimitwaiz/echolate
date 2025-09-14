#!/usr/bin/env python3
"""
Echo CLI - Command-line interface for speech analysis
Usage: python -m cli.echo_cli input.wav [options]
"""

import argparse
import logging
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

# Add parent directory to path for imports
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
from app.utils.session_store import session_store
from app.settings import settings


class EchoCLI:
    """Command-line interface for Echo speech analysis."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def analyze_file(self, 
                    audio_path: str,
                    speaking_mode: str = "Presentation",
                    output_format: str = "text",
                    output_file: Optional[str] = None,
                    generate_report: bool = False,
                    save_session: bool = False,
                    session_name: Optional[str] = None,
                    verbose: bool = False) -> Dict[str, Any]:
        """Analyze an audio file and return results."""
        
        start_time = time.time()
        
        try:
            # Validate input file
            audio_path = Path(audio_path)
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            if verbose:
                print(f"🎤 Analyzing: {audio_path.name}")
                print(f"📊 Speaking mode: {speaking_mode}")
            
            # Load and preprocess audio
            audio, sr = audio_processor.load_audio_file(audio_path)
            audio = audio_processor.trim_to_max_duration(audio, sr)
            audio = audio_processor.normalize_volume(audio)
            
            duration = len(audio) / sr
            if verbose:
                print(f"⏱️  Audio duration: {duration:.1f}s at {sr}Hz")
            
            # Run ASR
            if verbose:
                print("🗣️  Running speech recognition...")
            asr_result = asr_manager.transcribe(str(audio_path))
            
            if not asr_result.text.strip():
                raise ValueError("No speech detected in audio")
            
            if verbose:
                print(f"✅ Recognized {len(asr_result.words)} words")
                print(f"📝 Transcript: {asr_result.text[:100]}{'...' if len(asr_result.text) > 100 else ''}")
            
            # Run VAD
            vad_segments = vad_processor.process_audio(audio)
            speech_segments = [seg for seg in vad_segments if seg.is_speech]
            total_speech_duration = sum(seg.duration for seg in speech_segments)
            
            if total_speech_duration < 1.0:
                raise ValueError("Insufficient speech detected (less than 1 second)")
            
            if verbose:
                print(f"🎯 Speech analysis: {len(speech_segments)} segments, {total_speech_duration:.1f}s total")
            
            # Run all analyses
            if verbose:
                print("🔍 Running comprehensive analysis...")
            
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
                "audio_filename": audio_path.name,
                "audio_duration": total_speech_duration,
                "word_count": len(asr_result.words),
                "speaking_mode": speaking_mode,
                "processing_time": time.time() - start_time,
                "analysis_timestamp": time.time(),
                "transcript": asr_result.text
            }
            
            # Generate personalized tips
            tips = tips_generator.generate_personalized_tips(
                analysis_results,
                max_tips=7,
                speaking_mode=speaking_mode
            )
            
            # Calculate overall score
            overall_score = self._calculate_overall_score(analysis_results)
            
            # Save session if requested
            session_id = None
            if save_session:
                try:
                    session_id = session_store.save_session(
                        analysis_results,
                        audio_path.name,
                        session_name or f"CLI Analysis {time.strftime('%Y-%m-%d %H:%M')}"
                    )
                    if verbose:
                        print(f"💾 Session saved: {session_id}")
                except Exception as e:
                    if verbose:
                        print(f"⚠️  Failed to save session: {e}")
            
            # Generate report if requested
            report_path = None
            if generate_report:
                try:
                    if verbose:
                        print("📄 Generating PDF report...")
                    report_result = report_generator.generate_report(
                        analysis_results,
                        audio_path.name,
                        "pdf"
                    )
                    if report_result["success"]:
                        report_path = report_result["report_path"]
                        if verbose:
                            print(f"📊 Report saved: {report_path}")
                    else:
                        if verbose:
                            print(f"⚠️  Report generation failed: {report_result.get('error')}")
                except Exception as e:
                    if verbose:
                        print(f"⚠️  Report generation failed: {e}")
            
            # Prepare final results
            final_results = {
                "success": True,
                "overall_score": overall_score,
                "processing_time": time.time() - start_time,
                "session_id": session_id,
                "report_path": report_path,
                "tips": tips,
                "analysis_results": analysis_results
            }
            
            # Output results based on format
            if output_format == "json":
                self._output_json(final_results, output_file)
            else:
                self._output_text(final_results, output_file, verbose)
            
            if verbose:
                print(f"✅ Analysis completed in {final_results['processing_time']:.2f}s")
            
            return final_results
            
        except Exception as e:
            error_result = {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time
            }
            
            if output_format == "json":
                self._output_json(error_result, output_file)
            else:
                print(f"❌ Analysis failed: {e}")
            
            return error_result
    
    def _calculate_overall_score(self, analysis_results: Dict[str, Any]) -> int:
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
    
    def _output_text(self, results: Dict[str, Any], output_file: Optional[str], verbose: bool):
        """Output results in human-readable text format."""
        if not results["success"]:
            output = f"❌ Analysis failed: {results['error']}"
        else:
            output = self._format_text_results(results, verbose)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(output)
            if verbose:
                print(f"📄 Results saved to: {output_file}")
        else:
            print(output)
    
    def _output_json(self, results: Dict[str, Any], output_file: Optional[str]):
        """Output results in JSON format."""
        json_output = json.dumps(results, indent=2, default=str)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(json_output)
        else:
            print(json_output)
    
    def _format_text_results(self, results: Dict[str, Any], verbose: bool) -> str:
        """Format results as human-readable text."""
        analysis = results["analysis_results"]
        overall_score = results["overall_score"]
        
        # Header
        output = []
        output.append("=" * 60)
        output.append("🎤 ECHO SPEECH ANALYSIS RESULTS")
        output.append("=" * 60)
        output.append("")
        
        # Overall score with visual indicator
        score_bar = "█" * (overall_score // 5) + "░" * (20 - overall_score // 5)
        output.append(f"📊 Overall Score: {overall_score}/100")
        output.append(f"    [{score_bar}]")
        output.append("")
        
        # Key metrics
        output.append("📈 KEY METRICS")
        output.append("-" * 40)
        
        # Clarity
        clarity_analysis = analysis.get("clarity_analysis", {})
        clarity_score = clarity_analysis.get("clarity_score", 0)
        clarity_assessment = clarity_analysis.get("assessment", "unknown").replace("_", " ").title()
        output.append(f"🗣️  Speech Clarity:     {clarity_score}/100 ({clarity_assessment})")
        
        # Pacing
        pacing_analysis = analysis.get("pacing_analysis", {})
        wpm = pacing_analysis.get("overall_wpm", 0)
        target_range = pacing_analysis.get("target_range", [140, 160])
        pacing_status = "✅ Optimal" if target_range[0] <= wpm <= target_range[1] else "⚠️ Needs Adjustment"
        output.append(f"⏱️  Speaking Pace:      {wpm} WPM ({pacing_status})")
        
        # Fillers
        filler_analysis = analysis.get("filler_analysis", {})
        filler_count = filler_analysis.get("filler_count", 0)
        fillers_per_min = filler_analysis.get("fillers_per_minute", 0)
        filler_assessment = filler_analysis.get("assessment", "unknown").replace("_", " ").title()
        output.append(f"🔄 Filler Words:       {filler_count} total ({fillers_per_min:.1f}/min - {filler_assessment})")
        
        # Prosody
        prosody_analysis = analysis.get("prosody_analysis", {})
        uptalk_count = prosody_analysis.get("uptalk_analysis", {}).get("uptalk_count", 0)
        stability_score = prosody_analysis.get("stability_metrics", {}).get("overall_stability_score", 0)
        output.append(f"🎵 Prosody Control:    {stability_score:.0f}/100 (Uptalk: {uptalk_count} instances)")
        
        output.append("")
        
        # Pronunciation risks
        phoneme_analysis = analysis.get("phoneme_analysis", {})
        if phoneme_analysis and phoneme_analysis.get("phoneme_risks"):
            output.append("🔤 PRONUNCIATION ANALYSIS")
            output.append("-" * 40)
            
            phoneme_risks = phoneme_analysis["phoneme_risks"]
            for group, data in phoneme_risks.items():
                risk_score = data.get("risk_score", 0)
                word_count = data.get("word_count", 0)
                
                if risk_score > 60:
                    risk_level = "🔴 High Risk"
                elif risk_score > 30:
                    risk_level = "🟡 Moderate Risk"
                else:
                    risk_level = "🟢 Low Risk"
                
                output.append(f"{group} sounds: {risk_score:.0f}% risk ({word_count} words) - {risk_level}")
            
            output.append("")
        
        # Pause analysis
        pause_analysis = analysis.get("pause_analysis", {})
        if pause_analysis and pause_analysis.get("pause_metrics"):
            metrics = pause_analysis["pause_metrics"]
            output.append("⏸️  PAUSE ANALYSIS")
            output.append("-" * 40)
            output.append(f"Total Pauses:        {metrics.get('total_pauses', 0)}")
            output.append(f"Hesitation Pauses:   {metrics.get('hesitation_pauses', 0)} (❌ Reduce these)")
            output.append(f"Rhetorical Pauses:   {metrics.get('rhetorical_pauses', 0)} (✅ Good for emphasis)")
            output.append("")
        
        # Audio quality
        snr_analysis = analysis.get("snr_analysis", {})
        if snr_analysis:
            snr_db = snr_analysis.get("snr_db", 0)
            snr_ok = snr_analysis.get("snr_ok", False)
            quality_status = "✅ Good" if snr_ok else "⚠️ Poor"
            output.append("🎧 AUDIO QUALITY")
            output.append("-" * 40)
            output.append(f"Signal-to-Noise Ratio: {snr_db:.1f} dB ({quality_status})")
            output.append("")
        
        # Tips
        tips = results.get("tips", [])
        if tips:
            output.append("💡 PERSONALIZED RECOMMENDATIONS")
            output.append("-" * 40)
            for i, tip in enumerate(tips, 1):
                # Clean tip text (remove emojis for CLI)
                clean_tip = tip.replace("🎯", "").replace("💡", "").replace("📝", "").replace("🎵", "").replace("⏸️", "").strip()
                output.append(f"{i}. {clean_tip}")
            output.append("")
        
        # Metadata
        if verbose:
            metadata = analysis.get("metadata", {})
            output.append("📋 ANALYSIS DETAILS")
            output.append("-" * 40)
            output.append(f"Audio Duration:      {metadata.get('audio_duration', 0):.1f}s")
            output.append(f"Words Recognized:    {metadata.get('word_count', 0)}")
            output.append(f"Processing Time:     {results.get('processing_time', 0):.2f}s")
            output.append(f"Speaking Mode:       {metadata.get('speaking_mode', 'Unknown')}")
            
            if results.get("session_id"):
                output.append(f"Session ID:          {results['session_id']}")
            
            if results.get("report_path"):
                output.append(f"Report Generated:    {results['report_path']}")
            
            output.append("")
            
            # Transcript
            transcript = metadata.get("transcript", "")
            if transcript:
                output.append("📝 TRANSCRIPT")
                output.append("-" * 40)
                output.append(transcript)
                output.append("")
        
        output.append("=" * 60)
        output.append("Generated by Echo v1.0.0 - On-device speech analysis")
        output.append("=" * 60)
        
        return "\n".join(output)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Echo - Speech Analysis CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m cli.echo_cli recording.wav
  python -m cli.echo_cli speech.mp3 --mode Interview --report
  python -m cli.echo_cli audio.wav --json --output results.json
  python -m cli.echo_cli presentation.wav --save-session --session-name "Practice Run"
        """
    )
    
    parser.add_argument(
        "audio_file",
        help="Path to audio file to analyze"
    )
    
    parser.add_argument(
        "--mode", "-m",
        choices=["Presentation", "Interview", "Casual"],
        default="Presentation",
        help="Speaking context mode (default: Presentation)"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Output file path (default: print to console)"
    )
    
    parser.add_argument(
        "--format", "-f",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)"
    )
    
    parser.add_argument(
        "--report", "-r",
        action="store_true",
        help="Generate PDF report"
    )
    
    parser.add_argument(
        "--save-session", "-s",
        action="store_true",
        help="Save analysis session for later review"
    )
    
    parser.add_argument(
        "--session-name", "-n",
        help="Name for saved session"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output with processing details"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress messages"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="Echo CLI v1.0.0"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    if args.quiet:
        logging.basicConfig(level=logging.ERROR)
    elif args.verbose:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    else:
        logging.basicConfig(level=logging.WARNING)
    
    # Check if audio file exists
    if not Path(args.audio_file).exists():
        print(f"❌ Error: Audio file not found: {args.audio_file}")
        sys.exit(1)
    
    # Initialize CLI
    cli = EchoCLI()
    
    try:
        # Run analysis
        results = cli.analyze_file(
            audio_path=args.audio_file,
            speaking_mode=args.mode,
            output_format=args.format,
            output_file=args.output,
            generate_report=args.report,
            save_session=args.save_session,
            session_name=args.session_name,
            verbose=args.verbose and not args.quiet
        )
        
        # Exit with appropriate code
        sys.exit(0 if results["success"] else 1)
        
    except KeyboardInterrupt:
        print("\n❌ Analysis interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()