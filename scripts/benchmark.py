#!/usr/bin/env python3
"""
Benchmark script for Echo performance analysis.
Tests processing time and accuracy across different components.
"""

import time
import statistics
import sys
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
import json

# Add parent directory to path
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
from app.utils.audio_io import audio_processor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EchoBenchmark:
    """Benchmark Echo performance components."""
    
    def __init__(self):
        self.results = {}
        
    def time_function(self, func, *args, **kwargs) -> Tuple[Any, float]:
        """Time a function execution."""
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        return result, end_time - start_time
    
    def benchmark_audio_loading(self, audio_path: str, iterations: int = 3) -> Dict[str, float]:
        """Benchmark audio loading performance."""
        logger.info(f"🎵 Benchmarking audio loading ({iterations} iterations)...")
        
        times = []
        
        for i in range(iterations):
            _, duration = self.time_function(audio_processor.load_audio_file, audio_path)
            times.append(duration)
        
        return {
            "mean": statistics.mean(times),
            "median": statistics.median(times),
            "std": statistics.stdev(times) if len(times) > 1 else 0,
            "min": min(times),
            "max": max(times),
            "iterations": iterations
        }
    
    def benchmark_asr(self, audio_path: str, iterations: int = 3) -> Dict[str, Any]:
        """Benchmark ASR performance."""
        logger.info(f"🗣️  Benchmarking ASR ({iterations} iterations)...")
        
        available_engines = asr_manager.get_available_engines()
        results = {}
        
        for engine in available_engines:
            logger.info(f"    Testing {engine}...")
            times = []
            word_counts = []
            
            for i in range(iterations):
                result, duration = self.time_function(asr_manager.transcribe, audio_path, engine)
                times.append(duration)
                word_counts.append(len(result.words))
            
            results[engine] = {
                "mean_time": statistics.mean(times),
                "median_time": statistics.median(times),
                "std_time": statistics.stdev(times) if len(times) > 1 else 0,
                "min_time": min(times),
                "max_time": max(times),
                "mean_words": statistics.mean(word_counts),
                "iterations": iterations
            }
        
        return results
    
    def benchmark_vad(self, audio_path: str, iterations: int = 5) -> Dict[str, float]:
        """Benchmark VAD performance."""
        logger.info(f"🎯 Benchmarking VAD ({iterations} iterations)...")
        
        # Load audio once
        audio, sr = audio_processor.load_audio_file(audio_path)
        
        times = []
        segment_counts = []
        
        for i in range(iterations):
            result, duration = self.time_function(vad_processor.process_audio, audio)
            times.append(duration)
            segment_counts.append(len(result))
        
        return {
            "mean_time": statistics.mean(times),
            "median_time": statistics.median(times),
            "std_time": statistics.stdev(times) if len(times) > 1 else 0,
            "min_time": min(times),
            "max_time": max(times),
            "mean_segments": statistics.mean(segment_counts),
            "iterations": iterations
        }
    
    def benchmark_analysis_modules(self, audio_path: str, iterations: int = 3) -> Dict[str, Dict[str, float]]:
        """Benchmark all analysis modules."""
        logger.info(f"🔬 Benchmarking analysis modules ({iterations} iterations)...")
        
        # Prepare data once
        audio, sr = audio_processor.load_audio_file(audio_path)
        asr_result = asr_manager.transcribe(audio_path)
        vad_segments = vad_processor.process_audio(audio)
        speech_segments = [seg for seg in vad_segments if seg.is_speech]
        total_speech_duration = sum(seg.duration for seg in speech_segments)
        
        modules = {
            "SNR Analysis": lambda: snr_analyzer.analyze_audio(audio, sr),
            "Pacing Analysis": lambda: pacing_analyzer.analyze(asr_result, vad_segments),
            "Filler Analysis": lambda: filler_analyzer.analyze(asr_result, total_speech_duration),
            "Clarity Analysis": lambda: clarity_analyzer.analyze(asr_result),
            "Phoneme Analysis": lambda: phonemes_analyzer.analyze(asr_result),
            "Prosody Analysis": lambda: prosody_analyzer.analyze(audio, sr, asr_result),
            "Pause Analysis": lambda: pause_analyzer.analyze(vad_segments, asr_result)
        }
        
        results = {}
        
        for module_name, module_func in modules.items():
            logger.info(f"    Testing {module_name}...")
            times = []
            
            for i in range(iterations):
                try:
                    _, duration = self.time_function(module_func)
                    times.append(duration)
                except Exception as e:
                    logger.error(f"    ❌ {module_name} failed: {e}")
                    times.append(float('inf'))  # Mark as failed
            
            # Filter out failed runs
            valid_times = [t for t in times if t != float('inf')]
            
            if valid_times:
                results[module_name] = {
                    "mean_time": statistics.mean(valid_times),
                    "median_time": statistics.median(valid_times),
                    "std_time": statistics.stdev(valid_times) if len(valid_times) > 1 else 0,
                    "min_time": min(valid_times),
                    "max_time": max(valid_times),
                    "success_rate": len(valid_times) / len(times),
                    "iterations": len(valid_times)
                }
            else:
                results[module_name] = {
                    "mean_time": 0,
                    "success_rate": 0,
                    "iterations": 0
                }
        
        return results
    
    def benchmark_end_to_end(self, audio_path: str, iterations: int = 3) -> Dict[str, Any]:
        """Benchmark complete end-to-end analysis."""
        logger.info(f"🏁 Benchmarking end-to-end analysis ({iterations} iterations)...")
        
        times = []
        component_times = []
        
        for i in range(iterations):
            start_time = time.perf_counter()
            
            # Component timing
            component_time = {}
            
            # Audio loading
            audio_start = time.perf_counter()
            audio, sr = audio_processor.load_audio_file(audio_path)
            audio = audio_processor.trim_to_max_duration(audio, sr)
            audio = audio_processor.normalize_volume(audio)
            component_time["audio_loading"] = time.perf_counter() - audio_start
            
            # ASR
            asr_start = time.perf_counter()
            asr_result = asr_manager.transcribe(audio_path)
            component_time["asr"] = time.perf_counter() - asr_start
            
            if not asr_result.text.strip():
                logger.warning(f"    No speech detected in iteration {i+1}")
                continue
            
            # VAD
            vad_start = time.perf_counter()
            vad_segments = vad_processor.process_audio(audio)
            speech_segments = [seg for seg in vad_segments if seg.is_speech]
            total_speech_duration = sum(seg.duration for seg in speech_segments)
            component_time["vad"] = time.perf_counter() - vad_start
            
            if total_speech_duration < 1.0:
                logger.warning(f"    Insufficient speech in iteration {i+1}")
                continue
            
            # All analyses
            analysis_start = time.perf_counter()
            
            # Run all analyses (simplified for benchmark)
            snr_analyzer.analyze_audio(audio, sr)
            pacing_analyzer.analyze(asr_result, vad_segments)
            filler_analyzer.analyze(asr_result, total_speech_duration)
            clarity_analyzer.analyze(asr_result)
            phonemes_analyzer.analyze(asr_result)
            prosody_analyzer.analyze(audio, sr, asr_result)
            pause_analyzer.analyze(vad_segments, asr_result)
            
            component_time["analysis"] = time.perf_counter() - analysis_start
            
            total_time = time.perf_counter() - start_time
            times.append(total_time)
            component_times.append(component_time)
        
        if not times:
            return {"error": "No successful runs"}
        
        # Average component times
        avg_components = {}
        for component in component_times[0].keys():
            avg_components[component] = statistics.mean([ct[component] for ct in component_times])
        
        return {
            "total_time": {
                "mean": statistics.mean(times),
                "median": statistics.median(times),
                "std": statistics.stdev(times) if len(times) > 1 else 0,
                "min": min(times),
                "max": max(times),
            },
            "component_breakdown": avg_components,
            "successful_runs": len(times),
            "total_runs": iterations
        }
    
    def run_full_benchmark(self, audio_path: str) -> Dict[str, Any]:
        """Run complete benchmark suite."""
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        logger.info("🚀 Starting Echo Performance Benchmark")
        logger.info("=" * 60)
        
        benchmark_start = time.perf_counter()
        
        # System info
        import platform
        system_info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "processor": platform.processor(),
            "available_engines": asr_manager.get_available_engines()
        }
        
        # Audio info
        audio, sr = audio_processor.load_audio_file(audio_path)
        audio_info = {
            "filename": Path(audio_path).name,
            "duration": len(audio) / sr,
            "sample_rate": sr,
            "channels": 1,  # Always mono after processing
            "samples": len(audio)
        }
        
        results = {
            "timestamp": time.time(),
            "system_info": system_info,
            "audio_info": audio_info,
            "benchmarks": {}
        }
        
        # Run benchmarks
        try:
            results["benchmarks"]["audio_loading"] = self.benchmark_audio_loading(audio_path)
        except Exception as e:
            logger.error(f"Audio loading benchmark failed: {e}")
            results["benchmarks"]["audio_loading"] = {"error": str(e)}
        
        try:
            results["benchmarks"]["asr"] = self.benchmark_asr(audio_path)
        except Exception as e:
            logger.error(f"ASR benchmark failed: {e}")
            results["benchmarks"]["asr"] = {"error": str(e)}
        
        try:
            results["benchmarks"]["vad"] = self.benchmark_vad(audio_path)
        except Exception as e:
            logger.error(f"VAD benchmark failed: {e}")
            results["benchmarks"]["vad"] = {"error": str(e)}
        
        try:
            results["benchmarks"]["analysis_modules"] = self.benchmark_analysis_modules(audio_path)
        except Exception as e:
            logger.error(f"Analysis modules benchmark failed: {e}")
            results["benchmarks"]["analysis_modules"] = {"error": str(e)}
        
        try:
            results["benchmarks"]["end_to_end"] = self.benchmark_end_to_end(audio_path)
        except Exception as e:
            logger.error(f"End-to-end benchmark failed: {e}")
            results["benchmarks"]["end_to_end"] = {"error": str(e)}
        
        total_benchmark_time = time.perf_counter() - benchmark_start
        results["benchmark_duration"] = total_benchmark_time
        
        return results
    
    def format_results(self, results: Dict[str, Any]) -> str:
        """Format benchmark results for display."""
        output = []
        output.append("🚀 ECHO PERFORMANCE BENCHMARK RESULTS")
        output.append("=" * 60)
        output.append("")
        
        # System info
        system_info = results["system_info"]
        output.append("💻 SYSTEM INFORMATION")
        output.append("-" * 30)
        output.append(f"Platform:         {system_info['platform']}")
        output.append(f"Python Version:   {system_info['python_version']}")
        output.append(f"Processor:        {system_info.get('processor', 'Unknown')}")
        output.append(f"ASR Engines:      {', '.join(system_info['available_engines'])}")
        output.append("")
        
        # Audio info
        audio_info = results["audio_info"]
        output.append("🎵 AUDIO INFORMATION")
        output.append("-" * 30)
        output.append(f"Filename:         {audio_info['filename']}")
        output.append(f"Duration:         {audio_info['duration']:.1f}s")
        output.append(f"Sample Rate:      {audio_info['sample_rate']} Hz")
        output.append(f"Total Samples:    {audio_info['samples']:,}")
        output.append("")
        
        # Benchmarks
        benchmarks = results["benchmarks"]
        
        # Audio Loading
        if "audio_loading" in benchmarks and "error" not in benchmarks["audio_loading"]:
            al = benchmarks["audio_loading"]
            output.append("📂 AUDIO LOADING PERFORMANCE")
            output.append("-" * 30)
            output.append(f"Mean Time:        {al['mean']:.3f}s")
            output.append(f"Median Time:      {al['median']:.3f}s")
            output.append(f"Min/Max:          {al['min']:.3f}s / {al['max']:.3f}s")
            output.append("")
        
        # ASR Performance
        if "asr" in benchmarks and "error" not in benchmarks["asr"]:
            asr_results = benchmarks["asr"]
            output.append("🗣️  ASR PERFORMANCE")
            output.append("-" * 30)
            for engine, metrics in asr_results.items():
                output.append(f"{engine}:")
                output.append(f"  Mean Time:      {metrics['mean_time']:.3f}s")
                output.append(f"  Mean Words:     {metrics['mean_words']:.1f}")
                output.append(f"  Words/Second:   {metrics['mean_words']/metrics['mean_time']:.1f}")
            output.append("")
        
        # Analysis Modules
        if "analysis_modules" in benchmarks and "error" not in benchmarks["analysis_modules"]:
            am = benchmarks["analysis_modules"]
            output.append("🔬 ANALYSIS MODULES PERFORMANCE")
            output.append("-" * 30)
            for module, metrics in am.items():
                if metrics.get("success_rate", 0) > 0:
                    output.append(f"{module}:")
                    output.append(f"  Mean Time:      {metrics['mean_time']:.3f}s")
                    output.append(f"  Success Rate:   {metrics['success_rate']:.1%}")
                else:
                    output.append(f"{module}: FAILED")
            output.append("")
        
        # End-to-End
        if "end_to_end" in benchmarks and "error" not in benchmarks["end_to_end"]:
            e2e = benchmarks["end_to_end"]
            if "error" not in e2e:
                output.append("🏁 END-TO-END PERFORMANCE")
                output.append("-" * 30)
                total = e2e["total_time"]
                output.append(f"Mean Total Time:  {total['mean']:.3f}s")
                output.append(f"Median Time:      {total['median']:.3f}s")
                output.append(f"Min/Max:          {total['min']:.3f}s / {total['max']:.3f}s")
                output.append("")
                
                output.append("Component Breakdown:")
                breakdown = e2e["component_breakdown"]
                for component, time_val in breakdown.items():
                    percentage = (time_val / total['mean']) * 100
                    output.append(f"  {component.replace('_', ' ').title():15} {time_val:.3f}s ({percentage:.1f}%)")
                output.append("")
        
        # Performance Summary
        output.append("📊 PERFORMANCE SUMMARY")
        output.append("-" * 30)
        
        # Calculate processing speed relative to audio duration
        if "end_to_end" in benchmarks and "error" not in benchmarks["end_to_end"]:
            e2e = benchmarks["end_to_end"]
            if "error" not in e2e:
                audio_duration = audio_info["duration"]
                processing_time = e2e["total_time"]["mean"]
                speed_ratio = audio_duration / processing_time
                
                output.append(f"Real-time Factor: {speed_ratio:.1f}x")
                if speed_ratio >= 1.0:
                    output.append("✅ Faster than real-time processing")
                else:
                    output.append("⚠️  Slower than real-time processing")
                
                output.append(f"Processing Speed: {processing_time/audio_duration:.2f}s per second of audio")
        
        output.append("")
        output.append(f"Total Benchmark Time: {results['benchmark_duration']:.1f}s")
        output.append("")
        output.append("=" * 60)
        
        return "\n".join(output)


def main():
    """Main benchmark entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Echo Performance Benchmark")
    parser.add_argument("audio_file", help="Path to audio file for benchmarking")
    parser.add_argument("--output", "-o", help="Output file for results (JSON)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    benchmark = EchoBenchmark()
    
    try:
        results = benchmark.run_full_benchmark(args.audio_file)
        
        # Format and display results
        formatted_results = benchmark.format_results(results)
        print(formatted_results)
        
        # Save to file if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\n📄 Detailed results saved to: {args.output}")
    
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()