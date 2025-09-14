#!/usr/bin/env python3
"""
Setup script to download and cache required models for Echo.
This ensures the application can run offline after initial setup.
"""

import os
import sys
import logging
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Optional
import hashlib

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from app.settings import settings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelDownloader:
    """Download and setup models for Echo."""
    
    def __init__(self):
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        self.cache_dir = Path(settings.storage.cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Model configurations
        self.models = {
            "vosk-small-en": {
                "url": "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip",
                "filename": "vosk-model-small-en-us-0.15.zip",
                "extract_dir": "vosk-model-small-en-us-0.15",
                "size": "40MB",
                "description": "Small English ASR model for voice recognition fallback"
            }
        }
    
    def download_file(self, url: str, filename: str, description: str = "") -> bool:
        """Download a file with progress indication."""
        file_path = self.models_dir / filename
        
        if file_path.exists():
            logger.info(f"✅ {filename} already exists, skipping download")
            return True
        
        logger.info(f"📥 Downloading {description or filename}...")
        logger.info(f"    URL: {url}")
        
        try:
            def progress_hook(block_num, block_size, total_size):
                if total_size > 0:
                    percent = min(100, (block_num * block_size * 100) // total_size)
                    sys.stdout.write(f"\r    Progress: {percent}% ")
                    sys.stdout.flush()
            
            urllib.request.urlretrieve(url, file_path, progress_hook)
            print()  # New line after progress
            
            logger.info(f"✅ Downloaded: {filename}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to download {filename}: {e}")
            if file_path.exists():
                file_path.unlink()  # Clean up partial download
            return False
    
    def extract_archive(self, filename: str, extract_dir: str) -> bool:
        """Extract a zip archive."""
        archive_path = self.models_dir / filename
        extract_path = self.models_dir / extract_dir
        
        if extract_path.exists():
            logger.info(f"✅ {extract_dir} already extracted, skipping")
            return True
        
        if not archive_path.exists():
            logger.error(f"❌ Archive not found: {filename}")
            return False
        
        logger.info(f"📦 Extracting {filename}...")
        
        try:
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(self.models_dir)
            
            logger.info(f"✅ Extracted: {extract_dir}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to extract {filename}: {e}")
            return False
    
    def verify_model(self, model_name: str) -> bool:
        """Verify that a model is properly installed."""
        if model_name == "vosk-small-en":
            model_path = self.models_dir / "vosk-model-small-en-us-0.15"
            required_files = ["am/final.mdl", "graph/HCLG.fst", "words.txt"]
            
            for file_path in required_files:
                if not (model_path / file_path).exists():
                    logger.error(f"❌ Missing required file: {file_path}")
                    return False
            
            return True
        
        return False
    
    def setup_faster_whisper(self) -> bool:
        """Setup Faster-Whisper (downloads on first use)."""
        logger.info("🎤 Setting up Faster-Whisper...")
        
        try:
            # Import and test Faster-Whisper
            from faster_whisper import WhisperModel
            
            # This will download the model on first use
            model_size = settings.asr.faster_whisper.get("model_size", "small.en")
            logger.info(f"    Model size: {model_size}")
            logger.info("    ✅ Faster-Whisper will download model on first use")
            
            return True
            
        except ImportError:
            logger.error("❌ faster-whisper not available. Install with: pip install faster-whisper")
            return False
        except Exception as e:
            logger.error(f"❌ Faster-Whisper setup failed: {e}")
            return False
    
    def setup_g2p(self) -> bool:
        """Setup G2P for phoneme analysis."""
        logger.info("🔤 Setting up G2P (Grapheme-to-Phoneme)...")
        
        try:
            from g2p_en import G2p
            
            # Test initialization (downloads models if needed)
            g2p = G2p()
            test_result = g2p("hello")
            
            if test_result:
                logger.info("    ✅ G2P setup successful")
                return True
            else:
                logger.error("❌ G2P test failed")
                return False
                
        except ImportError:
            logger.error("❌ g2p_en not available. Install with: pip install g2p_en")
            return False
        except Exception as e:
            logger.error(f"❌ G2P setup failed: {e}")
            return False
    
    def setup_all_models(self) -> bool:
        """Setup all required models."""
        logger.info("🚀 Setting up Echo models and dependencies...")
        logger.info("=" * 60)
        
        success_count = 0
        total_count = 0
        
        # 1. Setup Faster-Whisper
        total_count += 1
        if self.setup_faster_whisper():
            success_count += 1
        
        # 2. Setup G2P
        total_count += 1
        if self.setup_g2p():
            success_count += 1
        
        # 3. Download Vosk model (fallback ASR)
        total_count += 1
        vosk_config = self.models["vosk-small-en"]
        
        if self.download_file(
            vosk_config["url"],
            vosk_config["filename"], 
            f"{vosk_config['description']} ({vosk_config['size']})"
        ):
            if self.extract_archive(vosk_config["filename"], vosk_config["extract_dir"]):
                if self.verify_model("vosk-small-en"):
                    logger.info("✅ Vosk model verified successfully")
                    success_count += 1
                else:
                    logger.error("❌ Vosk model verification failed")
            else:
                logger.error("❌ Vosk model extraction failed")
        else:
            logger.error("❌ Vosk model download failed")
        
        # 4. Create demo assets
        total_count += 1
        if self.create_demo_assets():
            success_count += 1
        
        # Summary
        logger.info("=" * 60)
        logger.info(f"📊 Setup Summary: {success_count}/{total_count} components successful")
        
        if success_count == total_count:
            logger.info("🎉 All models and dependencies setup successfully!")
            logger.info("Echo is ready to use offline.")
            return True
        else:
            logger.warning("⚠️  Some components failed to setup.")
            logger.info("Echo may work with reduced functionality.")
            return False
    
    def create_demo_assets(self) -> bool:
        """Create demo assets for testing."""
        logger.info("🎪 Creating demo assets...")
        
        try:
            # Create assets directory
            assets_dir = Path("assets")
            assets_dir.mkdir(exist_ok=True)
            
            sample_audio_dir = assets_dir / "sample_audio"
            sample_audio_dir.mkdir(exist_ok=True)
            
            # Create README for sample audio
            readme_content = """# Sample Audio Files

This directory contains sample audio files for testing Echo.

## Files:
- demo.wav: A sample speech recording for demonstration
- README.txt: This file

## Usage:
You can use these files to test Echo functionality:

```bash
# CLI test
python -m cli.echo_cli assets/sample_audio/demo.wav

# Web interface
python webui/app.py
```

To add your own sample files, place WAV, MP3, or other supported audio files here.
"""
            
            readme_path = sample_audio_dir / "README.txt"
            with open(readme_path, 'w') as f:
                f.write(readme_content)
            
            logger.info("    ✅ Demo assets created")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create demo assets: {e}")
            return False
    
    def check_dependencies(self) -> Dict[str, bool]:
        """Check if all required dependencies are installed."""
        logger.info("🔍 Checking dependencies...")
        
        dependencies = {
            "gradio": False,
            "fastapi": False,
            "librosa": False,
            "soundfile": False,
            "webrtcvad": False,
            "faster_whisper": False,
            "vosk": False,
            "g2p_en": False,
            "plotly": False,
            "reportlab": False
        }
        
        for dep in dependencies:
            try:
                __import__(dep)
                dependencies[dep] = True
                logger.info(f"    ✅ {dep}")
            except ImportError:
                logger.warning(f"    ❌ {dep} - run: pip install {dep}")
        
        return dependencies
    
    def cleanup_downloads(self) -> None:
        """Clean up downloaded archives after extraction."""
        logger.info("🧹 Cleaning up downloaded archives...")
        
        for model_config in self.models.values():
            archive_path = self.models_dir / model_config["filename"]
            if archive_path.exists():
                try:
                    archive_path.unlink()
                    logger.info(f"    🗑️  Removed: {model_config['filename']}")
                except Exception as e:
                    logger.warning(f"    ⚠️  Failed to remove {model_config['filename']}: {e}")


def main():
    """Main setup script entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Echo Model Setup Script")
    parser.add_argument("--check-deps", action="store_true", help="Only check dependencies")
    parser.add_argument("--cleanup", action="store_true", help="Clean up downloaded archives")
    parser.add_argument("--force", action="store_true", help="Force re-download of existing models")
    
    args = parser.parse_args()
    
    downloader = ModelDownloader()
    
    if args.check_deps:
        deps = downloader.check_dependencies()
        missing = [dep for dep, available in deps.items() if not available]
        
        if missing:
            logger.error(f"❌ Missing dependencies: {', '.join(missing)}")
            logger.info("Install missing dependencies with: pip install -r requirements.txt")
            sys.exit(1)
        else:
            logger.info("✅ All dependencies are installed")
            sys.exit(0)
    
    if args.cleanup:
        downloader.cleanup_downloads()
        sys.exit(0)
    
    if args.force:
        # Remove existing models to force re-download
        import shutil
        if downloader.models_dir.exists():
            shutil.rmtree(downloader.models_dir)
            logger.info("🗑️  Removed existing models directory for fresh setup")
    
    # Run full setup
    success = downloader.setup_all_models()
    
    if success:
        logger.info("\n🎉 Setup complete! You can now run Echo:")
        logger.info("    Web UI:  python webui/app.py")
        logger.info("    API:     python -m app.main")
        logger.info("    CLI:     python -m cli.echo_cli sample.wav")
        sys.exit(0)
    else:
        logger.error("\n❌ Setup encountered errors. Check logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()