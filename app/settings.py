import os
import yaml
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any


class ASRConfig(BaseModel):
    default_model: str = "faster_whisper"
    faster_whisper: Dict[str, Any] = Field(default_factory=dict)
    vosk: Dict[str, Any] = Field(default_factory=dict)
    whisperx: Dict[str, Any] = Field(default_factory=dict)


class AnalysisConfig(BaseModel):
    snr_db_threshold: float = 20.0
    min_speech_duration: float = 1.0
    wpm: Dict[str, Any] = Field(default_factory=dict)
    clarity: Dict[str, float] = Field(default_factory=dict)
    prosody: Dict[str, float] = Field(default_factory=dict)
    pauses: Dict[str, float] = Field(default_factory=dict)
    phonemes: Dict[str, Any] = Field(default_factory=dict)


class UIConfig(BaseModel):
    theme: str = "default"
    high_contrast: bool = False
    speaking_modes: List[str] = Field(default_factory=list)
    default_speaking_mode: str = "Presentation"


class StorageConfig(BaseModel):
    session_history_limit: int = 10
    drill_history_limit: int = 10
    reports_dir: str = "reports"
    cache_dir: str = "cache"


class Settings(BaseModel):
    app: Dict[str, Any] = Field(default_factory=dict)
    audio: Dict[str, Any] = Field(default_factory=dict)
    vad: Dict[str, Any] = Field(default_factory=dict)
    asr: ASRConfig = Field(default_factory=ASRConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    fillers: Dict[str, List[str]] = Field(default_factory=dict)
    video: Dict[str, Any] = Field(default_factory=dict)
    ui: UIConfig = Field(default_factory=UIConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    logging: Dict[str, str] = Field(default_factory=dict)


def load_config(config_path: Optional[str] = None) -> Settings:
    """Load configuration from YAML file and environment variables."""
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config.yaml"
    
    # Load from YAML
    config_data = {}
    if Path(config_path).exists():
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
    
    # Override with environment variables
    env_overrides = {
        'app': {
            'debug': os.getenv('ECHO_DEBUG', '').lower() == 'true',
            'host': os.getenv('ECHO_HOST', '0.0.0.0'),
            'port': int(os.getenv('ECHO_PORT', '7860')),
        },
        'asr': {
            'whisperx': {
                'enabled': os.getenv('WHISPERX_ENABLED', '').lower() == 'true'
            },
            'faster_whisper': {
                'model_size': os.getenv('FASTER_WHISPER_MODEL', 'small.en'),
                'device': os.getenv('FASTER_WHISPER_DEVICE', 'cpu'),
                'compute_type': os.getenv('FASTER_WHISPER_COMPUTE_TYPE', 'int8'),
            },
            'vosk': {
                'model_path': os.getenv('VOSK_MODEL_PATH', 'models/vosk-model-small-en-us-0.15')
            }
        },
        'analysis': {
            'snr_db_threshold': float(os.getenv('SNR_DB_THRESHOLD', '20.0')),
            'prosody': {
                'uptalk_threshold': float(os.getenv('UPTALK_THRESHOLD', '0.18'))
            }
        },
        'video': {
            'enabled': os.getenv('VIDEO_ENABLED', '').lower() == 'true'
        },
        'ui': {
            'theme': os.getenv('UI_THEME', 'default'),
            'high_contrast': os.getenv('HIGH_CONTRAST', '').lower() == 'true',
            'default_speaking_mode': os.getenv('DEFAULT_SPEAKING_MODE', 'Presentation')
        },
        'storage': {
            'reports_dir': os.getenv('REPORTS_DIR', 'reports'),
            'cache_dir': os.getenv('CACHE_DIR', 'cache'),
            'session_history_limit': int(os.getenv('SESSION_HISTORY_LIMIT', '10')),
            'drill_history_limit': int(os.getenv('DRILL_HISTORY_LIMIT', '10'))
        }
    }
    
    # Deep merge config_data with env_overrides
    def deep_merge(base: dict, override: dict) -> dict:
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                deep_merge(base[key], value)
            else:
                base[key] = value
        return base
    
    merged_config = deep_merge(config_data, env_overrides)
    
    return Settings(**merged_config)


# Global settings instance
settings = load_config()