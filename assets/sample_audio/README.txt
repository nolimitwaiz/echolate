# Echo Sample Audio Files

This directory contains sample audio files for testing and demonstration.

## Usage

You can use these files to test Echo's functionality:

### Command Line Interface (CLI)
```bash
# Basic analysis
python -m cli.echo_cli assets/sample_audio/demo.wav

# With full report
python -m cli.echo_cli assets/sample_audio/demo.wav --report --verbose

# JSON output
python -m cli.echo_cli assets/sample_audio/demo.wav --format json --output results.json
```

### Web Interface
1. Start the web interface:
   ```bash
   python webui/app.py
   ```
2. Open your browser to http://localhost:7860
3. Upload one of these sample files or use the "Try Demo" button

### API Interface
1. Start the API server:
   ```bash
   python -m app.main
   ```
2. Use the API endpoints at http://localhost:8000

## File Formats Supported

Echo supports the following audio formats:
- WAV (recommended for best quality)
- MP3
- M4A
- FLAC
- WEBM

## Adding Your Own Files

To test Echo with your own audio:

1. Record or save audio files in this directory
2. Keep files under 90 seconds for best performance
3. Use clear speech with minimal background noise
4. Speak at a normal conversational pace

## Sample File Guidelines

Good sample files should have:
- Clear speech (SNR > 20 dB)
- 10-60 seconds duration
- Single speaker
- Natural speaking pace (120-180 WPM)
- Minimal background noise
- Standard pronunciation

## Demo Content Suggestions

For testing different Echo features, try recording:

### Pacing Practice
- Read a paragraph at different speeds
- Practice counting from 1-20 at various paces

### Clarity Practice  
- Read tongue twisters
- Practice difficult words
- Speak technical terms clearly

### Phoneme Practice
- TH sounds: "The thirty thieves thought they thwarted the theater"
- R/L sounds: "Red leather, yellow leather"
- W/V sounds: "Very well, we value vivid wolves"

### Filler Reduction
- Practice presentations with and without "um", "uh", "like"
- Record impromptu speaking exercises

### Prosody Practice
- Practice statements vs. questions (avoid uptalk)
- Read with varying intonation patterns
- Practice using rhetorical pauses

## Benchmarking

To benchmark Echo performance:
```bash
python scripts/benchmark.py assets/sample_audio/demo.wav
```

This will measure processing speed and component performance.