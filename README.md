# AudioShake Hackathon Examples

This repository contains utilities and example use-cases for the AudioShake API. For more information, check out [developer.audioshake.ai](https://developer.audioshake.ai). `audioshake_client.py` can easily be integrated into other applications.

## Dependencies

### Python Dependencies

Install Python dependencies with:

```bash
pip install -r requirements.txt
```

### System Dependencies

- **ffmpeg**: Required for audio processing
  - Install on Mac: `brew install ffmpeg`
  - Install on Ubuntu: `sudo apt install ffmpeg`
  - Install on Windows: [Download](https://ffmpeg.org/download.html) or `choco install ffmpeg`

## API Keys

An AudioShake API key is hardcoded in the examples. It will expire after the session ends. The following API keys are required for certain features:

- **OpenAI API**: Required for speech transcription
  - Set as environment variable: `OPENAI_API_KEY`
  - Get from: [OpenAI Platform](https://platform.openai.com/account/api-keys)

- **ElevenLabs API**: Required for voice synthesis
  - Set as environment variable: `ELEVENLABS_API_KEY`
  - Get from: [ElevenLabs Dashboard](https://elevenlabs.io/app/account)

## Examples

The repository contains several examples:

- **Audio Source Separation**: Split songs into vocals, drums, bass, etc.
- **Speech Recognition**: Convert audio to text using OpenAI's Whisper model
- **Voice Synthesis**: Generate speech with ElevenLabs voices

## Getting Started

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Install ffmpeg (see System Dependencies above)
4. Set up your API keys as environment variables
5. Run the example scripts in the folders

## Folder Structure

- `01_api_basics/`: Basic API usage examples
- `02_music_stems/`: Music source separation examples
- `03_copyright_compliance`: Music detection, removal, and replacement
- `04_asr_workflows/`: Speech recognition workflows
- Media helpers for audio processing