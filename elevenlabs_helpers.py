import os
import requests

def list_elevenlabs_voices():
    """List all available voices in ElevenLabs."""
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise EnvironmentError("ELEVENLABS_API_KEY environment variable is not set.")
    
    url = "https://api.elevenlabs.io/v1/voices"
    headers = {"xi-api-key": api_key}
    
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    
    voices = response.json().get("voices", [])
    for voice in voices:
        print(f"ID: {voice['voice_id']}, Name: {voice['name']}")
    
    return voices

def elevenlabs_tts(text: str, voice_id: str = None, output_path: str = "output.mp3"):
    """
    Generate speech from text using ElevenLabs and save it as an MP3.

    Parameters:
        text (str): The text to synthesize.
        voice_id (str, optional): Your ElevenLabs voice ID. If None, uses the first available voice.
        output_path (str, optional): Path to save the resulting MP3 (default: "output.mp3").

    Returns:
        str: The path to the saved MP3 file.
    """
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise EnvironmentError("ELEVENLABS_API_KEY environment variable is not set.")
    
    # If no voice_id is provided, get the first available voice
    if not voice_id:
        voices = list_elevenlabs_voices()
        if not voices:
            raise ValueError("No voices available in your ElevenLabs account")
        voice_id = voices[0]["voice_id"]
        print(f"Using default voice: {voices[0]['name']} (ID: {voice_id})")
    
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": api_key
    }
    
    data = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5
        }
    }
    
    response = requests.post(url, json=data, headers=headers)
    
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(f"Response: {response.text}")
        response.raise_for_status()
    
    with open(output_path, "wb") as f:
        f.write(response.content)
    
    return output_path