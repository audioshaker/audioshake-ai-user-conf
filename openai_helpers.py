import os
from openai import OpenAI

def transcribe_audio(file_path, include_timestamps=False):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")
    
    client = OpenAI(api_key=api_key)
    with open(file_path, "rb") as audio_file:
        if include_timestamps:
            transcript_data = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="verbose_json"
            )
            # Convert to regular dict for JSON serialization
            result = {
                "text": transcript_data.text,
                "segments": [
                    {
                        "id": s.id,
                        "start": s.start,
                        "end": s.end,
                        "text": s.text
                    }
                    for s in transcript_data.segments
                ]
            }
            return result
        else:
            transcript_data = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
            return transcript_data.text

def translate_to_sindarin(text):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")
    
    client = OpenAI(api_key=api_key)
    prompt = f"translate this to Sindarin: {text}"
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        timeout=600  # Setting timeout to 600 seconds
    )
    return response.choices[0].message.content.strip()