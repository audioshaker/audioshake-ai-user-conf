import sys
from pathlib import Path
from pprint import pprint
sys.path.append(str(Path(__file__).resolve().parent.parent))

from audioshake_client import AudioShakeClient

client = AudioShakeClient(token="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJjbGllbnRJZCI6ImNtOWtqdzNkcTA4bGI4bmJ2YXQ2ZWc2OTkiLCJsaWNlbnNlSWQiOiJjbTlrandodmUwYTcyMDFwcDRyczFmdmhxIiwiaWF0IjoxNzQ0ODQ1NDAzLCJleHAiOjE5MDI1MjU0MDN9.0zR47M607sTvyZfvg47TCOyJA1ILv4q8sSVbithBvUE")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Using AudioShake to isolate dialogue 

# PROBLEM
# I recorded myself in a noisey environment and want to isolate my voice

# SOLUTION
# Use the source separation model to isolate my voice

# Isolate dialogue from a noisy environment
# ~~
client.process_job(
    file_path="noisy_speech.mp3",
    metadata={"name": "dialogue", "format": "mp3"}
)

# That sounds pretty good, but it's a little artificial sounding.
# What if we mix in some of the background noise?
# ~~
# client.process_job(
#     file_path="noisy_speech.mp3",
#     metadata={"name": "dialogue", "format": "mp3", "residual": True}
# )

# That sounds good, but it's a little unnatural sounding.
# What if we mix in some of the background noise?
# ~~
# from media_helpers import adjust_volume
# adjust_volume("dialogue_residual.mp3", "dialogue_residual_quieter.mp3", 0.5)

# from media_helpers import sum_audio
# sum_audio(["dialogue_residual_quieter.mp3", "dialogue.mp3"], "final_mix.mp3")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Using AudioShake to automatically transcribe and translate dialogue

# PROBLEM
# I have a podcast that I generated with multiple speakers but they are combined in the same audio file.
# A lot of my audience likes to listen to the podcast in Sindarin, the ancient language of the elves.
# However, I only have the original audio file with all the speakers mixed together.

# SOLUTION
# Isolate each speaker
# Transcribe each speaker
# Translate to Sindarin
# Re-generate the audio
# Combine the results

# First, we can separate out each speaker into an individual audio file
# ~~
# client.process_job(
#     file_path="ai_podcast.mp3",
#     metadata={"name": "multi_voice", "format": "mp3", "variant": "two_speaker"}
# )

# Use openai to transcribe and translate to Sindarin
# ~~
# from openai_helpers import transcribe_audio, translate_to_sindarin

# Transcribe each speaker
# ~~
# speaker_1 = transcribe_audio("voice_two_speaker_01.mp3")
# speaker_2 = transcribe_audio("voice_two_speaker_02.mp3")

# Translate to Sindarin
# ~~
# speaker_1_sindarin = translate_to_sindarin(speaker_1)
# speaker_2_sindarin = translate_to_sindarin(speaker_2)

# Use elevenlabs to generate the speech from text (tts)
# ~~
# from elevenlabs_helpers import elevenlabs_tts

# elevenlabs_tts(speaker_1_sindarin, voice_id="IKne3meq5aSn9XLyUdCD", output_path="sindarin_speaker_01.mp3")
# elevenlabs_tts(speaker_2_sindarin, voice_id="cgSgspJ2msm6clMCkdW9", output_path="sindarin_speaker_02.mp3")

# NEXT STEPS
# - Use the alignment data and use TTS for each segment and then combine the results into a single audio file
