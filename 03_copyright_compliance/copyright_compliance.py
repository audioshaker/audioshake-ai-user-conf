import sys
from pathlib import Path
from pprint import pprint
sys.path.append(str(Path(__file__).resolve().parent.parent))

from audioshake_client import AudioShakeClient

client = AudioShakeClient(token="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJjbGllbnRJZCI6ImNtOWtqdzNkcTA4bGI4bmJ2YXQ2ZWc2OTkiLCJsaWNlbnNlSWQiOiJjbTlrandodmUwYTcyMDFwcDRyczFmdmhxIiwiaWF0IjoxNzQ0ODQ1NDAzLCJleHAiOjE5MDI1MjU0MDN9.0zR47M607sTvyZfvg47TCOyJA1ILv4q8sSVbithBvUE")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Using AudioShake to assist in copyright compliance

# PROBLEM
# I am a streaming gamer and I want to share a clip of my stream to social media.
# One activity that drives engagement is when my viewers get to choose the music in the stream.
# However, I can't repost this clip because it contains copyrighted music and I need to remove it.

# SOLUTION
# Detect the music in the video and remove it and replace it.

# Detect Music in a video
client.process_job(
    file_path="fortnite_stream.mp4",
    metadata={"name": "music_detection", "format": "json"}
)

# Remove Music from a video
# ~~
client.process_job(
    file_path="fortnite_stream.mp4",
    metadata={"name": "music_removal", "format": "mp3", "residual": True}
)

# Replace Music in a video
# ~~
from media_helpers import replace_audio_in_video, sum_audio

sum_audio(["music_removal.mp3", "instrumental.mp3"], "new_background.mp3")
replace_audio_in_video("fortnite_stream.mp4", "new_background.mp3", "fortnite_stream_no_music.mp4")

# NEXT LEVEL
# Generate embeddings for the music residual and use them to search for similar music in our library
# Warning: This might take a while to download the model!
# ~~
# from media_helpers import find_similar_audio, replace_audio_in_video, sum_audio, match_volume
# path_str = find_similar_audio("music_removal_residual.mp3", "music_library")
# pprint(path_str)
# Get the most similar music track
# ~~
# similar_music_path = path_str[0][0]

# Match volume of replacement music to the original music residual
# ~~
# volume_factor = match_volume(similar_music_path, "music_removal_residual.mp3", "volume_adjusted_music.mp3")
# print(f"Applied volume factor: {volume_factor:.2f}")

# Replace the music in the video with the new music
# ~~
# sum_audio(["music_removal.mp3", "volume_adjusted_music.mp3"], "new_background.mp3")
# replace_audio_in_video("fortnite_stream.mp4", "new_background.mp3", "fortnite_stream_no_music.mp4")

# NEXT STEPS (Ideas)
# - Integrate ACRCloud API to attribute the music in the video and check for copyright permissions
# - Use music labelling models to collect data on the extracted music which can be used to generate prompts for a generative music model