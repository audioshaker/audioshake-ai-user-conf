import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from audioshake_client import AudioShakeClient
from pedalboard import Pedalboard, Distortion, Reverb, Gain, Compressor

client = AudioShakeClient(token="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJjbGllbnRJZCI6ImNtYW42cG0zYjBkYjNmNWVucHVuNTN0NWIiLCJsaWNlbnNlSWQiOiJjbWFuNnFuc2Mwam5nMDFvOGhpeDFnY29yIiwiaWF0IjoxNzQ3MTgxNDc3LCJleHAiOjE5MDQ4NjE0Nzd9.qSIKGVBtlrbGniib5U4kDNacT02qyrQLvzxvko5CXew")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Using AudioShake to make music more editable

# PROBLEM
# I have an AI generated track and I am unable to edit it
# It has vocals that I don't want, and I don't have access to the stems

# SOLUTION
# We can use AudioShake to separate the track into stems
# We can then use the stems to remix the track programatically

# We'll be using an AI generated track. A common problem that many generative music models face is that they generate all stems at once.
# Using a source separation, we can isolate each instrument and remix our track programatically.

# Get just the instrumentals (removes the vocals)
# client.process_job(
#     file_path="ai_conf.mp3",
#     metadata={"name": "instrumental", "format": "mp3"}
# )

# That's nice, but what about bass and drums? 
client.process_jobs(
    file_path="ai_conf.mp3",
    metadata_list=[
        {"name": "bass", "format": "mp3",},
        {"name": "drums", "format": "mp3"},
    ],
    sum_output=True # Convenience function to sum the output of the jobs
)

# What if I want the bass to be beefier? 
# In this example we'll use Pedalboard, a really powerful library for audio effects, to apply effects to the bass and drum tracks
client.process_jobs(
    file_path="ai_conf.mp3",
    metadata_list=[
        {"name": "bass", "format": "mp3",},
        {"name": "drums", "format": "mp3"},
        {"name": "vocals", "format": "mp3"},
    ],
    post_process=[ # Convenience function to apply a pedalboard to the output of the job
        {"name": "bass", "board": Pedalboard([Distortion(drive_db=20), Gain(gain_db=-8), Compressor(threshold_db=-24, ratio=4, attack_ms=20, release_ms=100)])},
        {"name": "drums", "board": Pedalboard([Gain(gain_db=3), Reverb(room_size=0.1, damping=0.9), Compressor(threshold_db=-20, ratio=10, attack_ms=10, release_ms=200)])}
    ],
    sum_output=True
)

# NEXT STEPS (Ideas)
# - Use a voice cloner to change the vocals?
# - Use llm / pydantic models to make a prompt based music remixing? 
# - Use infilling models in addition to the generative models to make more complex remixes?
# - Convert individual stems to MIDI (score) and create a music "language" model? 