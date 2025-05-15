import sys
from pathlib import Path
from pprint import pprint

sys.path.append(str(Path(__file__).resolve().parent.parent))

from audioshake_client import AudioShakeClient
client = AudioShakeClient(token="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJjbGllbnRJZCI6ImNtYW42cG0zYjBkYjNmNWVucHVuNTN0NWIiLCJsaWNlbnNlSWQiOiJjbWFuNnFuc2Mwam5nMDFvOGhpeDFnY29yIiwiaWF0IjoxNzQ3MTgxNDc3LCJleHAiOjE5MDQ4NjE0Nzd9.qSIKGVBtlrbGniib5U4kDNacT02qyrQLvzxvko5CXew")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Basic API concepts

# (1) Upload the file and create an Asset
# The input file is called a "mixture"
asset = client.upload_file("sample.mp3")
pprint("Asset created:")
pprint(asset)

# (2) Create a Job
# For this example, we'll isolate the vocals of our test file
job = client.create_job(asset["id"], {"name": "vocals", "format": "mp3"})
pprint("Job created:")
pprint(job)

# (3) Check the Job status
# There are 3 possible statuses:
# - "created" - the job has been created (in the queue)
# - "processing" - the job is processing
# - "completed" - the job is complete and output assets are available
status = client.get_job(job["id"])
pprint("Job status:")
pprint(status)

# (4) Download output Assets if the Job is complete
if status["job"]["status"] == "completed":
    for asset in status["outputAssets"]:
        pprint(f"Downloading {asset['name']}...")
        client.download_asset(asset["link"], f"{asset['name']}.mp3")
        pprint(f"Downloaded {asset['name']}.")
        
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# AudioShake model families

# We have 3 families of models:
# - Music
# - Film / Dialogue / Television
# - Transcription & Detection

# Music Models
# - vocals (variant: high_quality)
# - instrumental (variant: high_quality)
# - drums
# - bass
# - strings
# - piano
# - guitar
# - other

# Film / Sports / Television
# - dialogue
# - music & effects
# - music removal
# - multi_voice (variants: two_speaker, n_speaker)

# Transcription & MusicDetection (json, txt, srt)
# - transcription
# - alignment
# - music detection

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# AudioShake client helper functions

# (1) Process a single job on a single file
# - metadata.name: The name of the model to use
# - metadata.format: The format of the output file (wav, flac, mp3, aiff, mp4, json, txt, srt)
# - metadata.residual: Whether to return the residual audio
# ~~
result = client.process_job(
    file_path="sample.mp3",
    metadata={"name": "vocals", "format": "mp3"},
)

# (2) We can process multiple jobs on a single file
# ~~
results = client.process_jobs(
    file_path="sample.mp3",
    metadata_list=[
        {"name": "drums", "format": "mp3"},
        {"name": "bass", "format": "mp3"},
    ]
)

# (3) Residuals
# - When a model is run with the residual option, two files are returned
# - One is the requested target (e.g. guitar), the other is everything else (residual = mixture - target)
# ~~
result = client.process_job(
    file_path="sample.mp3",
    metadata={"name": "guitar", "format": "mp3", "residual": True},
)

# (4) Variants
# - Certain models (such as vocals, instrumental and multi_voice) offer multiple variants
# ~~
result = client.process_job(
    file_path="sample.mp3",
    metadata={"name": "instrumental", "format": "mp3", "variant": "high_quality"},
)
