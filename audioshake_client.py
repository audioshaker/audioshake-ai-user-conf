import os
import time
import requests
import subprocess
import concurrent.futures
from typing import List, Dict

# Pedalboard imports for post-processing
# Make sure to: pip install pedalboard soundfile numpy
import numpy as np
import soundfile as sf
from pedalboard import Pedalboard
from pedalboard.io import AudioFile

class AudioShakeClient:
    def __init__(self, token: str, base_url: str = "https://groovy.audioshake.ai"):
        self.token = token
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/json"
        }

    def upload_file(self, file_path: str) -> dict:
        """Uploads an audio file and returns the asset info."""
        url = f"{self.base_url}/upload/"
        with open(file_path, "rb") as f:
            files = {"file": f}
            resp = requests.post(url, headers=self.headers, files=files)
        resp.raise_for_status()
        return resp.json()

    def create_job(self, asset_id: str, metadata: dict, callback_url: str = None) -> dict:
        """
        Creates a job with the given asset_id and metadata.
        Example metadata: {"name": "vocals", "format": "wav", "variant": "...", "residual": true}
        """
        url = f"{self.base_url}/job/"
        payload = {"assetId": asset_id, "metadata": metadata}
        if callback_url:
            payload["callbackUrl"] = callback_url
        resp = requests.post(url, headers={**self.headers, "Content-Type": "application/json"}, json=payload)
        resp.raise_for_status()
        return resp.json()['job']

    def get_job(self, job_id: str) -> dict:
        """Retrieves the status and details of a job."""
        url = f"{self.base_url}/job/{job_id}"
        resp = requests.get(url, headers=self.headers)
        resp.raise_for_status()
        return resp.json()

    def download_asset(self, link: str, destination_path: str) -> None:
        """Downloads an asset to a local file."""
        resp = requests.get(link, stream=True)
        resp.raise_for_status()
        with open(destination_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

    def process_job(self, file_path: str, metadata: dict, callback_url: str = None,
                    poll_interval: int = 5, timeout: int = 600) -> dict:
        """
        Single-job workflow: upload a file, create a job, poll for completion,
        and download output assets.

        :param file_path: Local path to audio file
        :param metadata: e.g. {"name": "vocals", "format": "wav", "variant": "...", "residual": true}
        :param callback_url: Optional callback URL
        :param poll_interval: How often to poll for job status (seconds)
        :param timeout: Max time in seconds to wait for job completion

        :return: Final job info dict
        """
        # 1. Upload file
        asset = self.upload_file(file_path)

        # 2. Create job
        job = self.create_job(asset_id=asset["id"], metadata=metadata, callback_url=callback_url)
        job_id = job["id"]
        start_time = time.time()

        # 3. Poll job status
        while True:
            current_status = self.get_job(job_id)
            job_info = current_status["job"]
            status = job_info["status"]

            if status == "completed":
                # 4. Download output assets
                output_assets = job_info.get("outputAssets", [])
                for out_asset in output_assets:
                    link = out_asset.get("link")
                    if link:
                        filename = out_asset.get("name", f"{job_id}.wav")
                        self.download_asset(link, filename)
                return job_info  # Return final job info

            elif status in ("failed", "error"):
                raise RuntimeError(f"Job {job_id} failed with status: {status}")

            if time.time() - start_time > timeout:
                raise TimeoutError(f"Job {job_id} did not complete within {timeout} seconds.")

            time.sleep(poll_interval)

    def _apply_pedalboard_to_file(self, filename: str, board: Pedalboard) -> None:
        """
        Applies a given Pedalboard to an audio file in-place.

        :param filename: The path to the audio file to be processed.
        :param board: A Pedalboard instance (e.g. Pedalboard([Distortion(drive_db=20)])).
        """
        with AudioFile(filename) as f:
            audio_data = f.read(f.frames)
            sample_rate = f.samplerate
            # Pedalboard expects shape == (channels, samples).
            # AudioFile returns shape (channels, samples).
            processed = board(audio_data, sample_rate)

        # Now write the processed audio back, but soundfile expects (samples, channels).
        sf.write(filename, processed.transpose(), sample_rate)
        print(f"Applied pedalboard to {filename}")

    def _process_single_job_no_upload(
        self,
        asset_id: str,
        metadata: dict,
        callback_url: str,
        poll_interval: int,
        timeout: int,
        post_map: Dict[str, Pedalboard]
    ) -> dict:
        """
        Internal helper for creating, polling, downloading a job, and optionally
        applying a pedalboard when the asset is already uploaded.
        """
        job = self.create_job(asset_id=asset_id, metadata=metadata, callback_url=callback_url)
        job_id = job["id"]
        start_time = time.time()

        while True:
            response = self.get_job(job_id)
            job_info = response["job"]
            status = job_info["status"]

            if status == "completed":
                output_assets = job_info.get("outputAssets", [])
                # Download each asset
                for out_asset in output_assets:
                    link = out_asset.get("link")
                    if link:
                        filename = out_asset.get("name", f"{job_id}.wav")
                        self.download_asset(link, filename)

                        # If there's a post_process board for this job name, apply it
                        job_name = job_info["metadata"].get("name")
                        if job_name and job_name in post_map:
                            self._apply_pedalboard_to_file(filename, post_map[job_name])

                return job_info

            if status in ("failed", "error"):
                raise RuntimeError(f"Job {job_id} failed with status: {status}")

            if time.time() - start_time > timeout:
                raise TimeoutError(f"Job {job_id} did not complete within {timeout} seconds.")

            time.sleep(poll_interval)

    def _pedalboard_sum_audio(self, file_paths: List[str], output_file: str):
        """
        Sum (overlay) multiple audio files using pedalboard+NumPy.

        :param file_paths: A list of local audio paths (e.g. ["bass.mp3", "drums.mp3"]).
        :param output_file: Name for the combined output file (e.g. "summed_output.wav").
        """
        combined_data = None
        sample_rate = None

        for path in file_paths:
            with AudioFile(path) as f:
                audio_data = f.read(f.frames)
                sr = f.samplerate

            # Initialize combined_data if first file
            if combined_data is None:
                combined_data = audio_data
                sample_rate = sr
            else:
                # Ensure sample rates match
                if sr != sample_rate:
                    raise ValueError(f"Mismatched sample rates: {sr} vs {sample_rate}")

                # If lengths differ, zero-pad the shorter array
                channels_comb, len_comb = combined_data.shape
                channels_new, len_new = audio_data.shape

                if channels_comb != channels_new:
                    raise ValueError("All stems must have the same channel count to sum them.")

                # Pad if needed
                if len_new > len_comb:
                    combined_data = np.pad(
                        combined_data, ((0,0),(0, len_new - len_comb)),
                        mode='constant', constant_values=0
                    )
                elif len_comb > len_new:
                    audio_data = np.pad(
                        audio_data, ((0,0),(0, len_comb - len_new)),
                        mode='constant', constant_values=0
                    )

                # Sum in-place
                combined_data += audio_data

        if combined_data is not None:
            sf.write(output_file, combined_data.transpose(), sample_rate)
            print(f"Pedalboard sum saved to {output_file}")
        else:
            print("No audio data to sum.")

    def process_jobs(
        self,
        file_path: str,
        metadata_list: List[dict],
        callback_url: str = None,
        poll_interval: int = 5,
        timeout: int = 600,
        post_process: List[dict] = None,
        sum_output: bool = False,
        sum_filename: str = "sum_output.wav"
    ) -> List[dict]:
        """
        Multi-job workflow for the same file. Uploads once, then creates and processes
        multiple jobs (concurrently) based on the provided metadata list. Optionally:
            - Applies per-job pedalboards via `post_process`.
            - Sums (overlays) all post-processed stems into one file using pedalboard.

        :param file_path: Local path to the audio file
        :param metadata_list: A list of metadata dicts (e.g. [{"name": "bass"}, {"name": "drums"}])
        :param callback_url: Optional callback URL for each job
        :param poll_interval: Seconds between job status checks
        :param timeout: Max time in seconds to wait for each job
        :param post_process: A list of dicts specifying job name + Pedalboard instance
                            (e.g. [{"name": "bass", "board": Pedalboard([...])}])
        :param sum_output: If True, combine all downloaded stems into a single output
        :param sum_filename: Output filename for the summed track
        :return: A list of completed job info dicts
        """
        # 1. Build a quick lookup from "name" -> Pedalboard
        post_map = {}
        if post_process:
            for item in post_process:
                job_name = item.get("name")
                board = item.get("board")
                if job_name and board:
                    post_map[job_name] = board

        # 2. Upload file once
        asset = self.upload_file(file_path)
        asset_id = asset["id"]

        # 3. Concurrently process each job
        results = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    self._process_single_job_no_upload,
                    asset_id,
                    meta,
                    callback_url,
                    poll_interval,
                    timeout,
                    post_map
                )
                for meta in metadata_list
            ]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())

        # 4. Optionally sum all post-processed files
        if sum_output:
            downloaded_files = []
            for job_info in results:
                output_assets = job_info.get("outputAssets", [])
                for out_asset in output_assets:
                    filename = out_asset.get("name", f"{job_info['id']}.wav")
                    if os.path.exists(filename):
                        downloaded_files.append(filename)

            if len(downloaded_files) > 1:
                self._pedalboard_sum_audio(downloaded_files, sum_filename)
            else:
                print("No multiple stems to sum or only one file found.")

        return results