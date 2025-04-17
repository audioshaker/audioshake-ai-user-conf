import subprocess
import os
import numpy as np
import tempfile
import torch
import torchaudio
import torchaudio.transforms as T

# Global variable to hold the model when it's loaded
audio_model = None

def get_audio_model():
    """Lazy-load the audio model only when it's needed"""
    global audio_model
    if audio_model is None:
        try:
            from panns_inference import AudioTagging
            audio_model = AudioTagging(checkpoint_path=None)
            return audio_model
        except Exception as e:
            print(f"Warning: Could not load PANNS model. Falling back to basic features. Error: {e}")
            return None
    return audio_model

def replace_audio_in_video(video_path: str, audio_path: str, output_path: str) -> None:
    """
    Replace the audio track of a video file with a new audio file using ffmpeg.

    Parameters:
        video_path (str): Path to the input video file (e.g., an mp4).
        audio_path (str): Path to the new audio file (e.g., an mp3).
        output_path (str): Path for the output video with replaced audio.
    
    Raises:
        RuntimeError: If ffmpeg encounters an error during processing.
    """
    command = [
        "ffmpeg",
        "-y",  # Overwrite output file if it exists.
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy",   # Copy the video stream without re-encoding.
        "-map", "0:v:0",  # Select the video stream from the first input.
        "-map", "1:a:0",  # Select the audio stream from the second input.
        "-shortest",      # Truncate the output to the shortest input's length.
        output_path
    ]

    process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if process.returncode != 0:
        raise RuntimeError(f"ffmpeg error: {process.stderr}")

def sum_audio(input_paths: list[str], output_path: str) -> None:
    """
    Combine multiple audio files into a single MP3 output file.

    Parameters:
        input_paths (list[str]): List of paths to input audio files (can be various formats).
        output_path (str): Path for the output MP3 file.
    
    Raises:
        RuntimeError: If ffmpeg encounters an error during processing.
        ValueError: If no input files are provided.
    """
    if not input_paths:
        raise ValueError("At least one input audio file must be provided")

    command = ["ffmpeg", "-y"]  # -y to overwrite output if exists
    
    # Add all input files
    for path in input_paths:
        command.extend(["-i", path])
    
    # Use the amix filter to mix all inputs together
    filter_complex = f"amix=inputs={len(input_paths)}:duration=longest"
    
    command.extend([
        "-filter_complex", filter_complex,
        "-c:a", "libmp3lame",  # Use MP3 codec
        "-q:a", "2",          # High quality MP3 (0-9, 0 is highest)
        output_path
    ])

    process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if process.returncode != 0:
        raise RuntimeError(f"ffmpeg error: {process.stderr}")

def convert_to_wav(input_path: str) -> str:
    """
    Convert an audio file to WAV format using ffmpeg.
    Returns the path to the temporary WAV file.
    """
    # Create a temporary file with .wav extension
    temp_fd, temp_path = tempfile.mkstemp(suffix='.wav')
    os.close(temp_fd)  # Close the file descriptor
    
    command = [
        "ffmpeg",
        "-y",  # Overwrite output file if it exists
        "-i", input_path,
        "-acodec", "pcm_s16le",  # Use PCM 16-bit encoding
        "-ar", "44100",  # Set sample rate to 44.1kHz
        "-ac", "1",      # Convert to mono
        temp_path
    ]
    
    process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if process.returncode != 0:
        os.unlink(temp_path)  # Clean up the temp file
        raise RuntimeError(f"ffmpeg error: {process.stderr}")
    
    return temp_path

def extract_features(file_path: str, n_mfcc: int = 20) -> np.ndarray:
    """
    Extract audio features using mel spectrogram and MFCCs.
    
    Parameters:
        file_path (str): Path to the audio file.
        n_mfcc (int): Number of MFCC coefficients to extract.
        
    Returns:
        np.ndarray: Audio feature vector combining multiple audio characteristics.
    """
    # Convert to WAV if needed
    temp_wav = None
    try:
        if not file_path.lower().endswith('.wav'):
            temp_wav = convert_to_wav(file_path)
            file_path = temp_wav
        else:
            # Still need to ensure correct format
            temp_wav = convert_to_wav(file_path)
            file_path = temp_wav

        # Load audio file
        waveform, sr = torchaudio.load(file_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Normalize
        waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)
        
        # Create transforms
        mel_transform = T.MelSpectrogram(
            sample_rate=sr,
            n_fft=2048,
            win_length=1024,
            hop_length=512,
            n_mels=128,
            power=2.0
        )
        
        mfcc_transform = T.MFCC(
            sample_rate=sr,
            n_mfcc=n_mfcc,
            melkwargs={
                'n_fft': 2048,
                'n_mels': 128,
                'hop_length': 512,
                'mel_scale': 'htk',
            }
        )
        
        # Calculate features
        mel_spec = mel_transform(waveform)
        mfcc = mfcc_transform(waveform)
        
        # Convert to log scale
        mel_spec = torch.log(mel_spec + 1e-9)
        
        # Calculate statistics over time
        mel_mean = torch.mean(mel_spec, dim=2)
        mel_std = torch.std(mel_spec, dim=2)
        mfcc_mean = torch.mean(mfcc, dim=2)
        mfcc_std = torch.std(mfcc, dim=2)
        
        # Calculate energy and rhythm features
        energy = torch.sqrt(torch.sum(mel_spec ** 2, dim=1))
        energy_stats = torch.tensor([
            torch.mean(energy),
            torch.std(energy),
            torch.max(energy),
            torch.median(energy)
        ])
        
        # Combine all features
        features = torch.cat([
            mel_mean.flatten(),
            mel_std.flatten(),
            mfcc_mean.flatten(),
            mfcc_std.flatten(),
            energy_stats
        ])
        
        return features.numpy()
            
    finally:
        if temp_wav and os.path.exists(temp_wav):
            try:
                os.unlink(temp_wav)
            except Exception:
                pass

def find_similar_audio(input_file: str, folder_path: str, threshold: float = 0.7) -> list[tuple[str, float]]:
    """
    Compare the input audio file against all audio files in folder_path using
    a combination of mel spectrogram and MFCC features.
    Returns a list of similar files sorted by similarity score.

    Parameters:
        input_file (str): Path to the input audio file.
        folder_path (str): Folder containing audio files to compare against.
        threshold (float): Similarity threshold (0 to 1) for including matches.
        
    Returns:
        list[tuple[str, float]]: List of (file_path, similarity_score) pairs, sorted by similarity.
    """
    # Get audio model only when this function is called
    model = get_audio_model()
    
    input_features = extract_features(input_file)
    matches = []
    
    # Normalize the input features
    input_features = (input_features - np.mean(input_features)) / (np.std(input_features) + 1e-8)

    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(('.mp3', '.wav')):
            candidate = os.path.join(folder_path, file_name)
            if os.path.abspath(candidate) == os.path.abspath(input_file):
                continue  # Skip if it's the same file as input
            try:
                # Extract and normalize candidate features
                candidate_features = extract_features(candidate)
                candidate_features = (candidate_features - np.mean(candidate_features)) / (np.std(candidate_features) + 1e-8)
                
                # Calculate cosine similarity
                similarity = np.dot(input_features, candidate_features) / (
                    np.linalg.norm(input_features) * np.linalg.norm(candidate_features)
                )
                
                # Convert similarity to range [0, 1]
                similarity = (similarity + 1) / 2
                
                if similarity >= threshold:
                    matches.append((candidate, float(similarity)))
            except Exception as e:
                print(f"Failed to process {candidate}: {e}")

    # Sort by similarity score in descending order
    matches.sort(key=lambda x: x[1], reverse=True)
    return matches

def adjust_volume(input_path: str, output_path: str, volume_factor: float) -> None:
    """
    Adjust the volume of an audio file using ffmpeg.
    
    Parameters:
        input_path (str): Path to the input audio file.
        output_path (str): Path for the output audio file.
        volume_factor (float): Volume adjustment factor (1.0 = no change, 0.5 = half volume, 2.0 = double volume).
    
    Raises:
        RuntimeError: If ffmpeg encounters an error during processing.
    """
    command = [
        "ffmpeg",
        "-y",  # Overwrite output file if it exists
        "-i", input_path,
        "-filter:a", f"volume={volume_factor}",
        output_path
    ]
    
    process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if process.returncode != 0:
        raise RuntimeError(f"ffmpeg error: {process.stderr}")

def calculate_rms(file_path: str) -> float:
    """
    Calculate the Root Mean Square (RMS) amplitude of an audio file.
    
    Parameters:
        file_path (str): Path to the audio file.
    
    Returns:
        float: The RMS amplitude value.
    """
    # Convert to WAV for consistent processing
    temp_wav = None
    try:
        if not file_path.lower().endswith('.wav'):
            temp_wav = convert_to_wav(file_path)
            file_path = temp_wav
        
        # Load audio file
        waveform, sr = torchaudio.load(file_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Calculate RMS amplitude
        rms = torch.sqrt(torch.mean(waveform ** 2))
        return float(rms)
        
    finally:
        if temp_wav and os.path.exists(temp_wav):
            try:
                os.unlink(temp_wav)
            except Exception:
                pass

def match_volume(source_file: str, target_file: str, output_file: str) -> float:
    """
    Adjust the volume of the source file to match the volume of the target file.
    
    Parameters:
        source_file (str): Path to the audio file to adjust.
        target_file (str): Path to the reference audio file.
        output_file (str): Path for the volume-adjusted output file.
    
    Returns:
        float: The volume factor applied.
    
    Raises:
        RuntimeError: If ffmpeg encounters an error during processing.
    """
    # Calculate RMS values
    source_rms = calculate_rms(source_file)
    target_rms = calculate_rms(target_file)
    
    # Calculate volume factor (how much to adjust source to match target)
    # Add a small epsilon to prevent division by zero
    volume_factor = (target_rms / (source_rms + 1e-10))
    
    # Apply volume adjustment
    adjust_volume(source_file, output_file, volume_factor)
    
    return volume_factor