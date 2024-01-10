from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi import WebSocket
from fastapi.responses import JSONResponse, FileResponse
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from pydub import AudioSegment
from typing import List
import yt_dlp
import torch
import io
import os

app = FastAPI()

# Ensure the downloads directory exists
download_dir = 'downloads'
os.makedirs(download_dir, exist_ok=True)

# Function to set up the model and processor
def setup_model():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-large-v3"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=15,
        batch_size=16,
        torch_dtype=torch_dtype,
        device=device,
    )

    return pipe

# Initialize the model and processor
pipe = setup_model()

# Function to convert the sample rate of the audio file
def convert_sample_rate(audio_bytes):
    # Load the audio file using pydub
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
    # Check if the sample rate is 16000 Hz, if not convert it
    if audio.frame_rate != 16000:
        audio = audio.set_frame_rate(16000)
    return audio

# Recursive function to get all mp3 files from a directory and its subdirectories
def get_mp3_files(directory: str) -> List[str]:
    mp3_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.mp3'):
                mp3_files.append(os.path.join(root, file))
    return mp3_files

# Function to transcribe audio using Whisper
def transcribe_audio(pipe, audio_bytes):
    # Convert the audio bytes to the correct sample rate
    audio = convert_sample_rate(audio_bytes)
    # Save the converted audio file to a temporary file
    temp_file = "temp_audio.mp3"
    audio.export(temp_file, format="mp3")

    # Run the transcription pipeline
    result = pipe(temp_file)
    return result["text"]

@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Receive audio chunk
            audio_bytes = await websocket.receive_bytes()
            
            # Process and transcribe the audio chunk
            # NOTE: This requires the Whisper model to handle real-time audio streams
            transcription = transcribe_audio(pipe, audio_bytes)
            
            # Send the transcription back
            await websocket.send_text(transcription)
    except Exception as e:
        await websocket.close(code=1001, reason=str(e))


def clean_up_file(file_path: str):
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        print(f"Error deleting file: {e}")
