from fastapi import FastAPI, HTTPException, WebSocket, UploadFile, File
from fastapi.responses import StreamingResponse, Response, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from ollama import AsyncClient, ResponseError
from kokoro import KPipeline
import json
import wave
import io
import numpy as np
import asyncio
from typing import AsyncGenerator, Optional, Dict, List
import re
import torch
import sounddevice as sd
import time
import os
import cProfile
import pstats
import io as sysio
import traceback
import base64
from PyPDF2 import PdfReader
# Import our streaming TTS implementation
from tts_streaming import StreamingTTS, AudioPlayer
# Import autocast for mixed precision - works with both CUDA and MPS
from torch.amp import autocast
from mlx_audio.tts.generate import generate_audio
from mlx_audio.stt.generate import generate as stt_generate
import unicodedata
from subprocess import run, CalledProcessError

# Enable GPU acceleration for M4 Max
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
torch.backends.mps.enable_fallback_to_cpu = True

# Check if MPS (Metal Performance Shaders) is available
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Metal Performance Shaders) for GPU acceleration")
    # Verify MPS is working
    test_tensor = torch.randn(2, 3).to(device)
    print(f"Test tensor device: {test_tensor.device}")
else:
    device = torch.device("cpu")
    print("MPS not available, using CPU")

# Set matmul precision and threads for Apple Silicon
try:
    torch.set_float32_matmul_precision('high')
    torch.set_num_threads(1)
    print(f"Set matmul precision to high and num_threads to 1")
except Exception as e:
    print(f"Warning: Could not set matmul precision or num_threads: {e}")

app = FastAPI()

# Initialize TTS pipeline with GPU acceleration
tts_pipeline = KPipeline(lang_code='a')  # Reintroduced lang_code with default value 'en'

# Optimize for M4 Max
if torch.backends.mps.is_available():
    # Set optimal thread count for M4 Max
    torch.set_num_threads(8)  # Adjust based on your core count
    # Enable memory efficient attention
    torch.backends.mps.enable_mem_efficient_attention = True
    # Set optimal memory allocation strategy
    torch.backends.mps.enable_mem_efficient_sdp = True
    print("MPS optimizations enabled for M4 Max")

# Preload the preferred voice at startup
preferred_voice_filename = "af_heart.pt"  # Your preferred voice
voice_path = os.path.join(os.path.dirname(__file__), "kokoro_voices", preferred_voice_filename)

if not os.path.exists(voice_path):
    print(f"Error: Preferred voice file {voice_path} not found. Please ensure it's downloaded.")
else:
    print(f"Preloading preferred voice: {preferred_voice_filename}")
    preloaded_voice = torch.load(voice_path, weights_only=True)

# Move pipeline to GPU if available (ensure all submodules are moved)
if hasattr(tts_pipeline, 'to'):
    tts_pipeline = tts_pipeline.to(device)
    # If KPipeline has submodules, move them too (pseudo-code, adjust as needed)
    if hasattr(tts_pipeline, 'modules'):
        for m in tts_pipeline.modules():
            if hasattr(m, 'to'):
                m.to(device)
    print(f"Pipeline moved to device: {device}")
    if hasattr(tts_pipeline, 'device'):
        print(f"Pipeline device: {tts_pipeline.device}")

# Pydantic model for incoming messages (now includes optional 'thinking' and 'images')
class Message(BaseModel):
    role: str
    content: str
    images: list[str] | None = None  # Added for image input
    file_type: str | None = None  # Added to distinguish between image and PDF

class ChatRequest(BaseModel):
    model: str
    messages: list[Message]

# Voice speed configurations
VOICE_SPEEDS = {
    'im_nicola': 1.0,
    'af_bella': 1.0,
    'af_heart': 1.0,
    'af_nicole': 1.3
}

class TTSRequest(BaseModel):
    text: str
    voice: str
    is_streaming: bool = False

# Cleanup function for temporary audio files
def cleanup_temp_files():
    """Clean up any temporary TTS audio files that might have been left behind."""
    try:
        for file in os.listdir('.'):
            if file.startswith('tts_') and file.endswith('.wav'):
                try:
                    os.remove(file)
                    print(f"Cleaned up temporary file: {file}")
                except Exception as e:
                    print(f"Warning: Could not remove temporary file {file}: {e}")
    except Exception as e:
        print(f"Error during cleanup: {e}")

# Run cleanup on startup
cleanup_temp_files()

@app.get("/api/models")
async def list_models():
    try:
        client = AsyncClient()
        model_list = await client.list()
        models = [m.get("name") or m.get("model") for m in model_list.get("models", [])]
        return {"models": models}
    except ResponseError as e:
        raise HTTPException(status_code=e.status_code, detail=e.error)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        messages_payload = []
        for m in request.messages:
            msg_dict = {"role": m.role, "content": m.content}
            if m.images:
                # If it's a PDF, extract text from the base64 content
                if m.file_type == 'pdf':
                    try:
                        # Decode base64 PDF content
                        pdf_content = base64.b64decode(m.images[0])
                        pdf_file = io.BytesIO(pdf_content)
                        pdf_reader = PdfReader(pdf_file)
                        
                        # Extract text from all pages
                        pdf_text = ""
                        for page in pdf_reader.pages:
                            pdf_text += page.extract_text() + "\n"
                        
                        # Add the extracted text to the message content
                        msg_dict["content"] = f"{m.content}\n\nPDF Content:\n{pdf_text}"
                    except Exception as e:
                        print(f"Error processing PDF: {str(e)}")
                        msg_dict["content"] = f"{m.content}\n\n[Error processing PDF content]"
                else:
                    msg_dict["images"] = m.images
            messages_payload.append(msg_dict)
        
        client = AsyncClient()
        stream = await client.chat(model=request.model, messages=messages_payload, stream=True)

        async def generate():
            async for chunk in stream:
                content = chunk.get("message", {}).get("content", "")
                if content:
                    yield content

        return StreamingResponse(generate(), media_type="text/plain")
    except ResponseError as e:
        raise HTTPException(status_code=e.status_code, detail=e.error)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Helper function to split text into smaller chunks for batch processing
def split_text_into_chunks(text, max_length=200):
    """Split text into chunks of a specified maximum length."""
    return [text[i:i + max_length] for i in range(0, len(text), max_length)]

# Initialize global streaming TTS instance
streaming_tts = None

# Function to get or create the streaming TTS instance
def get_streaming_tts():
    global streaming_tts, tts_pipeline, device
    if streaming_tts is None:
        print("Creating new StreamingTTS instance")
        streaming_tts = StreamingTTS(tts_pipeline, device=device)
    return streaming_tts

# WebSocket connection manager for TTS streaming
class TTSConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            
    async def send_audio_chunk(self, client_id: str, audio_data):
        if client_id in self.active_connections:
            try:
                websocket = self.active_connections[client_id]
                await websocket.send_bytes(audio_data)
            except Exception as e:
                print(f"Error sending audio chunk: {e}")
                self.disconnect(client_id)

# Instantiate the connection manager
tts_manager = TTSConnectionManager()

# WebSocket endpoint for real-time streaming TTS
@app.websocket("/api/tts/stream/{client_id}")
async def stream_tts(websocket: WebSocket, client_id: str):
    await tts_manager.connect(websocket, client_id)
    active_generation = False
    
    try:
        while True:
            data = await websocket.receive_text()
            if not data:
                continue
                
            try:
                message = json.loads(data)
                text = message.get("text", "")
                voice = message.get("voice", "af_heart")
                
                if not text:
                    await websocket.send_json({"status": "error", "message": "No text provided"})
                    continue

                # Clean the text before TTS processing
                cleaned_text = preprocess_text_for_tts(text)
                if not cleaned_text:
                    await websocket.send_json({"status": "error", "message": "No text remaining after cleaning"})
                    continue
                
                # Stop any existing generation first
                if active_generation:
                    # No need to stop as we're using non-streaming generation
                    pass
                
                active_generation = True
                
                # Generate audio using mlx_audio.tts.generate
                timestamp = int(time.time())
                output_path = f"tts_{timestamp}.wav"
                
                generate_audio(
                    text=cleaned_text,
                    voice=voice,
                    speed=VOICE_SPEEDS.get(voice, 1.0),
                    lang_code="a",
                    file_prefix=f"tts_{timestamp}",
                    audio_format="wav",
                    sample_rate=24000,
                    join_audio=True,
                    verbose=True
                )
                
                # Read and send the generated audio
                if os.path.exists(output_path):
                    with open(output_path, 'rb') as f:
                        wav_data = f.read()
                    
                    # Send audio data
                    await tts_manager.send_audio_chunk(client_id, wav_data)
                    
                    # Clean up
                    try:
                        os.remove(output_path)
                    except Exception as e:
                        print(f"Warning: Could not remove temporary file {output_path}: {e}")
                    
                    await websocket.send_json({
                        "status": "complete",
                        "message": "TTS generation complete"
                    })
                else:
                    await websocket.send_json({
                        "status": "error",
                        "message": "Generated audio file not found"
                    })
                
            except json.JSONDecodeError:
                await websocket.send_json({"status": "error", "message": "Invalid JSON"})
            except Exception as e:
                print(f"Error in stream_tts: {str(e)}")
                await websocket.send_json({
                    "status": "error", 
                    "message": f"TTS streaming error: {str(e)}"
                })
    except Exception as e:
        print(f"WebSocket connection error: {str(e)}")
    finally:
        active_generation = False
        tts_manager.disconnect(client_id)

def preprocess_text_for_tts(text: str) -> str:
    """Clean text before sending to TTS to avoid reading formatting marks and emojis."""
    # Remove emojis and other special characters
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"  # dingbats
        u"\U000024C2-\U0001F251" 
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub('', text)
    
    # Remove markdown formatting marks
    text = re.sub(r'[*_~`]', '', text)
    
    # Remove any remaining control characters
    text = ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'C')
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    return text.strip()

# Optimize TTS processing with two options:
# 1. Traditional approach (batched with lower latency)
# 2. Real-time streaming for immediate feedback
@app.post("/api/tts")
async def text_to_speech(request: TTSRequest):
    if request.is_streaming:
        return JSONResponse(
            status_code=400,
            content={"error": "For streaming TTS, use the WebSocket endpoint /api/tts/stream"}
        )
    
    try:
        start_time = time.time()
        print(f"[{start_time}] TTS request received")

        if not request.text:
            return JSONResponse(
                status_code=400,
                content={"error": "No text provided"}
            )

        # Clean the text before TTS processing
        cleaned_text = preprocess_text_for_tts(request.text)
        if not cleaned_text:
            return JSONResponse(
                status_code=400,
                content={"error": "No text remaining after cleaning"}
            )

        # Get the speed for the selected voice
        speed = VOICE_SPEEDS.get(request.voice, 1.0)

        # Generate a unique timestamp for the file
        timestamp = int(start_time)
        output_path = f"tts_{timestamp}.wav"

        try:
            # Generate audio using mlx_audio.tts.generate
            generate_audio(
                text=cleaned_text,
                voice=request.voice,
                speed=speed,
                lang_code="a",
                file_prefix=f"tts_{timestamp}",
                audio_format="wav",
                sample_rate=24000,
                join_audio=True,
                verbose=True
            )

            # Read the generated audio file
            if not os.path.exists(output_path):
                raise FileNotFoundError(f"Generated audio file not found: {output_path}")

            with open(output_path, 'rb') as f:
                wav_data = f.read()

            return Response(content=wav_data, media_type="audio/wav")
            
        finally:
            # Clean up the temporary file
            try:
                if os.path.exists(output_path):
                    os.remove(output_path)
                    print(f"Cleaned up temporary file: {output_path}")
            except Exception as e:
                print(f"Warning: Could not remove temporary file {output_path}: {e}")
                # Try cleanup again on next request
                cleanup_temp_files()

        total_time = time.time() - start_time
        print(f"[{time.time()}] Total TTS processing time: {total_time:.3f}s")
        
    except Exception as e:
        print(f"TTS Error: {str(e)}")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": f"TTS processing failed: {str(e)}"}
        )

@app.post("/api/stt")
async def speech_to_text(file: UploadFile = File(...)):
    """Accepts an audio file (webm or wav), converts if needed, transcribes it using mlx_audio STT, and returns the text. Cleans up all temp files after processing."""
    import shutil
    import mimetypes
    import uuid
    import logging
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("stt_endpoint")
    
    temp_id = uuid.uuid4().hex
    input_file = f"temp_stt_{temp_id}.input"
    temp_wav = f"temp_stt_{temp_id}.wav"
    
    try:
        # Save uploaded file to a temp location
        file_bytes = await file.read()
        file_size = len(file_bytes)
        logger.info(f"Received audio file: {file.filename}, size: {file_size} bytes")
        
        with open(input_file, "wb") as f:
            f.write(file_bytes)
            
        # Check for empty/corrupt files
        if file_size < 1000:
            logger.warning(f"Audio file too small: {file_size} bytes")
            return JSONResponse(status_code=400, content={"error": "Uploaded audio file is empty or too small."})

        # Try to determine file type from filename extension or content
        extension = file.filename.split('.')[-1].lower() if '.' in file.filename else ''
        logger.info(f"File extension: {extension}")

        # Convert to wav using ffmpeg with appropriate settings
        ffmpeg_success = False
        ffmpeg_errors = []
        
        # Different conversion strategies based on file extension
        conversion_commands = []
        
        if extension in ['webm', 'ogg', 'opus']:
            conversion_commands = [
                # Try direct conversion first
                ["ffmpeg", "-y", "-i", input_file, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", temp_wav],
                # Try with explicit format specification
                ["ffmpeg", "-y", "-f", extension, "-i", input_file, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", temp_wav],
                # Try with higher volume
                ["ffmpeg", "-y", "-i", input_file, "-ar", "16000", "-ac", "1", "-filter:a", "volume=1.5", "-c:a", "pcm_s16le", temp_wav]
            ]
        else:  # Try generic conversion for unknown formats
            conversion_commands = [
                ["ffmpeg", "-y", "-i", input_file, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", temp_wav],
                ["ffmpeg", "-y", "-f", "webm", "-i", input_file, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", temp_wav],
                ["ffmpeg", "-y", "-f", "ogg", "-i", input_file, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", temp_wav],
                # Try with volume boost
                ["ffmpeg", "-y", "-i", input_file, "-ar", "16000", "-ac", "1", "-filter:a", "volume=1.5", "-c:a", "pcm_s16le", temp_wav],
            ]
        
        # Try conversion commands in sequence until one succeeds
        for cmd in conversion_commands:
            try:
                logger.info(f"Running ffmpeg command: {' '.join(cmd)}")
                result = run(cmd, capture_output=True, check=True)
                ffmpeg_success = True
                logger.info("ffmpeg conversion successful")
                break
            except CalledProcessError as e:
                error_msg = e.stderr.decode() if e.stderr else str(e)
                ffmpeg_errors.append(f"{cmd[3]}: {error_msg}")
                logger.warning(f"ffmpeg conversion failed: {error_msg}")
        
        if not ffmpeg_success:
            logger.error(f"All ffmpeg conversions failed: {ffmpeg_errors}")
            return JSONResponse(status_code=500, content={"error": f"Audio conversion failed. Please try a different format or recording."})
        
        # Use smaller model for faster processing if accuracy is acceptable
        # Use whisper-large-v3-turbo for higher accuracy but slower processing
        model_path = "mlx-community/whisper-large-v3-turbo"
        
        # Optional: Use a smaller model for faster processing (uncomment if preferred)
        # model_path = "mlx-community/whisper-small-int4"
        
        output_path = temp_wav + ".txt"
        logger.info(f"Starting transcription with model: {model_path}")
        
        # Run transcription
        from mlx_audio.stt.generate import generate as stt_generate
        segments = stt_generate(
            model_path=model_path, 
            audio_path=temp_wav, 
            output_path=output_path, 
            format="txt", 
            verbose=True
        )
        
        # Get transcription text
        transcription_text = getattr(segments, "text", "")
        logger.info(f"Transcription complete. Text length: {len(transcription_text)}")
        
        # Clean up and return text
        return {"text": transcription_text}
    except Exception as e:
        logger.error(f"STT processing error: {str(e)}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        # Always clean up temp files
        for path in [input_file, temp_wav, temp_wav + ".txt", temp_wav + ".txt.txt"]:
            try:
                if os.path.exists(path):
                    os.remove(path)
                    logger.info(f"Cleaned up: {path}")
            except Exception as e:
                logger.warning(f"Failed to clean up {path}: {str(e)}")
                
        # Remove any lingering temp_stt_* files (catch-all)
        for f in os.listdir('.'):
            if f.startswith('temp_stt_'):
                try:
                    os.remove(f)
                    logger.info(f"Cleaned up lingering file: {f}")
                except Exception as e:
                    logger.warning(f"Failed to clean up lingering file {f}: {str(e)}")

# Mount static files
app.mount("/", StaticFiles(directory="frontend/build", html=True), name="static")
