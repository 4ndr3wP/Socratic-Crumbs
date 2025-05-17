from fastapi import FastAPI, HTTPException, WebSocket
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
# Import our streaming TTS implementation
from tts_streaming import StreamingTTS, AudioPlayer
# Import autocast for mixed precision - works with both CUDA and MPS
from torch.amp import autocast

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

class ChatRequest(BaseModel):
    model: str
    messages: list[Message]

class TTSRequest(BaseModel):
    text: str
    is_streaming: bool = False  # Flag to indicate if this is part of a streaming response

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
                msg_dict["images"] = m.images
            messages_payload.append(msg_dict)
        
        # Ensure images are only attached to the last message if it's from the user,
        # or handle as per Ollama's multi-message image support if applicable.
        # For now, assuming images are part of the latest user prompt.
        
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
    streamer = get_streaming_tts()
    active_generation = False
    
    try:
        while True:
            data = await websocket.receive_text()
            if not data:
                continue
                
            try:
                message = json.loads(data)
                text = message.get("text", "")
                
                if not text:
                    await websocket.send_json({"status": "error", "message": "No text provided"})
                    continue
                
                # Stop any existing generation first
                if active_generation:
                    streamer.stop()
                
                active_generation = True
                
                # Optimized segment callback with reduced processing
                async def on_segment_callback(i, gs, ps, audio):
                    if isinstance(audio, torch.Tensor):
                        # Keep on GPU until last moment
                        audio = audio.detach().cpu().numpy()
                    
                    # Optimize audio processing
                    if audio.dtype != np.float32:
                        audio = audio.astype(np.float32)
                    
                    # Use memory-efficient WAV conversion
                    with io.BytesIO() as wav_buffer:
                        with wave.open(wav_buffer, 'wb') as wav_file:
                            wav_file.setnchannels(1)
                            wav_file.setsampwidth(2)
                            wav_file.setframerate(24000)
                            # Direct conversion to int16 without intermediate steps
                            wav_file.writeframes((audio * 32767).astype(np.int16).tobytes())
                        
                        wav_data = wav_buffer.getvalue()
                    
                    # Send audio chunk immediately
                    await tts_manager.send_audio_chunk(client_id, wav_data)
                    
                    # Send minimal metadata
                    await websocket.send_json({
                        "status": "segment",
                        "index": i,
                        "is_final": False
                    })
                
                # Use preloaded voice and optimized generation
                voice = preloaded_voice if 'preloaded_voice' in globals() else None
                
                # Launch TTS generation with optimized parameters
                asyncio.create_task(
                    asyncio.to_thread(
                        streamer.generate_and_play,
                        text=text,
                        voice=voice,
                        speed=1.0,
                        split_pattern=r'[.!?]\s+',
                        on_segment_callback=on_segment_callback
                    )
                )
                
                await websocket.send_json({
                    "status": "started", 
                    "message": "TTS streaming started"
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
        if streamer:
            streamer.stop()
        tts_manager.disconnect(client_id)

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
    
    pr = cProfile.Profile()
    pr.enable()
    try:
        start_time = time.time()
        print(f"[{start_time}] TTS request received")

        if not request.text:
            return JSONResponse(
                status_code=400,
                content={"error": "No text provided"}
            )

        # Optimize chunk size for M4 Max
        text_chunks = split_text_into_chunks(request.text, max_length=1000)  # Increased for better throughput
        print(f"Text split into {len(text_chunks)} chunks for batch processing")

        # Profile generator creation for each chunk with mixed precision
        generator_creation_start = time.time()
        audio_segments = []
        
        # Use optimized mixed precision and batch processing
        with autocast(device_type='mps' if torch.backends.mps.is_available() else 'cpu'):
            # Process chunks in parallel using asyncio
            async def process_chunk(chunk):
                with torch.inference_mode():
                    generator = tts_pipeline(
                        chunk,
                        voice=preloaded_voice if 'preloaded_voice' in globals() else None,
                        speed=1.0,
                        split_pattern=r'[.!?]\s+'
                    )
                    return [audio for _, _, audio in generator]

            # Process chunks concurrently
            tasks = [process_chunk(chunk) for chunk in text_chunks]
            chunk_results = await asyncio.gather(*tasks)
            
            # Flatten results
            for result in chunk_results:
                audio_segments.extend(result)
        
        generator_creation_end = time.time()
        print(f"Generator creation and segment collection time: {generator_creation_end - generator_creation_start:.3f}s")

        if not audio_segments:
            return JSONResponse(
                status_code=500,
                content={"error": "Failed to generate audio"}
            )

        # Optimize concatenation
        concatenation_start = time.time()
        combined_audio = torch.cat(audio_segments, dim=0)
        concatenation_end = time.time()
        print(f"Audio concatenation time: {concatenation_end - concatenation_start:.3f}s")

        # Optimize CPU transfer
        move_to_cpu_start = time.time()
        if isinstance(combined_audio, torch.Tensor):
            combined_audio = combined_audio.detach().cpu()
        move_to_cpu_end = time.time()
        print(f"Move to CPU time: {move_to_cpu_end - move_to_cpu_start:.3f}s")

        # Optimize WAV conversion
        wav_conversion_start = time.time()
        wav_data = convert_to_wav(combined_audio)
        wav_conversion_end = time.time()
        print(f"WAV conversion time: {wav_conversion_end - wav_conversion_start:.3f}s")

        total_time = time.time() - start_time
        print(f"[{time.time()}] Total TTS processing time: {total_time:.3f}s")

        return Response(content=wav_data, media_type="audio/wav")
        
    except Exception as e:
        print(f"TTS Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": f"TTS processing failed: {str(e)}"}
        )
    finally:
        pr.disable()
        s = sysio.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats(30)
        print("\n--- cProfile TTS /api/tts ---\n" + s.getvalue())

def convert_to_wav(audio_data):
    """Convert audio data to WAV format with proper headers."""
    # Convert to numpy array if it's a tensor
    if isinstance(audio_data, torch.Tensor):
        audio_data = audio_data.cpu().numpy()
    
    # Ensure audio is in the correct format
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32)
    
    # Create WAV file in memory
    with io.BytesIO() as wav_buffer:
        # Write WAV header and data
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(24000)  # Sample rate
            wav_file.writeframes((audio_data * 32767).astype(np.int16).tobytes())
        
        # Get the complete WAV data
        wav_data = wav_buffer.getvalue()
    
    return wav_data

# ✅ Only mount the frontend after defining /api routes
app.mount("/", StaticFiles(directory="frontend/build", html=True), name="static")
