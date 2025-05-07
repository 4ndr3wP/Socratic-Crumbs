from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import ollama

app = FastAPI()

# Pydantic model for incoming messages (now includes optional 'thinking')
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: list[Message]

@app.get("/api/models")
def list_models():
    try:
        model_list = ollama.list()
        models = [m.get("name") or m.get("model") for m in model_list.get("models", [])]
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
def chat(request: ChatRequest):
    try:
        messages_payload = [{"role": m.role, "content": m.content} for m in request.messages]
        stream = ollama.chat(model=request.model, messages=messages_payload, stream=True)

        def generate():
            for chunk in stream:
                content = chunk.get("message", {}).get("content", "")
                if content:
                    yield content

        return StreamingResponse(generate(), media_type="text/plain")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ✅ Only mount the frontend after defining /api routes
app.mount("/", StaticFiles(directory="frontend/build", html=True), name="static")
