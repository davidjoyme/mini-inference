"""
main.py - the HTTP server. A thin layer: turns HTTP requests into Engine calls.

Run from the project root with:
    python main.py

Endpoints:
    GET  /health                 server status and model info
    POST /generate               whole reply at once
    POST /generate_stream        reply streamed as Server-Sent Events (for TTFT/TPOT)
    GET  /metrics/memory         current memory usage
    POST /metrics/memory/reset   reset GPU peak-memory counters
"""
import json

import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

import config
from api.schemas import InferenceRequest
from engine import metrics
from engine.engine import Engine

engine = Engine()   # loads the model once at startup
app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok", **engine.info()}


@app.post("/generate")
def generate(req: InferenceRequest):
    result = engine.generate(req.prompt, req.max_new_tokens, req.temperature)
    return {"input": req.prompt, **result}


@app.post("/generate_stream")
def generate_stream(req: InferenceRequest):
    def sse():
        for event in engine.stream(req.prompt, req.max_new_tokens, req.temperature):
            yield f"data: {json.dumps(event)}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(sse(), media_type="text/event-stream")


@app.get("/metrics/memory")
def get_memory():
    return metrics.memory_snapshot()


@app.post("/metrics/memory/reset")
def reset_memory():
    return metrics.reset_peak_memory()


if __name__ == "__main__":
    uvicorn.run(app, host=config.HOST, port=config.PORT)