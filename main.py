"""
main.py - the HTTP server. A thin layer: turns HTTP requests into engine calls.

Run from the project root with:
    python main.py                                              # settings from config.py
    ENGINE=custom SCHEDULER=continuous KV_CACHE=paged python main.py
    ENGINE=custom SCHEDULER=static KV_CACHE=contiguous QUANT=int4_wo python main.py

Endpoints:
    GET  /health                 server status, model, quantization and engine features
    POST /generate               whole reply at once
    POST /generate_stream        reply streamed as Server-Sent Events (for TTFT/TPOT)
    GET  /metrics/memory         memory, GPU utilization and engine state (batch, KV cache)
    POST /metrics/memory/reset   reset GPU peak-memory counters
    GET  /metrics/engine         engine state only (scheduler and KV cache counters)
"""
import json

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

import config
from api.schemas import InferenceRequest
from engine import metrics
from engine.factory import create_engine

engine = create_engine()   # loads the model once at startup (ENGINE=hf or custom)
app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok", **engine.info()}


@app.post("/generate")
def generate(req: InferenceRequest):
    try:
        result = engine.generate(req.prompt, req.max_new_tokens, req.temperature)
    except ValueError as e:          # e.g. prompt longer than the context window
        raise HTTPException(status_code=400, detail=str(e))
    return {"input": req.prompt, **result}


@app.post("/generate_stream")
def generate_stream(req: InferenceRequest):
    def sse():
        try:
            for event in engine.stream(req.prompt, req.max_new_tokens, req.temperature):
                yield f"data: {json.dumps(event)}\n\n"
        except ValueError as e:
            yield f"data: {json.dumps({'done': True, 'error': str(e)})}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(sse(), media_type="text/event-stream")


@app.get("/metrics/memory")
def get_memory():
    return {**metrics.memory_snapshot(), **engine.stats()}


@app.post("/metrics/memory/reset")
def reset_memory():
    return {**metrics.reset_peak_memory(), **engine.stats()}


@app.get("/metrics/engine")
def get_engine_stats():
    return engine.stats()


if __name__ == "__main__":
    uvicorn.run(app, host=config.HOST, port=config.PORT)