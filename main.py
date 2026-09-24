"""
main.py - the inference server.

Run with:
    python main.py
All settings (model, port, generation defaults) come from config.py.

Endpoints:
    GET  /health           - server status and model info
    POST /generate         - returns the whole reply at once
    POST /generate_stream  - streams the reply token-by-token (Server-Sent Events),
                             so clients can measure TTFT / TPOT / ITL
"""
import json
import threading
import time

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import TextIteratorStreamer, pipeline

import config

# Uncomment to hide transformers warnings (only real errors will be printed)
# from transformers.utils import logging as hf_logging
# hf_logging.set_verbosity_error()

# ---------------- Load the model once at startup ----------------
print(f"Loading model: {config.MODEL_NAME} ...")
load_start = time.perf_counter()
generator = pipeline(
    "text-generation",
    model=config.MODEL_NAME,
    device_map=config.DEVICE_MAP,
    dtype=config.DTYPE,   # on older transformers versions, rename this to torch_dtype
)
print(f"Model loaded in {time.perf_counter() - load_start:.1f}s")

# ---------------- Diagnostics: where is the model running? ----------------
print("\n===== Model diagnostics =====")
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print(f"GPU memory used by model: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    total = torch.cuda.get_device_properties(0).total_memory
    print(f"GPU total memory: {total / 1e9:.2f} GB")
print("Model device:", generator.model.device)
print("Model dtype:", generator.model.dtype)
print("Device map:", getattr(generator.model, "hf_device_map", None))
print("=============================\n")

# The pipeline isn't safe to call from several threads at once,
# so this lock makes requests use the model one at a time.
model_lock = threading.Lock()

app = FastAPI()


class InferenceRequest(BaseModel):
    prompt: str
    max_new_tokens: int | None = None   # None = use the default from config.py
    temperature: float | None = None


def build_gen_kwargs(req: InferenceRequest) -> dict:
    gen_kwargs = {
        "max_new_tokens": req.max_new_tokens or config.MAX_NEW_TOKENS,
        "do_sample": config.DO_SAMPLE,
    }
    if config.DO_SAMPLE:
        gen_kwargs["temperature"] = req.temperature if req.temperature is not None else config.TEMPERATURE
    return gen_kwargs


def build_model_input(prompt: str):
    """Chat models take a list of messages; base models take raw text."""
    if config.IS_CHAT_MODEL:
        return [{"role": "user", "content": prompt}]
    return prompt


def count_tokens(text: str) -> int:
    return len(generator.tokenizer(text, add_special_tokens=False)["input_ids"])


@app.get("/health")
def health():
    """Quick check that the server is up, and which model/settings it's using."""
    return {
        "status": "ok",
        "model": config.MODEL_NAME,
        "is_chat_model": config.IS_CHAT_MODEL,
        "dtype": str(generator.model.dtype),
        "device": str(generator.model.device),
        "cuda_available": torch.cuda.is_available(),
        "do_sample": config.DO_SAMPLE,
    }


# ---------------- Non-streaming: whole reply at once ----------------
@app.post("/generate")
def generate(req: InferenceRequest):
    arrived = time.perf_counter()
    gen_kwargs = build_gen_kwargs(req)
    model_input = build_model_input(req.prompt)

    with model_lock:
        started = time.perf_counter()   # time spent waiting for the lock = queue time
        if config.IS_CHAT_MODEL:
            result = generator(model_input, **gen_kwargs)
            reply = result[0]["generated_text"][-1]["content"]
        else:
            result = generator(model_input, return_full_text=False, **gen_kwargs)
            reply = result[0]["generated_text"]
        finished = time.perf_counter()

    generated_tokens = count_tokens(reply)
    generation_time = finished - started

    return {
        "input": req.prompt,
        "output": reply,
        "model": config.MODEL_NAME,
        "generated_tokens": generated_tokens,
        "queue_time_s": round(started - arrived, 3),
        "generation_time_s": round(generation_time, 3),
        "tokens_per_s": round(generated_tokens / generation_time, 2) if generation_time > 0 else None,
    }


# ---------------- Streaming: reply sent piece by piece ----------------
@app.post("/generate_stream")
def generate_stream(req: InferenceRequest):
    arrived = time.perf_counter()
    gen_kwargs = build_gen_kwargs(req)
    model_input = build_model_input(req.prompt)

    # The streamer receives tokens from the model as they're generated.
    # skip_prompt=True so we don't stream the prompt back to the client.
    streamer = TextIteratorStreamer(generator.tokenizer, skip_prompt=True, skip_special_tokens=True)
    timings = {}

    def run_generation():
        # Runs in a background thread; the lock still ensures one generation at a time.
        with model_lock:
            timings["started"] = time.perf_counter()
            try:
                if config.IS_CHAT_MODEL:
                    generator(model_input, streamer=streamer, **gen_kwargs)
                else:
                    generator(model_input, streamer=streamer, return_full_text=False, **gen_kwargs)
            except Exception as e:
                timings["error"] = f"{type(e).__name__}: {e}"
                streamer.end()   # unblock the loop below so the request doesn't hang
            timings["finished"] = time.perf_counter()

    def event_stream():
        thread = threading.Thread(target=run_generation)
        thread.start()

        first_chunk_at = None
        pieces = []
        for text in streamer:
            if not text:
                continue
            if first_chunk_at is None:
                first_chunk_at = time.perf_counter()
            pieces.append(text)
            # Server-Sent Events format: "data: <json>" followed by a blank line
            yield f"data: {json.dumps({'text': text})}\n\n"

        thread.join()

        started = timings.get("started", arrived)
        finished = timings.get("finished", time.perf_counter())
        reply = "".join(pieces)
        generated_tokens = count_tokens(reply)
        decode_time = finished - first_chunk_at if first_chunk_at else None

        final = {
            "done": True,
            "model": config.MODEL_NAME,
            "generated_tokens": generated_tokens,
            "queue_time_s": round(started - arrived, 3),
            # time from generation start to first chunk ~ prefill (processing the prompt)
            "prefill_time_s": round(first_chunk_at - started, 3) if first_chunk_at else None,
            "decode_time_s": round(decode_time, 3) if decode_time is not None else None,
            "generation_time_s": round(finished - started, 3),
            "server_ttft_s": round(first_chunk_at - arrived, 3) if first_chunk_at else None,
        }
        if "error" in timings:
            final["error"] = timings["error"]

        yield f"data: {json.dumps(final)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ---------------- Memory metrics ----------------
def read_proc_status_mb(field: str):
    """Read a memory field (e.g. VmRSS, VmHWM) for this process from /proc (Linux only)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith(field + ":"):
                    return round(int(line.split()[1]) / 1024, 1)   # kB -> MB
    except OSError:
        pass
    return None


def memory_snapshot() -> dict:
    snap = {
        "process_rss_mb": read_proc_status_mb("VmRSS"),        # current RAM used by the server
        "process_peak_rss_mb": read_proc_status_mb("VmHWM"),   # highest RAM since the server started
    }
    if torch.cuda.is_available():
        allocated = reserved = peak_allocated = peak_reserved = device_used = device_total = 0
        for i in range(torch.cuda.device_count()):
            allocated += torch.cuda.memory_allocated(i)            # memory held by tensors right now
            reserved += torch.cuda.memory_reserved(i)              # memory PyTorch has claimed (incl. cache)
            peak_allocated += torch.cuda.max_memory_allocated(i)   # peak since last reset
            peak_reserved += torch.cuda.max_memory_reserved(i)
            free, total = torch.cuda.mem_get_info(i)               # whole-GPU view, like nvidia-smi
            device_used += total - free
            device_total += total
        mb = 1024 ** 2
        snap.update({
            "gpu_allocated_mb": round(allocated / mb, 1),
            "gpu_reserved_mb": round(reserved / mb, 1),
            "gpu_peak_allocated_mb": round(peak_allocated / mb, 1),
            "gpu_peak_reserved_mb": round(peak_reserved / mb, 1),
            "gpu_device_used_mb": round(device_used / mb, 1),
            "gpu_device_total_mb": round(device_total / mb, 1),
        })
    return snap


@app.get("/metrics/memory")
def get_memory():
    return memory_snapshot()


@app.post("/metrics/memory/reset")
def reset_memory_peaks():
    """Reset the GPU peak counters so a test run measures only its own peak."""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(i)
    return memory_snapshot()


if __name__ == "__main__":
    uvicorn.run(app, host=config.HOST, port=config.PORT)