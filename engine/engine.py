"""
engine/engine.py - the inference engine.

Everything that touches the model lives here, so the rest of the code (server,
benchmarks, profiling) only talks to the Engine class. For now it uses the
transformers pipeline internally; later this is where the custom prefill/decode
loop, KV cache strategies and decoding strategies plug in.
"""
import threading
import time

import torch
from transformers import TextIteratorStreamer

import config
from engine.model_loader import load_model
from profiling.nvtx import nvtx_range


class Engine:
    def __init__(self):
        self.generator = load_model()
        self.tokenizer = self.generator.tokenizer
        # The pipeline isn't safe to call from several threads at once.
        self.lock = threading.Lock()

    # ---------------- helpers ----------------
    def info(self) -> dict:
        return {
            "model": config.MODEL_NAME,
            "is_chat_model": config.IS_CHAT_MODEL,
            "dtype": str(self.generator.model.dtype),
            "device": str(self.generator.model.device),
            "cuda_available": torch.cuda.is_available(),
            "do_sample": config.DO_SAMPLE,
        }

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def _gen_kwargs(self, max_new_tokens=None, temperature=None) -> dict:
        kwargs = {
            "max_new_tokens": max_new_tokens or config.MAX_NEW_TOKENS,
            "do_sample": config.DO_SAMPLE,
        }
        if config.DO_SAMPLE:
            kwargs["temperature"] = temperature if temperature is not None else config.TEMPERATURE
        return kwargs

    def _run_model(self, prompt: str, gen_kwargs: dict, streamer=None):
        """One full generation. Returns the reply text (or None when streaming,
        since the text is delivered through the streamer instead)."""
        model_input = [{"role": "user", "content": prompt}] if config.IS_CHAT_MODEL else prompt
        extra = {} if config.IS_CHAT_MODEL else {"return_full_text": False}

        with nvtx_range("generate"):
            if streamer is not None:
                self.generator(model_input, streamer=streamer, **extra, **gen_kwargs)
                return None
            result = self.generator(model_input, **extra, **gen_kwargs)

        if config.IS_CHAT_MODEL:
            return result[0]["generated_text"][-1]["content"]
        return result[0]["generated_text"]

    # ---------------- whole reply at once ----------------
    def generate(self, prompt: str, max_new_tokens=None, temperature=None) -> dict:
        arrived = time.perf_counter()
        gen_kwargs = self._gen_kwargs(max_new_tokens, temperature)

        with self.lock:
            started = time.perf_counter()   # waiting for the lock = queue time
            reply = self._run_model(prompt, gen_kwargs)
            finished = time.perf_counter()

        tokens = self.count_tokens(reply)
        gen_time = finished - started
        return {
            "output": reply,
            "model": config.MODEL_NAME,
            "generated_tokens": tokens,
            "queue_time_s": round(started - arrived, 3),
            "generation_time_s": round(gen_time, 3),
            "tokens_per_s": round(tokens / gen_time, 2) if gen_time > 0 else None,
        }

    # ---------------- streamed reply ----------------
    def stream(self, prompt: str, max_new_tokens=None, temperature=None):
        """
        Yields events as the reply is generated:
            {"text": "..."}                    for each chunk of text
            {"done": True, ...metrics}         once at the end
        """
        arrived = time.perf_counter()
        gen_kwargs = self._gen_kwargs(max_new_tokens, temperature)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        timings = {}

        def worker():
            with self.lock:
                timings["started"] = time.perf_counter()
                try:
                    self._run_model(prompt, gen_kwargs, streamer=streamer)
                except Exception as e:
                    timings["error"] = f"{type(e).__name__}: {e}"
                    streamer.end()   # unblock the loop below so the caller doesn't hang
                timings["finished"] = time.perf_counter()

        thread = threading.Thread(target=worker)
        thread.start()

        first_chunk_at = None
        pieces = []
        for text in streamer:
            if not text:
                continue
            if first_chunk_at is None:
                first_chunk_at = time.perf_counter()
            pieces.append(text)
            yield {"text": text}

        thread.join()

        started = timings.get("started", arrived)
        finished = timings.get("finished", time.perf_counter())
        final = {
            "done": True,
            "model": config.MODEL_NAME,
            "generated_tokens": self.count_tokens("".join(pieces)),
            "queue_time_s": round(started - arrived, 3),
            # generation start -> first chunk ~ prefill (processing the prompt)
            "prefill_time_s": round(first_chunk_at - started, 3) if first_chunk_at else None,
            "decode_time_s": round(finished - first_chunk_at, 3) if first_chunk_at else None,
            "generation_time_s": round(finished - started, 3),
            "server_ttft_s": round(first_chunk_at - arrived, 3) if first_chunk_at else None,
        }
        if "error" in timings:
            final["error"] = timings["error"]
        yield final