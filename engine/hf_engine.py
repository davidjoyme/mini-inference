"""
engine/hf_engine.py - the baseline engine: the transformers pipeline, one request at a time.

Requests take turns behind a lock, and generation is transformers' own generate() - an
independent reference that the custom engine (engine/custom_engine.py) is benchmarked and
correctness-tested against. Implements the interface in engine/base.py.
"""
import threading
import time

import torch
from transformers import TextIteratorStreamer
from transformers.generation.streamers import BaseStreamer

import config
from engine.base import Engine
from engine.model_loader import load_model
from profiling.nvtx import nvtx_range


class TokenTimer(BaseStreamer):
    """Called by generate() once with the prompt, then once per new token.
    Records exact per-token timestamps without needing a second thread."""

    def __init__(self):
        self.tokens, self.times = [], []
        self._prompt_seen = False

    def put(self, value):
        if not self._prompt_seen:      # first call carries the prompt
            self._prompt_seen = True
            return
        ids = value.reshape(-1).tolist()   # .tolist() waits for the GPU, so the time is real
        self.times.append(time.perf_counter())
        self.tokens.extend(ids)

    def end(self):
        pass


class HFEngine(Engine):
    def __init__(self):
        self.generator, self.load_info = load_model()
        self.tokenizer = self.generator.tokenizer
        # The pipeline isn't safe to call from several threads at once.
        self.lock = threading.Lock()

    # ---------------- helpers ----------------
    @property
    def model(self):
        return self.generator.model

    def engine_config(self) -> dict:
        return {"engine": "hf"}

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

    # ---------------- timed run on the calling thread (benchmarks, profiling) ----------------
    def generate_timed(self, prompt: str, max_new_tokens=None, temperature=None) -> dict:
        """Runs generation on this thread (so profilers see every op) and returns exact
        per-token timings: TTFT, TPOT and the gaps between tokens (ITL)."""
        gen_kwargs = self._gen_kwargs(max_new_tokens, temperature)
        timer = TokenTimer()
        start = time.perf_counter()          # arrival: time waiting for the lock counts too
        with self.lock:
            started = time.perf_counter()
            self._run_model(prompt, gen_kwargs, streamer=timer)
            end = time.perf_counter()

        t, n = timer.times, len(timer.tokens)
        itl = [b - a for a, b in zip(t, t[1:])]
        return {
            "output": self.tokenizer.decode(timer.tokens, skip_special_tokens=True),
            "generated_tokens": n,
            "queue_time_s": round(started - start, 4),
            "latency_s": round(end - start, 4),
            "ttft_s": round(t[0] - start, 4) if t else None,
            "tpot_ms": round((t[-1] - t[0]) / (n - 1) * 1000, 3) if n > 1 else None,
            "itl_ms": [round(x * 1000, 3) for x in itl],
            "tokens_per_s": round(n / (end - start), 2) if end > start else None,
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