"""
engine/custom_engine.py - the custom engine: scheduler + KV cache + model runner (like vLLM's LLMEngine).

A background thread runs the engine loop:
    1. scheduler.schedule()   pick this step's batch (admit / preempt as needed)
    2. runner.forward()       one forward pass for every request in the batch
    3. sample                 one new token per request
    4. deliver                push tokens to each request's outbox; finish completed requests

Callers (HTTP handlers, benchmarks) never touch the model: they submit a request and read
tokens from its outbox. Implements the interface in engine/base.py, so main.py and every
benchmark work with this engine or the baseline (engine/hf_engine.py).

The building blocks live in engine/core/: scheduler, KV cache, model runner, sampler, request.
"""
import queue
import threading
import time

import torch

import config
from engine.core.kv_cache import KVCache, tokens_for_budget
from engine.core.model_runner import ModelRunner
from engine.core.request import FINISHED, Request, SamplingParams
from engine.core.sampler import sample
from engine.core.scheduler import Scheduler
from engine.base import Engine
from engine.model_loader import load_model
from profiling.nvtx import nvtx_range


class CustomEngine(Engine):
    def __init__(self):
        generator, self.load_info = load_model()
        self.model = generator.model
        self.tokenizer = generator.tokenizer
        cfg = self.model.config
        self.device = next(self.model.parameters()).device
        self.dtype = next(self.model.parameters()).dtype if self.model.dtype is None else self.model.dtype
        self.max_model_len = cfg.max_position_embeddings

        num_heads = cfg.num_attention_heads
        kv_heads = getattr(cfg, "num_key_value_heads", None) or num_heads
        head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // num_heads
        capacity = self._kv_budget_tokens(cfg.num_hidden_layers, kv_heads, head_dim)
        self.cache = KVCache(config.KV_CACHE, cfg.num_hidden_layers, kv_heads, head_dim, capacity,
                             self.dtype, self.device, block_size=config.KV_BLOCK_SIZE,
                             watermark=config.KV_WATERMARK, kv_dtype=config.KV_CACHE_DTYPE,
                             prefix_caching=config.PREFIX_CACHING)
        self.runner = ModelRunner(self.model, self.cache)
        self.scheduler = Scheduler(config.SCHEDULER, self.cache, config.MAX_NUM_SEQS,
                                   config.MAX_BATCHED_TOKENS, self.max_model_len,
                                   config.STATIC_BATCH_WAIT_S, chunked_prefill=config.CHUNKED_PREFILL)

        eos = self.model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, (list, tuple)) else [eos])
        if self.tokenizer.eos_token_id is not None:
            eos.add(self.tokenizer.eos_token_id)
        self.eos_ids = {e for e in eos if e is not None}

        self.lock = threading.Condition()
        self.last_error = None
        print(f"Custom engine: scheduler={config.SCHEDULER}, kv_cache={config.KV_CACHE}, "
              f"kv_dtype={self.cache.stats()['kv_cache_dtype']}, prefix_caching={self.cache.prefix_caching}, "
              f"chunked_prefill={self.scheduler.chunked_prefill} | "
              f"KV capacity {self.cache.capacity} tokens ({self.cache.memory_mb()} MB, "
              f"{self.cache.bytes_per_token / 1024:.1f} KB/token), max_num_seqs={self.scheduler.max_num_seqs}")
        self.thread = threading.Thread(target=self._loop, daemon=True, name="engine-loop")
        self.thread.start()

    # ------------------------------------------------------------------ setup
    def _kv_budget_tokens(self, layers, kv_heads, head_dim):
        if config.KV_CACHE_TOKENS:
            return config.KV_CACHE_TOKENS
        if config.KV_CACHE_MB:                                  # fixed memory: format decides tokens
            return tokens_for_budget(config.KV_CACHE_MB * 1024 ** 2, layers, kv_heads, head_dim,
                                     self.dtype, config.KV_CACHE_DTYPE)
        if self.device.type != "cuda":
            return config.CPU_KV_CACHE_TOKENS
        free, total = torch.cuda.mem_get_info(self.device)
        in_use = total - free                                   # model + anything else on the GPU
        budget = total * config.GPU_MEMORY_UTILIZATION - in_use - config.ACTIVATION_RESERVE_GB * 1024 ** 3
        tokens = tokens_for_budget(max(budget, 0), layers, kv_heads, head_dim, self.dtype,
                                   config.KV_CACHE_DTYPE)     # fp8/int8 fit ~2x the tokens
        if tokens < self.max_model_len:
            print(f"WARNING: only {tokens} KV tokens fit; set KV_CACHE_TOKENS or free GPU memory")
            tokens = max(tokens, 1024)
        return tokens

    # ------------------------------------------------------------------ engine loop
    def _loop(self):
        while True:
            with self.lock:
                while not self.scheduler.has_work():
                    self.lock.wait()
                with nvtx_range("schedule"):
                    chunks = self.scheduler.schedule(time.perf_counter())
            if not chunks:                          # static batching waiting to fill a batch
                time.sleep(0.002)
                continue
            try:
                logits = self.runner.forward(chunks)
                # .tolist() waits for the GPU to finish this step, so this range includes that wait
                with nvtx_range("sample_and_sync"):
                    tokens = sample(logits, [r.params.temperature for r, _, _ in chunks]).tolist()
            except Exception as e:                  # e.g. out of memory: fail this batch, keep serving
                self.last_error = f"{type(e).__name__}: {e}"
                print(f"[engine] step failed: {self.last_error}")
                with self.lock:
                    for r in self.scheduler.abort_all():
                        self._deliver_done(r, error=self.last_error)
                continue

            now = time.perf_counter()
            with self.lock, nvtx_range("deliver"):
                for (req, _, end), tok in zip(chunks, tokens):
                    req.num_computed = end
                    if end < req.total_len:          # chunked prefill: prompt not finished yet
                        self.cache.register(req)
                        continue
                    req.output_ids.append(tok)
                    if req.is_padding:
                        continue                     # static batching: computed, then thrown away
                    self.cache.register(req)         # prefix caching: publish newly full blocks
                    if req.first_token_at is None:
                        req.first_token_at = now
                    req.token_times.append(now)
                    req.outbox.put(("token", tok, now))
                    if (tok in self.eos_ids or len(req.output_ids) >= req.params.max_new_tokens
                            or req.total_len >= self.max_model_len):
                        req.finish_reason = "stop" if tok in self.eos_ids else "length"
                        self.scheduler.finish(req)
                        self._deliver_done(req)

    def _deliver_done(self, req, error=None):
        req.state = FINISHED
        req.finished_at = time.perf_counter()
        if error:
            req.finish_reason = "error"
        req.outbox.put(("done", error, req.finished_at))

    # ------------------------------------------------------------------ requests
    def _encode(self, prompt):
        if config.IS_CHAT_MODEL and self.tokenizer.chat_template:
            text = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                                      add_generation_prompt=True, tokenize=False)
            return self.tokenizer(text, add_special_tokens=False)["input_ids"]
        return self.tokenizer(prompt)["input_ids"]

    def submit(self, prompt, max_new_tokens=None, temperature=None):
        ids = self._encode(prompt)
        limit = min(self.max_model_len, self.cache.capacity)
        if len(ids) >= limit:
            raise ValueError(f"Prompt is {len(ids)} tokens; the limit is {limit - 1}")
        max_new = min(max_new_tokens or config.MAX_NEW_TOKENS, limit - len(ids))
        temp = 0.0 if not config.DO_SAMPLE else (temperature if temperature is not None else config.TEMPERATURE)
        req = Request(prompt_ids=ids, params=SamplingParams(max_new_tokens=max_new, temperature=temp))
        with self.lock:
            self.scheduler.add(req)
            self.lock.notify()
        return req

    def _events(self, req):
        while True:
            kind, value, t = req.outbox.get()
            yield kind, value, t
            if kind == "done":
                return

    # ------------------------------------------------------------------ public API (same as Engine)
    def stream(self, prompt, max_new_tokens=None, temperature=None):
        req = self.submit(prompt, max_new_tokens, temperature)
        ids, sent, error = [], "", None
        for kind, value, _ in self._events(req):
            if kind == "done":
                error = value
                break
            ids.append(value)
            text = self.tokenizer.decode(ids, skip_special_tokens=True)
            if text.endswith("�"):              # half of a multi-byte character: wait
                continue
            if len(text) > len(sent):
                yield {"text": text[len(sent):]}
                sent = text
        final = {"done": True, "model": config.MODEL_NAME, **self._request_metrics(req)}
        if error:
            final["error"] = error
        yield final

    def generate(self, prompt, max_new_tokens=None, temperature=None):
        req = self.submit(prompt, max_new_tokens, temperature)
        error = None
        for kind, value, _ in self._events(req):
            if kind == "done":
                error = value
        m = self._request_metrics(req)
        result = {"output": self.tokenizer.decode(self._delivered(req), skip_special_tokens=True),
                  "model": config.MODEL_NAME, **m,
                  "tokens_per_s": round(m["generated_tokens"] / m["generation_time_s"], 2)
                  if m.get("generation_time_s") else None}
        if error:
            result["error"] = error
        return result

    def generate_timed(self, prompt, max_new_tokens=None, temperature=None):
        req = self.submit(prompt, max_new_tokens, temperature)
        for _ in self._events(req):
            pass
        t, n = req.token_times, len(req.token_times)
        latency = req.finished_at - req.arrived
        return {
            "output": self.tokenizer.decode(self._delivered(req), skip_special_tokens=True),
            "generated_tokens": n,
            "queue_time_s": round(req.first_scheduled - req.arrived, 4) if req.first_scheduled else None,
            "latency_s": round(latency, 4),
            "ttft_s": round(t[0] - req.arrived, 4) if t else None,
            "tpot_ms": round((t[-1] - t[0]) / (n - 1) * 1000, 3) if n > 1 else None,
            "itl_ms": [round((b - a) * 1000, 3) for a, b in zip(t, t[1:])],
            "tokens_per_s": round(n / latency, 2) if latency > 0 else None,
        }

    @staticmethod
    def _delivered(req):
        return req.output_ids[: len(req.token_times)]

    @staticmethod
    def _request_metrics(req):
        r = lambda x: round(x, 3) if x is not None else None   # noqa: E731
        sched, first, done = req.first_scheduled, req.first_token_at, req.finished_at
        return {
            "generated_tokens": len(req.token_times),
            "queue_time_s": r(sched - req.arrived) if sched else None,
            "prefill_time_s": r(first - sched) if first and sched else None,
            "decode_time_s": r(done - first) if first and done else None,
            "generation_time_s": r(done - sched) if done and sched else None,
            "server_ttft_s": r(first - req.arrived) if first else None,
            "preemptions": req.preemptions,
        }

    def score(self, token_ids):
        """Logits for every position of token_ids, computed through this engine's KV cache
        (so KV quantization is included). Used by bench/eval_quality.py for perplexity."""
        req = Request(prompt_ids=list(token_ids), params=SamplingParams(max_new_tokens=1))
        alloc = self.cache.allocator
        with self.lock:                          # the engine loop can't schedule meanwhile
            saved = getattr(alloc, "prefix_caching", False)
            alloc.prefix_caching = False         # score every token; don't reuse or publish blocks
            try:
                if self.cache.admit(req, self.max_model_len) is None or not self.cache.ensure(req, len(token_ids)):
                    self.cache.free(req)
                    raise RuntimeError("Not enough free KV cache to score this text")
                try:
                    return self.runner.forward([(req, 0, len(token_ids))], all_logits=True)
                finally:
                    self.cache.free(req)
            finally:
                alloc.prefix_caching = saved

    def engine_config(self):
        return {
            "engine": "custom",
            "scheduler": config.SCHEDULER,
            "kv_cache": config.KV_CACHE,
            "kv_block_size": config.KV_BLOCK_SIZE if config.KV_CACHE == "paged" else None,
            "kv_watermark": config.KV_WATERMARK if config.KV_CACHE == "paged" else None,
            "kv_cache_dtype": self.cache.stats()["kv_cache_dtype"],
            "prefix_caching": self.cache.prefix_caching,
            "kv_capacity_tokens": self.cache.capacity,
            "kv_cache_mb": self.cache.memory_mb(),
            "max_num_seqs": self.scheduler.max_num_seqs,
            "max_batched_tokens": config.MAX_BATCHED_TOKENS,
            "chunked_prefill": self.scheduler.chunked_prefill,
            "moe_kernel": self.load_info.get("moe_kernel", "default"),
        }

    def stats(self):
        with self.lock:
            return {**self.scheduler.stats(), **self.cache.stats()}