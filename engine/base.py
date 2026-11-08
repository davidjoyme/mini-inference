"""
engine/base.py - the interface every engine implements.

The server (main.py) and every benchmark only ever talk to this interface, through
engine.factory.create_engine(). Which implementation runs is a startup choice (ENGINE=...),
the same way vLLM or SGLang put interchangeable backends behind one engine API:

    HFEngine      engine/hf_engine.py       baseline: transformers pipeline, one request at a time
    CustomEngine  engine/custom_engine.py   scheduler + KV cache + model runner (engine/core/)

Methods to implement:
    generate(prompt, max_new_tokens, temperature) -> dict
        the whole reply: {"output", "generated_tokens", "queue_time_s", "generation_time_s", ...}
    stream(prompt, max_new_tokens, temperature) -> iterator of dicts
        {"text": "..."} for each piece of the reply, then one {"done": True, ...metrics}
    generate_timed(prompt, max_new_tokens, temperature) -> dict
        exact per-token timings: {"ttft_s", "tpot_ms", "itl_ms": [...], "latency_s", ...}
    engine_config() -> dict
        the features this engine runs with, e.g. {"engine": "custom", "scheduler": ...}
Optional:
    stats() -> dict          live counters (batch size, KV usage, preemptions, ...)
    score(token_ids)         logits for every position (perplexity through the engine's own cache)
"""
from abc import ABC, abstractmethod
import torch
import config


class Engine(ABC):
    # set by each implementation's __init__
    tokenizer = None
    model = None
    load_info = None      # from engine.model_loader.load_model(): quantization, weight memory, load time

    @abstractmethod
    def generate(self, prompt, max_new_tokens=None, temperature=None) -> dict:
        ...

    @abstractmethod
    def stream(self, prompt, max_new_tokens=None, temperature=None):
        ...

    @abstractmethod
    def generate_timed(self, prompt, max_new_tokens=None, temperature=None) -> dict:
        ...

    @abstractmethod
    def engine_config(self) -> dict:
        ...

    def stats(self) -> dict:
        return {}

    # ---------------------------------------------------------------- shared helpers
    def count_tokens(self, text) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def info(self) -> dict:
        """What /health reports and every results file records."""
        return {
            "model": config.MODEL_NAME,
            "is_chat_model": config.IS_CHAT_MODEL,
            "dtype": str(self.model.dtype),
            "device": str(self.model.device),
            "cuda_available": torch.cuda.is_available(),
            "do_sample": config.DO_SAMPLE,
            "quantization": {k: v for k, v in self.load_info["quantization"].items()
                             if k not in ("skipped", "cache_path")},
            "weight_memory": self.load_info["memory"],
            "load_time_s": self.load_info["load_time_s"],
            "engine": self.engine_config(),
        }