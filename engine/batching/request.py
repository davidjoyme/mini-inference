"""
engine/batching/request.py - one request's state as it moves through the engine.

Life cycle:
    WAITING  -> in the queue, no KV cache yet
    RUNNING  -> admitted: has KV cache, gets one forward pass per engine step
    FINISHED -> done; the result is delivered and its KV cache is freed
(A preempted request goes back from RUNNING to WAITING and is recomputed later.)
"""
import itertools
import queue
import time
from dataclasses import dataclass, field

WAITING, RUNNING, FINISHED = "waiting", "running", "finished"
_ids = itertools.count()


@dataclass
class SamplingParams:
    max_new_tokens: int = 100
    temperature: float = 0.0      # 0 = greedy (deterministic)


@dataclass
class Request:
    prompt_ids: list
    params: SamplingParams
    id: int = field(default_factory=lambda: next(_ids))
    state: str = WAITING

    output_ids: list = field(default_factory=list)
    # Tokens whose keys/values are already in the KV cache
    num_computed: int = 0
    # Static batching: finished requests keep running as padding until the batch ends
    is_padding: bool = False
    finish_reason: str = None

    # Timings (time.perf_counter())
    arrived: float = field(default_factory=time.perf_counter)
    first_scheduled: float = None
    first_token_at: float = None
    finished_at: float = None
    token_times: list = field(default_factory=list)
    preemptions: int = 0

    # Results go here for the caller to read: ("token", id, time) ... then ("done", None, time)
    outbox: "queue.Queue" = field(default_factory=queue.Queue)

    @property
    def all_ids(self):
        """Prompt plus everything generated so far (what the model has to see)."""
        return self.prompt_ids + self.output_ids

    @property
    def total_len(self):
        return len(self.prompt_ids) + len(self.output_ids)

    @property
    def max_total_len(self):
        """Longest this request can get: what a contiguous cache has to reserve."""
        return len(self.prompt_ids) + self.params.max_new_tokens

    def num_new_tokens(self):
        """Tokens to feed this step: the whole (re)prompt on prefill, 1 when decoding."""
        return self.total_len - self.num_computed