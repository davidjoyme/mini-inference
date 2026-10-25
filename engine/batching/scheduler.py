"""
engine/batching/scheduler.py - decides which requests run in each engine step.

Every step is one forward pass. schedule() returns the "chunks" for that pass:
(request, start, end) = feed this request's tokens at positions start..end-1.

Policies:
  sequential  one request at a time (continuous logic with a batch size of 1)
  static      form a batch (wait up to STATIC_BATCH_WAIT_S to fill it), reserve worst-case
              KV cache for every row (like a padded [batch, max_len] tensor), and run until
              EVERY request in it is finished. Finished requests keep computing as padding;
              newcomers wait for the whole batch.
  continuous  every step: make room for running requests' next token (preempting the newest
              if the paged cache is full), then admit waiting requests while there is room
              in the batch, the per-step token budget and the KV cache.

Chunked prefill (CHUNKED_PREFILL=1; sequential and continuous):
  Without it, a new prompt is processed whole in one step: a long prompt makes that step long,
  so everyone who is generating stalls (an inter-token latency spike), and a prompt bigger than
  the per-step budget waits until nothing else is running.
  With it, each step's MAX_BATCHED_TOKENS budget goes first to decoding requests (1 token each),
  then to unfinished prefills, then to new prompts - each prompt only as large a slice as fits.
  Long prompts are spread over several steps and decoding never stops. A request produces its
  first token only once its last prompt slice has been processed.
"""
from collections import deque
from engine.batching.request import RUNNING, WAITING


class Scheduler:
    def __init__(self, policy, cache, max_num_seqs, max_batched_tokens, max_model_len, static_wait_s,
                 chunked_prefill=False):
        if policy not in ("sequential", "static", "continuous"):
            raise ValueError(f"Unknown SCHEDULER '{policy}' (sequential, static, continuous)")
        self.policy = policy
        self.cache = cache
        self.max_num_seqs = 1 if policy == "sequential" else max_num_seqs
        self.max_batched_tokens = max_batched_tokens
        self.max_model_len = max_model_len
        self.static_wait_s = static_wait_s
        self.chunked_prefill = chunked_prefill and policy != "static"
        if chunked_prefill and policy == "static":
            print("Note: chunked prefill applies to the sequential and continuous schedulers only.")

        self.waiting = deque()
        self.running = []
        self.reserved_len = {}          # static: per-request reservation (padded row length)
        # counters (cumulative)
        self.steps = 0
        self.preemptions = 0
        self.padding_tokens = 0         # static: tokens computed for already-finished requests
        self.prefill_tokens = 0
        self.decode_tokens = 0
        self.last_batch_size = 0
        self.last_step_tokens = 0
        self.max_step_tokens = 0        # largest forward pass so far (tokens)
        self.prefill_chunks = 0         # prompt slices that were NOT the end of their prompt
        self.mixed_steps = 0            # steps that ran decode and prefill work together

    def add(self, req):
        self.waiting.append(req)

    def has_work(self):
        return bool(self.waiting or self.running)

    # ------------------------------------------------------------------ scheduling
    def schedule(self, now):
        if self.policy == "static":
            chunks = self._schedule_static(now)
        else:
            chunks = self._schedule_continuous(now)
        if chunks:
            self.steps += 1
            self.last_batch_size = len(chunks)
            self.last_step_tokens = sum(e - s for _, s, e in chunks)
            self.max_step_tokens = max(self.max_step_tokens, self.last_step_tokens)
            has_decode = has_prefill = False
            for req, s, e in chunks:
                if req.is_padding:
                    self.padding_tokens += e - s
                elif e - s == 1 and req.output_ids and e == req.total_len:
                    self.decode_tokens += 1
                    has_decode = True
                else:
                    self.prefill_tokens += e - s
                    has_prefill = True
                    if e < req.total_len:
                        self.prefill_chunks += 1
            self.mixed_steps += has_decode and has_prefill
        return chunks

    def _admit(self, req, now, reserve_len=None, token_limit=None):
        """Give a waiting request KV cache and move it to running. Returns how many of its
        tokens to process this step (all remaining ones, or at most token_limit with chunked
        prefill), or None if it doesn't fit."""
        if reserve_len is not None:                 # static: worst-case reservation up front
            cached = self.cache.admit(req, self.max_model_len, size=reserve_len)
            if cached is None:
                return None
            if not self.cache.reserve(req, reserve_len):
                self.cache.free(req)
                return None
            self.reserved_len[req.id] = reserve_len
        else:
            cached = self.cache.admit(req, self.max_model_len)
            if cached is None:
                return None
        take = req.total_len - cached
        if token_limit is not None:
            take = min(take, token_limit)
        if not self.cache.ensure(req, cached + take):
            self.cache.free(req)
            return None
        # prefix caching: the first `cached` tokens' keys/values are already in shared blocks
        req.num_computed = cached
        self.waiting.remove(req)
        req.state = RUNNING
        if req.first_scheduled is None:
            req.first_scheduled = now
        self.running.append(req)
        return take

    def _preempt(self, req):
        """Free a running request's KV cache and put it back at the front of the queue.
        Its tokens so far are kept; the next prefill recomputes them."""
        self.cache.free(req)
        self.running.remove(req)
        req.state = WAITING
        req.num_computed = 0
        req.preemptions += 1
        self.preemptions += 1
        self.waiting.appendleft(req)

    def _schedule_continuous(self, now):
        if self.chunked_prefill:
            return self._schedule_chunked(now)
        # 1. room for every running request's next token; preempt the newest if needed
        for req in list(self.running):
            if req not in self.running:
                continue
            while not self.cache.ensure(req, req.total_len):
                victim = self.running[-1]
                self._preempt(victim)
                if victim is req:
                    break

        # 2. admit waiting requests (oldest first) while batch size, token budget and cache allow
        budget = self.max_batched_tokens - len(self.running)
        while self.waiting and len(self.running) < self.max_num_seqs:
            req = self.waiting[0]
            if req.num_new_tokens() > budget and self.running:   # a long prompt may still run alone
                break
            if not self._admit(req, now):
                break
            budget -= req.num_new_tokens()        # after admission: cached prefix tokens are free

        return [(r, r.num_computed, r.total_len) for r in self.running]

    def _schedule_chunked(self, now):
        """Token budget per step: decodes first, then unfinished prefills, then new prompts."""
        budget = self.max_batched_tokens
        scheduled = {}                            # request id -> (request, start, end)
        decoding = [r for r in self.running if r.num_new_tokens() == 1]
        prefilling = [r for r in self.running if r.num_new_tokens() > 1]
        for req in decoding + prefilling:
            if req not in self.running or budget <= 0:
                continue
            take = min(req.num_new_tokens(), budget)
            end = req.num_computed + take
            while not self.cache.ensure(req, end):
                victim = self.running[-1]
                self._preempt(victim)
                if victim.id in scheduled:        # it was already in this step: give back its budget
                    _, s, e = scheduled.pop(victim.id)
                    budget += e - s
                if victim is req:
                    break
            if req not in self.running:
                continue
            scheduled[req.id] = (req, req.num_computed, end)
            budget -= take

        while self.waiting and len(self.running) < self.max_num_seqs and budget > 0:
            req = self.waiting[0]
            take = self._admit(req, now, token_limit=budget)
            if take is None:
                break
            scheduled[req.id] = (req, req.num_computed, req.num_computed + take)
            budget -= take

        return list(scheduled.values())

    def _schedule_static(self, now):
        if not self.running:
            if not self.waiting:
                return []
            # give more requests a moment to arrive so the batch isn't tiny
            if len(self.waiting) < self.max_num_seqs and now - self.waiting[0].arrived < self.static_wait_s:
                return []
            candidates, tokens = [], 0
            for req in list(self.waiting)[: self.max_num_seqs]:
                if candidates and tokens + req.total_len > self.max_batched_tokens:
                    break
                candidates.append(req)
                tokens += req.total_len
            # every row is padded to the same length: longest prompt + largest max_new_tokens
            row_len = min(max(len(r.prompt_ids) for r in candidates) +
                          max(r.params.max_new_tokens for r in candidates), self.max_model_len)
            for req in candidates:
                if not self._admit(req, now, reserve_len=row_len):
                    break
            if not self.running:
                return []

        # decode (or prefill) every row, padding rows included, until the batch is done
        chunks = []
        for r in self.running:
            if r.total_len > self.reserved_len.get(r.id, self.max_model_len):
                continue                             # padding row reached its padded length
            self.cache.ensure(r, r.total_len)
            chunks.append((r, r.num_computed, r.total_len))
        return chunks

    def finish(self, req):
        """Called by the engine when a request is done."""
        if self.policy == "static":
            req.is_padding = True                    # keeps computing until the batch ends
            if all(r.is_padding for r in self.running):
                for r in self.running:
                    self.cache.free(r)
                    self.reserved_len.pop(r.id, None)
                self.running = []
        else:
            self.running.remove(req)
            self.cache.free(req)

    def abort_all(self):
        """Drop every running request (e.g. after an error) and free their cache."""
        dropped = list(self.running)
        for r in dropped:
            self.cache.free(r)
            self.reserved_len.pop(r.id, None)
        self.running = []
        return dropped

    def stats(self):
        return {
            "scheduler": self.policy,
            "running": sum(1 for r in self.running if not r.is_padding),
            "padding_rows": sum(1 for r in self.running if r.is_padding),
            "waiting": len(self.waiting),
            "batch_size": self.last_batch_size,
            "step_tokens": self.last_step_tokens,
            "steps": self.steps,
            "preemptions": self.preemptions,
            "prefill_tokens": self.prefill_tokens,
            "decode_tokens": self.decode_tokens,
            "padding_tokens": self.padding_tokens,
            "chunked_prefill": self.chunked_prefill,
            "prefill_chunks": self.prefill_chunks,
            "mixed_steps": self.mixed_steps,
            "max_step_tokens": self.max_step_tokens,
        }