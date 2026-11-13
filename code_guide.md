# Code guide: what every file does

A tour of the repository for learning it and explaining it in interviews. The main
[README](../README.md) covers *what* the project does and the results; this guide covers *where* each
idea lives in the code and *how* the pieces fit together.

Each file entry has:
- **What it does**: its job, in a sentence or two.
- **Key parts**: the functions and classes worth knowing by name.
- **Interview angle**: the concept it demonstrates, and questions it prepares you for.

---

## 1. The big picture in one paragraph

`main.py` is a small HTTP server. It hands every request to an **engine** chosen at startup. There are
two engines. `hf` is the baseline: transformers' own `generate()`, one request at a time. `custom` is the
real project, built like vLLM. A **scheduler** decides which requests run in each step. A **KV cache**
stores every request's keys and values in one shared GPU memory pool. A **model runner** does one forward
pass for the whole batch, reusing the Hugging Face model's weights but with its own attention and cache
handling. A **sampler** picks each request's next token. On top of that sit optional speed-ups (custom
Triton and CUDA C++ kernels, CUDA graphs, fused projections), quantization, and a measurement toolkit:
benchmarks, a sweep runner, a profiler and a bottleneck analyzer. Every feature is a switch in
`config.py`, and tests check that every combination produces exactly the same tokens as
Hugging Face's `generate()`.

---

## 2. The life of a request (the best thing to walk through in an interview)

Following one request through the custom engine touches almost every important file:

| Step | What happens | File |
|---|---|---|
| 1 | `POST /generate_stream` arrives; the body is validated | `main.py`, `api/schemas.py` |
| 2 | The prompt is tokenized (chat template applied) and wrapped in a `Request` | `engine/custom_engine.py` → `submit()`, `engine/core/request.py` |
| 3 | The request joins the scheduler's waiting queue | `engine/core/scheduler.py` → `add()` |
| 4 | The engine loop thread wakes up and asks for this step's batch | `engine/custom_engine.py` → `_loop()` |
| 5 | The scheduler admits the request if the KV cache has room (or preempts the newest request if blocks run out), and may cut a long prompt into chunks | `scheduler.py` → `schedule()`, `_admit()`, `_preempt()` |
| 6 | The KV cache reserves blocks for it, and may reuse cached blocks of an identical prompt prefix | `engine/core/kv_cache.py` → `PagedAllocator.admit()` |
| 7 | The runner builds batch metadata (token ids, positions, KV slots, page tables) and runs one forward pass over all requests' new tokens at once | `engine/core/model_runner.py` → `BatchMetadata`, `forward()` |
| 8 | Inside each layer: projections (maybe fused), keys/values written to the cache, attention (SDPA or the paged Triton kernel), MLP or MoE (maybe the fused MoE kernel) | `model_runner.py` → `_attention()`, `kernels/` |
| 9 | Pure-decode steps can instead replay a recorded CUDA graph | `engine/core/cuda_graphs.py` |
| 10 | Logits become next tokens | `engine/core/sampler.py` |
| 11 | Tokens go into each request's outbox; finished requests free their KV blocks | `custom_engine.py` → `_loop()`, `scheduler.finish()` |
| 12 | The server streams each token back as a server-sent event | `main.py` → `generate_stream()` |

---

## 3. Top-level files

### `config.py`
- **What it does:** every setting in one place: model choice, engine and features, quantization, kernels, batch limits, benchmark defaults. Most settings can be overridden with an environment variable (for example `MODEL=qwen3-4b`, `CUDA_GRAPHS=1`), so a feature is chosen at startup without editing code.
- **Key parts:**
  - `MODELS` (short names → Hugging Face ids), `MODEL`, `MAX_MODEL_LEN`
  - `ENGINE`, `SCHEDULER`, `KV_CACHE`, `KV_CACHE_DTYPE`, `PREFIX_CACHING`, `CHUNKED_PREFILL`
  - `MOE_KERNEL`, `MOE_ALIGN`, `ATTN_BACKEND`, `CUDA_GRAPHS`, `FUSED_PROJ`, `WEIGHT_LAYOUT`
  - `QUANTIZATION`, `MAX_NUM_SEQS`, `MAX_BATCHED_TOKENS`
- **Interview angle:** "Every optimization is a switch, so I can measure each one in isolation against the same baseline." Feature flags are what make A/B measurement possible.

### `main.py`
- **What it does:** the FastAPI server. It creates the engine once at startup and turns HTTP calls into engine calls.
- **Key parts:**
  - `/generate`: returns the full answer.
  - `/generate_stream`: returns tokens one by one as server-sent events (SSE). This is what lets clients measure time to first token.
  - `/metrics/memory`, `/metrics/engine`: memory, GPU usage and engine counters, for load tests.
  - `/health`: startup check.
- **Interview angle:** why streaming matters (users see the first token early); why the server is "thin" (all logic lives in the engine, so benchmarks can call the engine directly without HTTP noise).

### `api/schemas.py`
- **What it does:** the Pydantic model for a request body (`prompt`, optional `max_new_tokens`, `temperature`). FastAPI validates incoming JSON against it.
- **Interview angle:** input validation at the API boundary.

### `requirements.txt`, `.gitignore`
- Dependencies. `ninja` builds the CUDA C++ extension.
- `.gitignore` keeps large generated files (quantized weights, profiler traces) out of git.

---

## 4. `engine/`: the engines and model loading

### `engine/base.py`
- **What it does:** the abstract `Engine` interface every engine implements: `generate`, `stream`, `generate_timed`, `engine_config`, `stats`, `info`. The server and all benchmarks only talk to this interface.
- **Interview angle:** an interface boundary lets you swap the baseline and the custom engine without touching the server or the benchmarks. That's the same idea as vLLM's `LLMEngine` API.

### `engine/factory.py`
- **What it does:**
  - `create_engine()` builds the engine named in `config.ENGINE`.
  - `engine_label()` turns an engine's settings into a short name such as `continuous/paged+paged-attn+cuda-graphs`, used in every results table.

### `engine/hf_engine.py`
- **What it does:** the **baseline**: a transformers pipeline behind a lock, so requests run one at a time. `TokenTimer` records when each token appears, to measure TTFT and TPOT. Latency is measured from arrival, so time spent queued behind the lock counts.
- **Interview angle:** "What does an inference engine add over `model.generate()`?" This file is the answer's starting point: without batching, the GPU sits mostly idle and users queue.

### `engine/custom_engine.py`
- **What it does:** **the heart of the project.** It ties the scheduler, KV cache, model runner and sampler together, like vLLM's `LLMEngine`. A background thread runs the **engine loop**: schedule → forward → sample → deliver, step after step.
- **Key parts:**
  - `__init__`:
    - checks the model architecture (`engine/models.py`);
    - sizes the KV cache from free GPU memory (`_kv_budget_tokens`);
    - creates the runner and scheduler;
    - applies fused projections and captures CUDA graphs if enabled.
  - `_loop()`: the step loop. It also times decode steps and prompt steps separately, which is how the round 3 mixed-step finding was measured.
  - `submit()`: tokenizes a prompt, rejects empty prompts, and queues the request.
  - `stream()` / `generate()`: read tokens from a request's outbox.
  - `score()`: computes perplexity through the engine's own cache, for quality checks.
  - `stats()` / `engine_config()`: counters and settings, saved with every result.
- **Interview angle:** the producer/consumer design (HTTP threads submit, one engine thread steps the model); why one step serves many requests; how failures in one step are contained.

### `engine/models.py`
- **What it does:** the **model registry**. `describe()` checks the architecture is supported and detects how it differs from the others:
  - `qk_norm`: none (Llama, Qwen2), per-head (Qwen3), or over the full projection (OLMoE);
  - `moe`: Mixture-of-Experts or dense MLP;
  - head counts: query vs key/value heads (GQA) and head size.

  It rejects models it can't run correctly, such as sliding-window attention or OLMo 2's different layer layout.
- **Interview angle:** "How would you add a new model?" Answer: the engine reuses each model's own modules and only replaces attention and caching, so a new pre-norm decoder only needs its differences described here. This is what vLLM's and SGLang's model registries do, at a smaller scale.

### `engine/model_loader.py`
- **What it does:** `load_model()` loads the model in bf16 or quantized. If `MOE_KERNEL=triton`, it swaps in the fused MoE kernel. It then prints diagnostics: GPU, memory used, attention implementation, and whether anything was offloaded to CPU.
- **Interview angle:** where model loading meets optimization; the diagnostics are a guard against silent slow paths, such as layers offloaded to CPU.

### `engine/metrics.py`
- **What it does:** resource measurement:
  - `memory_snapshot()`: process RAM, GPU memory allocated and reserved, peak memory, and, via NVML, GPU utilization and power;
  - `reset_peak_memory()`.
- **Interview angle:** what to monitor in a serving system (memory headroom, utilization, power) and why peak memory matters for admission control.

---

## 5. `engine/core/`: the custom engine's building blocks

### `engine/core/request.py`
- **What it does:** `Request` (prompt ids, generated ids, state, timings, an outbox queue) and `SamplingParams`.
- **Key parts:** the state machine. `WAITING` → `RUNNING` → `FINISHED`, and back to `WAITING` if preempted.

### `engine/core/scheduler.py`
- **What it does:** decides **which requests run in each step**. `schedule()` returns a list of chunks `(request, start, end)`, meaning "feed this request's tokens from position start to end".
- **Key parts:**
  - Three policies:
    - `_schedule_static`: wait for a group, run it until all finish;
    - `_schedule_continuous`: requests join and leave every step (vLLM-style);
    - `_schedule_chunked`: chunked prefill.
  - `_admit()`: admission control. A request is admitted only if the KV cache can take it; the watermark keeps a little room so running requests can grow.
  - `_preempt()`: when the paged cache runs out of blocks, the newest request is evicted and recomputed later.
  - **Chunked prefill:** a per-step token budget (`MAX_BATCHED_TOKENS`) is filled with decode tokens first, then slices of long prompts, so one long prompt can't stall everyone's decoding.
  - `stats()`: steps, batch sizes, preemptions, mixed steps.
- **Interview angle:**
  - static vs continuous batching (padding waste, head-of-line blocking);
  - why decode goes first (latency for users already streaming);
  - the trade-off of preemption by recompute vs swapping to CPU.

### `engine/core/kv_cache.py`
- **What it does:** the **KV cache**: one big tensor pool `[layers, slots, kv_heads, head_dim]` for keys and one for values, plus two ways of handing out slots.
- **Key parts:**
  - `ContiguousAllocator`: each request reserves one contiguous region for prompt plus maximum output (pre-vLLM style). It wastes the unused reservation and fragments memory (first-fit allocation).
  - `PagedAllocator`: fixed 16-token blocks handed out as sequences grow, with a block table per request (PagedAttention).
    - **Prefix caching:** blocks are identified by a chained hash of their tokens and everything before them, shared between requests with reference counts, and evicted least-recently-used.
    - At least one prompt token is always recomputed, so there's a logit to sample from.
  - `KVCache`:
    - `write()` and `gather()`;
    - optional **FP8/INT8 storage**, with one scale per token per head, which roughly doubles capacity;
    - `decode_tables()`: page tables for the paged attention kernel;
    - `stats()`: usage, waste, fragmentation.
  - `reserve_scratch_block()`: a block reserved for CUDA graphs' padding rows.
- **Interview angle:** this file covers the most-asked inference topics: why KV memory limits concurrency, internal vs external fragmentation, how paging fixes it, how prefix caching works and when it helps (shared system prompts, multi-turn chat), and the accuracy/capacity trade-off of 8-bit KV.

### `engine/core/model_runner.py`
- **What it does:** **one forward pass for the whole batch.** All requests' new tokens are joined into one flat sequence with no padding. Token-wise layers (norms, projections, MLP/MoE) process them together, which is where batching pays off: weights are read once for everyone. Only attention keeps requests apart.
- **Key parts:**
  - `BatchMetadata`: everything the step needs, built once and reused by every layer (token ids, positions, KV write slots, which rows are prefill vs decode, decode page tables).
  - `ModelRunner.forward()`: a pure-decode step goes to the CUDA graph if graphs are enabled; otherwise it builds metadata and runs `forward_meta()`.
  - `forward_meta()`: the layer loop (attention + MLP/MoE). This is the function CUDA graphs record.
  - `_attention()`, which:
    - computes q/k/v (fused or separate) and applies the model's q/k norm style;
    - applies rotary embeddings;
    - writes k/v into the cache;
    - runs attention. Prefill uses SDPA, causal or with a mask after a cached prefix. Decode uses either SDPA over a gathered copy or the paged Triton kernel.
  - `fuse_projections()`: switches to fused q/k/v and gate/up matmuls.
- **Interview angle:**
  - prefill vs decode (compute-bound vs memory-bound);
  - why a flat batch beats padding;
  - how continuous batching mixes prefill and decode in one step.

### `engine/core/sampler.py`
- **What it does:** `sample()` turns logits into tokens: greedy at temperature 0, otherwise softmax sampling. Rows are handled together, so one batch can mix both.

### `engine/core/cuda_graphs.py`
- **What it does:** **CUDA graphs for decode.** A decode step launches hundreds of small kernels, and at small batches the CPU takes longer to *launch* them than the GPU takes to *run* them. A graph records the step once, and replaying it is a single launch.
- **Key parts:**
  - `DecodeGraphs.capture()`: one graph per batch-size bucket (1, 2, 4, 8, 12, 16, 24 … 64), sharing one memory pool.
  - `run()`: copies this step's values into fixed buffers (from pinned memory, asynchronously), pads the batch to the bucket size and replays. Padding rows write into the reserved scratch block.
  - Steps containing prompt tokens fall back to the normal path.
  - `emulate=True` runs the same bookkeeping on CPU, without recording, so tests can check it exactly.
- **Interview angle:**
  - why graphs need *static* shapes and addresses;
  - how that's solved with buckets, padding, fixed buffers and a scratch block;
  - why kernels must not sync with the CPU (the CUDA `moe_align` and the paged kernel were written to be capturable);
  - the measured result: 1.9× lower latency at batch 1.

### `engine/core/fused_linear.py`
- **What it does:** **fused projections and weight-layout autotuning** (round 3). Decode matmuls are skinny (1–64 rows), and cuBLAS's kernel choice made some run at 17–50% of peak bandwidth.
- **Key parts:**
  - `FusedLinear`: one matmul for q+k+v, and one for gate+up in dense models. The weight is stored as `[out, in]` or `[in, out]`, and `split()` returns views.
  - `fuse_model()` / `_fuse()`: build the fused weights. The original modules' weights become *views* into them, so no extra memory is used. Quantized layers are skipped; a tied output layer keeps its layout.
  - `tune_layout()` / `LayoutChooser`: time both layouts per weight shape at 1/16/64 rows. Weights rotate through ≥256 MB of copies so reads come from memory, not L2. The layout switches only when the other is ≥5% faster. Results are cached in `tuning/weight_layouts.json`, so runs are reproducible.
- **Interview angle:** memory-bound GEMMs and arithmetic intensity; how vLLM merges projections (`QKVParallelLinear`); autotuning and why its results are cached (Triton does the same).

---

## 6. `engine/quantization/`: weight quantization

| File | What it does |
|---|---|
| `__init__.py` | `get_method()`: builds the method named in `config.QUANTIZATION` |
| `base.py` | `QuantMethod` interface: `quantize`, `dequantize`, `linear`, `supports`, `describe` |
| `int8_wo.py` | **INT8 weight-only (W8A16):** one scale per output channel, `scale = max(|row|) / 127` |
| `int4_wo.py` | **INT4 group-wise (W4A16):** groups of 128 values, each with its own scale and minimum (asymmetric); two 4-bit values packed per byte |
| `fp8.py` | **FP8 W8A8 (e4m3):** per-channel weight scales, per-token activation scales computed on the fly; real FP8 matmuls (`torch._scaled_mm`) when the GPU supports them, emulated otherwise |
| `modules.py` | Drop-in layers: `QuantLinear` for `nn.Linear`, `QuantExperts` for 3D MoE expert weights (OLMoE and Qwen3-MoE layout) |
| `loader.py` | Quantizes **layer by layer** onto the GPU, so the full bf16 model never has to fit there. Caches the result as safetensors so later starts are fast. Stores tied weights once and re-ties them on load |
| `__main__.py` | Offline CLI: `python -m engine.quantization --method int4_wo` builds the cache ahead of time |

**Interview angle:**
- weight-only vs weight+activation quantization;
- per-channel vs group-wise scales, and why groups help INT4 accuracy;
- why INT4 saves memory but here *doesn't* speed up (weights are unpacked to bf16 at run time without a fused dequant-matmul kernel);
- FP8 formats (e4m3 vs e5m2).

---

## 7. `kernels/`: custom GPU kernels

Each kernel folder has a plain-PyTorch **reference** (obviously correct, slow) that the tests compare against, plus the optimized version.

### `kernels/fused_moe/` (round 1 and 2a)
| File | What it does |
|---|---|
| `reference.py` | `moe_reference()`: the MoE experts computed one expert at a time, used as ground truth |
| `fused_moe_triton.py` | `moe_align()` (PyTorch version): group (token, expert) pairs by expert and pad each group to the tile size. `_moe_matmul_kernel`: one Triton kernel used twice, first gate+up with SiLU fused in, then down × routing weight. `_tile_config()` picks tile sizes from the token count. `fused_moe()` runs the whole layer |
| `csrc/moe_align.cu` | **CUDA C++** `moe_align` in **one kernel launch** (replacing ~35 PyTorch ops). One thread block counts pairs per expert with shared-memory atomics, computes padded offsets, labels each tile with its expert and scatters pairs into place. Runs on PyTorch's current stream with no CPU sync, so CUDA graphs can capture it |
| `moe_align_cuda.py` | JIT-compiles the `.cu` file with `torch.utils.cpp_extension` the first time it's needed, and falls back to the PyTorch version if compiling fails |
| `module.py` | `FusedExperts`: a drop-in replacement for transformers' experts module, sharing its weights. `apply_fused_moe()` swaps it into a loaded model |

**Interview angle:**
- why the default MoE path was launch-bound (one small matmul per expert, ~2,260 kernels per step averaging 4 µs);
- how grouping by expert turns 64 small matmuls into one kernel;
- how the CUDA kernel uses shared-memory atomics;
- tiles and padding;
- the results: 1.6–5.8× per layer, and 1,521 → 2,400 tok/s at 64 users in round 1.

### `kernels/paged_attention/` (round 2b)
| File | What it does |
|---|---|
| `reference.py` | `paged_decode_reference()`: decode attention over a paged cache, one request at a time (ground truth) |
| `decode_triton.py` | `paged_decode_attention()`: **split-KV paged decode attention** (flash-decoding style). Reads keys and values *in place* through page tables instead of gathering a padded copy. Each request's context is split so even one request fills the GPU, and a second kernel merges the splits using log-sum-exp. Online softmax (running max and sum). FP8/INT8 caches are dequantized in registers. Supports GQA |

**Interview angle:**
- why decode attention is memory-bound;
- what PagedAttention means at the kernel level (block tables, address calculation);
- why split-KV matters at small batches;
- online softmax;
- the results: 92% of peak bandwidth, 2.2× faster decoding on long prompts, 14–19× faster attention on FP8 caches.

---

## 8. `bench/`: measuring everything

| File | What it does |
|---|---|
| `run_engine.py` | **The main benchmark.** Runs the engine directly (no HTTP), with `-c` requests in flight at once. Reports TTFT, TPOT, throughput, memory, steps, preemptions, and the average time of decode steps vs prompt steps. `--profile torch` adds a profiler trace and a bottleneck report. Every feature has a flag (`--model`, `--attn-backend`, `--cuda-graphs`, `--fused-proj`, `--kv-tokens`, …) |
| `test.py` | **Server load test** over HTTP: many concurrent streaming requests, with latency percentiles and server memory sampled over time |
| `sweep.py` | Runs the server and a load test for every combination of quantization × engine variant × users × prompt length, and writes one CSV row per run. Engine variants are written like `continuous/paged+prefix+kv-fp8` |
| `plot_sweep.py` | Charts from a sweep (throughput, latency, memory vs users) |
| `compare.py` | Side-by-side tables of any results files, including model and engine labels |
| `eval_quality.py` | **Quality checks:** perplexity, and greedy outputs compared with a reference run (exact-match rate, where outputs first differ). Used to show that quantization, kernels and fusion keep quality |
| `prompts.py` | Builds benchmark prompts: short, long with shared text (for prefix caching), or unique |
| `micro/bench_moe.py` | One MoE layer: default vs fused. Time, kernels per call, CPU launch time, bandwidth |
| `micro/bench_attention.py` | One layer's decode attention: gather + SDPA vs the paged kernel, for bf16/FP8 caches and equal or ragged lengths |
| `micro/bench_gemm.py` | Every linear layer of a model at decode batch sizes: GB/s, the kernel cuBLAS picked, fused vs separate, both weight layouts |

**Interview angle:**
- TTFT vs TPOT vs throughput, and per-user vs total throughput;
- why microbenchmarks have to defeat the L2 cache;
- why before/after pairs must run back to back on a shared GPU;
- how a microbenchmark predicts an end-to-end gain (2.9 ms predicted vs 2.91 ms measured; 2.1 vs 2.0 ms).

---

## 9. `profiling/`: finding bottlenecks

| File | What it does |
|---|---|
| `nvtx.py` | `nvtx_range()`: labels code regions (`prepare_batch`, `forward(...)`, `attention`, `moe`/`mlp`, `sample_and_sync`) for both Nsight and the PyTorch profiler |
| `torch_profiler.py` | Runs code under the PyTorch profiler across *all threads* (the engine loop runs in its own thread) and saves a trace |
| `analyze.py` | **The bottleneck report.** From a trace, it reports: time per engine-loop phase, GPU busy %, GPU time by region and by kernel kind, decode steps by batch size with estimated memory bandwidth vs peak, and plain-language hints. `model_profile()` estimates bytes read per step, counting tied output layers and only the experts a batch actually uses |
| `scripts/profile_suite.sh` | Profiles six configurations, each exposing a different bottleneck, then prints a summary |
| `scripts/nsys.sh`, `scripts/ncu.sh` | Wrappers for Nsight Systems (timeline) and Nsight Compute (per-kernel detail) |

**Interview angle:**
- the profile → bottleneck → fix → verify loop;
- reading GPU-busy % (launch-bound vs GPU-bound);
- roofline thinking: bytes per step ÷ peak bandwidth = the floor for a step;
- Amdahl's law: why fixing a 25% share can save at most 25%.

---

## 10. `tests/`: correctness

All tests run on CPU, with no downloads: they build tiny random models, and Triton kernels run in the interpreter. Most compare against Hugging Face `model.generate()` **token for token** in float32. On a GPU they also run the real kernels.

| File | What it proves |
|---|---|
| `test_batching.py` | Every scheduler × KV cache combination produces exactly `generate()`'s tokens, including when a tiny pool forces preemption. Also holds shared helpers (`build_tiny_model`, `make_engine`) |
| `test_kv_features.py` | KV quantization stays accurate and doubles capacity; prefix caching is exact, survives preemption and eviction, and never gives false hits |
| `test_chunked_prefill.py` | No step exceeds the token budget; decode and prompt slices share steps; outputs are exact |
| `test_quantization.py` | Quantize/dequantize accuracy, native vs emulated FP8, expert routing, and that a freshly quantized model matches its cached copy |
| `test_fused_moe.py` | Fused MoE vs reference and vs transformers. The CUDA `moe_align` groups pairs identically to the PyTorch one; all tile sizes agree |
| `test_paged_attention.py` | Paged kernel vs reference (GQA, page sizes, split counts, INT8/FP8); engine outputs exact with both caches, preemption, prefix caching and chunked prefill |
| `test_cuda_graphs.py` | Graph replay gives the same tokens across bucket changes, preemption, prefix caching and INT8 KV (emulated on CPU, real capture on GPU) |
| `test_models.py` | Tiny Qwen3, Qwen3-MoE, Qwen2 and Llama with every feature; fused projections in both layouts; quantization with tied embeddings; unsupported models rejected |

**Interview angle:** "How do you know your optimizations didn't break anything?" Exact token
equality against an independent reference, across every feature combination, including the hard
cases (preemption, cache eviction, padding rows in CUDA graphs).

---

## 11. Concept → where it lives

| Concept | File(s) |
|---|---|
| Continuous vs static batching | `engine/core/scheduler.py` |
| Paged KV cache / PagedAttention memory management | `engine/core/kv_cache.py` → `PagedAllocator` |
| Paged attention *kernel* | `kernels/paged_attention/decode_triton.py` |
| Prefix caching | `kv_cache.py` → `_block_hashes`, `_match`, `register`, LRU eviction |
| Chunked prefill | `scheduler.py` → `_schedule_chunked` |
| Preemption and recompute | `scheduler.py` → `_preempt`; `kv_cache.py` |
| KV cache quantization | `kv_cache.py` → `_quantize`, `write`, `gather` |
| Weight quantization | `engine/quantization/` |
| Fused MoE | `kernels/fused_moe/` |
| CUDA C++ extension | `kernels/fused_moe/csrc/moe_align.cu`, `moe_align_cuda.py` |
| CUDA graphs | `engine/core/cuda_graphs.py` |
| Fused projections and autotuning | `engine/core/fused_linear.py` |
| Supporting several models | `engine/models.py` |
| Prefill vs decode in one batch | `engine/core/model_runner.py` |
| TTFT / TPOT / throughput | `bench/run_engine.py`, `bench/test.py` |
| Bandwidth / roofline estimates | `profiling/analyze.py`, `bench/micro/*` |

---

## 12. Questions to practice with

Each one has a file that holds the answer:

1. *Walk me through what happens when a request arrives.* → section 2.
2. *Why is decode memory-bound and prefill compute-bound?* → `model_runner.py`; the bandwidth estimates in `profiling/analyze.py`.
3. *How does paged KV caching reduce waste? What's left over?* → `kv_cache.py` (the partly filled last block of each request).
4. *What happens when the KV cache is full?* → `scheduler.py` → `_preempt` (newest request evicted, recomputed later).
5. *How does prefix caching decide two prompts share blocks?* → chained block hashes in `kv_cache.py`.
6. *Why did the fused MoE kernel make batch-1 slower at first, and how did you fix it?* → launch overhead from `moe_align`; `csrc/moe_align.cu` (README round 2a).
7. *Why split-KV in the attention kernel?* → `decode_triton.py` docstring (one request = 16 programs on a 170-SM GPU).
8. *What constraints do CUDA graphs impose, and how did you meet them?* → `cuda_graphs.py` docstring.
9. *How did you find the dense-model bottleneck?* → `bench/micro/bench_gemm.py` → `fused_linear.py` (README round 3).
10. *How do you make benchmark results trustworthy on a shared GPU?* → fixed KV pool (`--kv-tokens`), back-to-back pairs, cached autotuning, L2-defeating microbenchmarks.