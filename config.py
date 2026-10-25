"""
config.py - all the important settings in one place.

The server (main.py), the engine, the profiling tools and the benchmarks all read from here.
"""
import os

# ---------------- Model ----------------
# OLMoE: Mixture-of-Experts, ~7B total parameters, ~1B active per token.
# All 7B must fit in memory (~14 GB in bfloat16), even though only ~1B are used per token.
MODEL_NAME = "allenai/OLMoE-1B-7B-0125-Instruct"

# True for chat/instruct models (they take a list of messages), False for base models.
IS_CHAT_MODEL = True

# "auto" spreads the model over available GPUs (needs `accelerate`).
# If it doesn't fit on the GPU, layers get offloaded to CPU and generation becomes very slow.
DEVICE_MAP = "auto"

# OLMoE was trained in bfloat16; float32 would double memory to ~28 GB.
DTYPE = "bfloat16"

# Hide transformers warnings (only real errors are printed)
QUIET_TRANSFORMERS = False

# ---------------- Quantization ----------------
# Which weight quantization to apply when the model loads. Changing it needs a restart.
#   "none"    - plain bfloat16 (~14 GB for OLMoE)
#   "int8_wo" - INT8 weights, bf16 activations, one scale per output channel (~7 GB)
#   "int4_wo" - INT4 weights in groups of QUANT_GROUP_SIZE, bf16 activations (~4 GB)
#   "fp8"     - FP8 (e4m3) weights AND activations (~7 GB); real FP8 matmuls on
#               Ada/Hopper/Blackwell GPUs, emulated (same accuracy, no speedup) elsewhere
# Override without editing this file:  QUANT=int4_wo python main.py
QUANTIZATION = os.environ.get("QUANT", "none")

# Which weights to quantize. In OLMoE:
#   "experts" - the 64 experts in every layer (~93% of all weights)
#   "linear"  - the other linear layers inside decoder layers (attention q/k/v/o projections)
#   "lm_head" - the output head (sensitive; usually left in bf16)
# The router (which picks experts) and the embeddings are never quantized.
QUANT_TARGETS = ["experts", "linear"]

QUANT_GROUP_SIZE = 128        # int4_wo only: weights sharing one scale/offset

# "auto" = real FP8 matmuls when the GPU supports them, otherwise emulation.
# "emulate" = always emulate (useful to compare accuracy on any GPU).
FP8_BACKEND = os.environ.get("FP8_BACKEND", "auto")

# Quantized weights are saved here after the first load, so later restarts with the same
# settings skip the bf16 load and quantization step. Set QUANT_USE_CACHE=0 to disable.
QUANT_CACHE_DIR = "quantized"
QUANT_USE_CACHE = os.environ.get("QUANT_USE_CACHE", "1") == "1"

# ---------------- Engine and serving features ----------------
# Pick features at startup; every one can be overridden with an environment variable,
# e.g.  ENGINE=custom SCHEDULER=continuous KV_CACHE=paged python main.py
#
# ENGINE
#   "hf"      transformers pipeline: one request at a time (the original baseline)
#   "custom"  this project's engine: own scheduler, KV cache and attention (settings below)
ENGINE = os.environ.get("ENGINE", "hf")

# SCHEDULER (custom engine)
#   "sequential"  one request at a time
#   "static"      wait for a group of requests, run them together until ALL finish;
#                 new requests wait for the whole batch, finished ones keep computing as padding
#   "continuous"  every step, finished requests leave and waiting ones join (vLLM-style)
SCHEDULER = os.environ.get("SCHEDULER", "continuous")

# KV_CACHE (custom engine) - both draw from the same memory budget, so they compare fairly
#   "contiguous"  each request reserves one contiguous region for prompt + max_new_tokens
#                 when it's admitted (pre-vLLM style: wasted reservations, fragmentation)
#   "paged"       fixed-size blocks handed out as sequences grow (PagedAttention-style);
#                 when blocks run out, the newest request is preempted and recomputed later
KV_CACHE = os.environ.get("KV_CACHE", "paged")
KV_BLOCK_SIZE = int(os.environ.get("KV_BLOCK_SIZE", 16))   # tokens per block (paged)
# Paged admission keeps this share of blocks free for running requests to grow into.
# Higher = fewer preemptions but fewer requests admitted at once (vLLM's default is 0.01).
KV_WATERMARK = float(os.environ.get("KV_WATERMARK", 0.01))

# How keys/values are stored (custom engine):
#   "auto"  the model's dtype (bfloat16 for OLMoE)
#   "fp8"   FP8 e4m3, one scale per token per head - the common production choice
#   "int8"  INT8, one scale per token per head
# 8-bit storage roughly halves KV memory, so about twice as many tokens fit.
KV_CACHE_DTYPE = os.environ.get("KV_CACHE_DTYPE", "auto")

# Prefix caching (custom engine, KV_CACHE=paged): reuse the KV blocks of identical prompt
# beginnings (system prompts, shared documents, earlier chat turns) instead of recomputing them.
PREFIX_CACHING = os.environ.get("PREFIX_CACHING", "0") == "1"

# KV cache budget. None = automatic: (GPU memory x GPU_MEMORY_UTILIZATION) minus what the
# model already uses, minus ACTIVATION_RESERVE_GB for temporary buffers during forward passes.
KV_CACHE_TOKENS = int(os.environ["KV_CACHE_TOKENS"]) if os.environ.get("KV_CACHE_TOKENS") else None
# Or fix the budget in memory instead: e.g. 2048 MB holds ~2x the tokens with KV_CACHE_DTYPE=fp8.
KV_CACHE_MB = float(os.environ["KV_CACHE_MB"]) if os.environ.get("KV_CACHE_MB") else None
GPU_MEMORY_UTILIZATION = float(os.environ.get("GPU_MEMORY_UTILIZATION", 0.90))
ACTIVATION_RESERVE_GB = 2.0
CPU_KV_CACHE_TOKENS = 32768        # budget used when there is no GPU

# Batch limits (custom engine)
MAX_NUM_SEQS = int(os.environ.get("MAX_NUM_SEQS", 64))           # requests in a batch at once
MAX_BATCHED_TOKENS = int(os.environ.get("MAX_BATCHED_TOKENS", 4096))  # tokens per forward pass
STATIC_BATCH_WAIT_S = float(os.environ.get("STATIC_BATCH_WAIT_S", 0.05))  # static: wait to fill a batch

# Chunked prefill (custom engine, sequential/continuous): split long prompts into slices so each
# step's MAX_BATCHED_TOKENS budget serves decoding requests first and prompts never stall them.
# A smaller MAX_BATCHED_TOKENS (e.g. 512) means smaller slices: smoother decoding, slower prefill.
CHUNKED_PREFILL = os.environ.get("CHUNKED_PREFILL", "0") == "1"


MAX_NEW_TOKENS = 100
DO_SAMPLE = False      # False = greedy (deterministic), True = random sampling
TEMPERATURE = 0.7      # only used when DO_SAMPLE is True

# ---------------- Server ----------------
HOST = "127.0.0.1"     # "0.0.0.0" to allow access from other machines
PORT = int(os.environ.get("PORT", 8000))   # PORT=8100 python main.py to override

# ---------------- Profiling ----------------
# NVTX labels show up in Nsight Systems timelines. Turned on by setting NVTX=1
# (profiling/scripts/nsys.sh does this for you).
NVTX_ENABLED = os.environ.get("NVTX", "0") == "1"
PROFILE_OUTPUT_DIR = "profiles"

# ---------------- Server load tests (bench/test.py) ----------------
TEST_STREAM = True           # True = use /generate_stream and measure TTFT/TPOT/ITL
TEST_URL = f"http://localhost:{PORT}/generate"
TEST_STREAM_URL = f"http://localhost:{PORT}/generate_stream"
TEST_NUM_REQUESTS = 8
TEST_CONCURRENCY = 4
TEST_PROMPT = "Tell me a short fact about space"
TEST_MAX_NEW_TOKENS = 50
TEST_TIMEOUT_S = 600
TEST_OUTPUT_FILE = "results.json"
TEST_MEMORY = True           # collect server memory metrics during the test
TEST_MEMORY_INTERVAL_S = 0.5

# ---------------- Direct engine runs (bench/run_engine.py) ----------------
ENGINE_BENCH_NUM_PROMPTS = 3
ENGINE_BENCH_WARMUP = 1      # warmup runs, not measured (first run is often slower)

# ---------------- Resource sweeps (bench/sweep.py) ----------------
# The sweep starts the server once per quantization, then runs a load test for every
# combination of the lists below and records memory, utilization, latency and throughput.
SWEEP_QUANTIZATIONS = ["none", "int4_wo"]
# "hf" or "<scheduler>/<kv_cache>" for the custom engine
SWEEP_ENGINES = ["hf", "static/contiguous", "continuous/contiguous", "continuous/paged"]
SWEEP_CONCURRENCY = [1, 4, 16, 32]    # number of simultaneous users
SWEEP_MAX_NEW_TOKENS = [64]           # output length
SWEEP_PROMPT_WORDS = [0, 1000]        # extra input length in words (0 = short prompt only)
SWEEP_REQUESTS_PER_USER = 2           # each run sends concurrency x this many requests
SWEEP_MEMORY_INTERVAL_S = 0.25        # sample resources more often than normal tests
SWEEP_STARTUP_TIMEOUT_S = 3600        # first start of a quantization can take a while
SWEEP_PORT = 8100                     # separate port, so it never hits a server you left running
SWEEP_OUTPUT_DIR = "sweeps"