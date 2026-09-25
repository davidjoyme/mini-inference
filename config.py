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

# ---------------- Generation defaults ----------------
MAX_NEW_TOKENS = 100
DO_SAMPLE = False      # False = greedy (deterministic), True = random sampling
TEMPERATURE = 0.7      # only used when DO_SAMPLE is True

# ---------------- Server ----------------
HOST = "127.0.0.1"     # "0.0.0.0" to allow access from other machines
PORT = 8000

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