"""
config.py - all the important settings in one place.

main.py (the server) and test.py (the test bench) both read from here.
Change a value here, restart the server, and re-run the tests.
"""

# ---------------- Model ----------------
# Hugging Face model ID or a local folder path, e.g. "/home/dj69594/models/my-model"
# OLMoE: Mixture-of-Experts, ~7B total parameters, ~1B active per token.
# All 7B must fit in memory (~14 GB in bfloat16), even though only ~1B are used per token.
MODEL_NAME = "allenai/OLMoE-1B-7B-0125-Instruct"

# True for chat/instruct models (they take a list of messages),
# False for plain base models (they take raw text).
IS_CHAT_MODEL = True

# "auto" spreads the model over available GPUs (needs `accelerate`).
# If it doesn't fit on the GPU, layers get offloaded to CPU and generation becomes very slow.
DEVICE_MAP = "auto"

# OLMoE was trained in bfloat16; loading in float32 would double memory to ~28 GB.
# Options: "auto", "float16", "bfloat16", "float32"
DTYPE = "bfloat16"

# ---------------- Generation defaults ----------------
# Used when a request doesn't specify its own values.
MAX_NEW_TOKENS = 100
DO_SAMPLE = False      # False = greedy (deterministic), True = random sampling
TEMPERATURE = 0.7      # only used when DO_SAMPLE is True

# ---------------- Server ----------------
HOST = "127.0.0.1"     # use "0.0.0.0" to allow access from other machines
PORT = 8000

# ---------------- Test bench ----------------
TEST_STREAM = True           # True = use /generate_stream and measure TTFT/TPOT/ITL
TEST_URL = f"http://localhost:{PORT}/generate"
TEST_STREAM_URL = f"http://localhost:{PORT}/generate_stream"
TEST_NUM_REQUESTS = 8        # total requests to send
TEST_CONCURRENCY = 4         # how many are in flight at the same time
TEST_PROMPT = "Tell me a short fact about space"
TEST_MAX_NEW_TOKENS = 50
TEST_TIMEOUT_S = 600         # per-request timeout
TEST_OUTPUT_FILE = "results.json"
TEST_MEMORY = True           # collect server memory metrics during the test
TEST_MEMORY_INTERVAL_S = 0.5 # how often to sample memory while requests run