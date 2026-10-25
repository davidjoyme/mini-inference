"""
bench/eval_quality.py - how much does quantization change the model's outputs?

Two measurements:
  1. Perplexity on a piece of text (lower = the model predicts the text better).
     Compare across quantizations: a small rise is normal, a big jump means trouble.
  2. Greedy agreement with a reference run: the same prompts are answered with greedy
     decoding, and the answers are compared token by token with a saved bf16 run.

Run from the project root:
    # 1) reference run in bf16 (do this once)
    python bench/eval_quality.py --quant none --out quality_none.json

    # 2) each quantization, compared against the reference
    python bench/eval_quality.py --quant int8_wo --reference quality_none.json --out quality_int8.json
    python bench/eval_quality.py --quant int4_wo --reference quality_none.json --out quality_int4.json
    python bench/eval_quality.py --quant fp8     --reference quality_none.json --out quality_fp8.json

    # KV cache quantization (custom engine): reference with a bf16 cache, then fp8 / int8
    python bench/eval_quality.py --engine custom --out quality_kv_auto.json
    python bench/eval_quality.py --engine custom --kv-dtype fp8 --reference quality_kv_auto.json --out quality_kv_fp8.json

Use --text-file to measure perplexity on your own text (more text = more reliable numbers).
"""
import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

import config  # noqa: E402
from engine.factory import create_engine, engine_label  # noqa: E402

# A short built-in text so the script works without downloads. For more reliable
# perplexity numbers, pass a longer file with --text-file.
DEFAULT_TEXT = """
The water cycle moves water between the oceans, the air and the land. Heat from the sun
causes water at the surface of lakes and seas to evaporate, and the vapour rises until it
cools and condenses into the tiny droplets that make up clouds. When those droplets grow
heavy enough they fall as rain or snow. Some of that water soaks into the ground and
refills underground stores, some runs off into rivers, and some is taken up by plants,
which release it back into the air through their leaves.

A computer program is a list of instructions that a machine follows step by step. The
instructions are written in a programming language, which is easier for people to read
than the binary codes the processor understands. A compiler or an interpreter translates
the program into a form the machine can run. Good programs are split into small functions
that each do one job, so that they are easier to test, reuse and change later.

Bread is made from flour, water, salt and yeast. The yeast feeds on sugars in the flour and
produces gas, which is trapped by a stretchy network of proteins called gluten. Kneading
the dough strengthens that network, and letting it rest gives the yeast time to work. In
the oven, the gas expands, the dough rises further, and the outside browns into a crust.

Trains became common in the nineteenth century and changed how far people could travel in
a day. Towns along new railway lines grew quickly, goods could be moved in bulk over long
distances, and many places adopted shared time zones so that timetables would make sense.
Today, high speed lines connect large cities, and electric trains are seen as one of the
more efficient ways to move many people at once.
""".strip()

DEFAULT_PROMPTS = [
    "Explain what a hash table is in two sentences.",
    "List three causes of the French Revolution.",
    "Write a haiku about autumn leaves.",
    "What is 17 multiplied by 23? Show your working.",
    "Summarize the plot of a heist movie in one paragraph you invent yourself.",
    "Give me a Python function that reverses a string.",
    "What are the main differences between TCP and UDP?",
    "Suggest a name for a coffee shop that also sells books.",
]


@torch.no_grad()
def perplexity(engine, text, window=512):
    """Perplexity over the text in windows. The custom engine scores through its own KV cache
    (so KV_CACHE_DTYPE=fp8/int8 shows up here); the hf engine uses the plain model."""
    ids = engine.tokenizer(text, return_tensors="pt")["input_ids"][0]
    nll_sum, n_tokens = 0.0, 0
    for start in range(0, len(ids) - 1, window):
        chunk = ids[start:start + window + 1]
        if len(chunk) < 2:
            break
        if hasattr(engine, "score"):
            logits = engine.score(chunk.tolist())[:-1]
            targets = chunk[1:].to(logits.device)
            nll = torch.nn.functional.cross_entropy(logits.float(), targets, reduction="sum").item()
        else:
            model = engine.model
            c = chunk.unsqueeze(0).to(model.device)
            nll = model(input_ids=c, labels=c).loss.float().item() * (len(chunk) - 1)
        nll_sum += nll
        n_tokens += len(chunk) - 1
    return math.exp(nll_sum / n_tokens), n_tokens


def compare(outputs, reference):
    """Token-level agreement between this run and a reference run, prompt by prompt."""
    ref = {r["prompt"]: r for r in reference["generations"]}
    exact, prefix_fracs, first_diffs = 0, [], []
    for r in outputs:
        other = ref.get(r["prompt"])
        if other is None:
            continue
        a, b = r["token_ids"], other["token_ids"]
        same = 0
        for x, y in zip(a, b):
            if x != y:
                break
            same += 1
        if a == b:
            exact += 1
        else:
            first_diffs.append(same)
        prefix_fracs.append(same / max(len(b), 1))
    n = len(prefix_fracs)
    return {
        "compared_prompts": n,
        "exact_match_rate": round(exact / n, 3) if n else None,
        # fraction of the reference answer reproduced before the first different token
        "avg_matching_prefix": round(sum(prefix_fracs) / n, 3) if n else None,
        "avg_first_difference_at_token": round(sum(first_diffs) / len(first_diffs), 1) if first_diffs else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Measure output quality under quantization")
    parser.add_argument("--quant", default=None, choices=["none", "int8_wo", "int4_wo", "fp8"],
                        help="override QUANTIZATION from config.py")
    parser.add_argument("--engine", default=None, choices=["hf", "custom"])
    parser.add_argument("--scheduler", default=None, choices=["sequential", "static", "continuous"])
    parser.add_argument("--kv-cache", default=None, choices=["contiguous", "paged"])
    parser.add_argument("--kv-dtype", default=None, choices=["auto", "fp8", "int8"],
                        help="override KV_CACHE_DTYPE (custom engine)")
    parser.add_argument("--text-file", default=None, help="text for perplexity (default: built-in)")
    parser.add_argument("--window", type=int, default=512, help="tokens per perplexity window")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--reference", default=None, help="results JSON of a reference (bf16) run")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    for flag, name in ((args.quant, "QUANTIZATION"), (args.engine, "ENGINE"),
                       (args.scheduler, "SCHEDULER"), (args.kv_cache, "KV_CACHE")):
        if flag:
            setattr(config, name, flag)
    if args.kv_dtype:
        config.KV_CACHE_DTYPE = args.kv_dtype
    config.DO_SAMPLE = False   # greedy, so runs are comparable
    engine = create_engine()
    method = engine.info()["quantization"].get("method", "none")

    text = open(args.text_file).read() if args.text_file else DEFAULT_TEXT
    t0 = time.perf_counter()
    ppl, n_tok = perplexity(engine, text, args.window)
    print(f"Perplexity: {ppl:.3f} over {n_tok} tokens ({time.perf_counter() - t0:.1f}s)")

    generations = []
    for prompt in DEFAULT_PROMPTS:
        out = engine.generate(prompt, max_new_tokens=args.max_new_tokens)["output"]
        token_ids = engine.tokenizer(out, add_special_tokens=False)["input_ids"]
        generations.append({"prompt": prompt, "output": out, "token_ids": token_ids})
        print(f"- {prompt[:50]:<50} -> {out[:70]!r}")

    result = {
        "quantization": method,
        "engine_label": engine_label(engine.info()),
        "engine": engine.info(),
        "perplexity": round(ppl, 4),
        "perplexity_tokens": n_tok,
        "text_file": args.text_file,
        "generations": generations,
    }

    if args.reference:
        reference = json.load(open(args.reference))
        result["vs_reference"] = {"reference": args.reference,
                                  "reference_quantization": reference.get("quantization"),
                                  "reference_perplexity": reference.get("perplexity"),
                                  **compare(generations, reference)}
        if reference.get("perplexity"):
            result["vs_reference"]["perplexity_change_pct"] = round(
                (ppl / reference["perplexity"] - 1) * 100, 2)

    print("\n===== Quality =====")
    print(f"{'quantization':>30}: {method}")
    print(f"{'perplexity':>30}: {ppl:.4f}")
    for k, v in result.get("vs_reference", {}).items():
        print(f"{k:>30}: {v}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()