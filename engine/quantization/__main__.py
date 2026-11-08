"""
Offline quantization: build the quantized checkpoint ahead of time, without starting the server.

Run from the project root:
    python -m engine.quantization --method int8_wo
    python -m engine.quantization --method int4_wo --group-size 64
    python -m engine.quantization --method fp8 --targets experts
    python -m engine.quantization --list

Afterwards, start the server with the same settings (e.g. QUANT=int4_wo python main.py)
and it loads straight from the saved file.
"""
import argparse
import glob
import os
import sys
sys.path.insert(0, os.getcwd())
import torch  # noqa: E402
import config  # noqa: E402
from engine.quantization import METHODS, get_method  # noqa: E402
from engine.quantization.loader import cache_path, load_quantized_model, weight_memory  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Quantize the model and save it for later runs")
    parser.add_argument("--method", choices=list(METHODS))
    parser.add_argument("--targets", nargs="+", default=config.QUANT_TARGETS,
                        choices=["experts", "linear", "lm_head"])
    parser.add_argument("--group-size", type=int, default=config.QUANT_GROUP_SIZE)
    parser.add_argument("--force", action="store_true", help="re-quantize even if a saved file exists")
    parser.add_argument("--list", action="store_true", help="list saved quantized files")
    args = parser.parse_args()

    if args.list:
        files = sorted(glob.glob(os.path.join(config.QUANT_CACHE_DIR, "*.safetensors")))
        for f in files:
            print(f"{os.path.getsize(f) / 1024**3:6.2f} GB  {f}")
        print(f"{len(files)} file(s)" if files else "No saved quantized files yet.")
        return
    if not args.method:
        parser.error("--method is required (or use --list)")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    method = get_method(args.method, device=device, group_size=args.group_size,
                        fp8_backend=config.FP8_BACKEND)
    path = cache_path(config.QUANT_CACHE_DIR, config.MODEL_NAME, method, args.targets)
    if os.path.exists(path):
        if not args.force:
            print(f"Already exists: {path}  (use --force to redo it)")
            return
        os.remove(path)

    model, stats = load_quantized_model(config.MODEL_NAME, method, args.targets, device, torch.bfloat16,
                                        cache_dir=config.QUANT_CACHE_DIR, use_cache=True)
    print(f"\nQuantized {stats['experts']} expert blocks, {stats['linear']} linear layers, "
          f"lm_head={bool(stats['lm_head'])}, skipped={len(stats['skipped'])}")
    print(weight_memory(model))
    print(f"Saved to {path} ({os.path.getsize(path) / 1024**3:.2f} GB)")


if __name__ == "__main__":
    main()