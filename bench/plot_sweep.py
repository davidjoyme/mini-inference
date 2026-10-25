"""
bench/plot_sweep.py - turn a sweep's results into charts.

Run from the project root (needs matplotlib: pip install matplotlib):
    python bench/plot_sweep.py                       # latest folder in sweeps/
    python bench/plot_sweep.py sweeps/20261020_154435

Writes PNGs into <sweep folder>/plots/:
    <metric>.png           metric vs number of users.
                           Several engines in the sweep: one line per engine variant,
                           one row of panels per quantization.
                           One engine only: one line per quantization.
                           Columns: one panel per workload (output length x prompt length).
    memory_timeline.png    memory above idle over time during the busiest run of each line
"""
import csv
import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Validated categorical palette, used in fixed order (colour-blind safe on adjacent pairs).
# Every line also has its own marker and a text label, so colour never works alone.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
MARKERS = ["o", "s", "^", "D", "v", "P"]
QUANT_NAMES = {"none": "bf16", "int8_wo": "INT8", "int4_wo": "INT4", "fp8": "FP8"}
QUANT_ORDER = ["none", "int8_wo", "int4_wo", "fp8"]
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

# (column, title, unit). Charts with no data (e.g. GPU metrics on a CPU run) are skipped.
METRICS = [
    ("throughput_tok_s", "Throughput (all users)", "tokens/s"),
    ("ttft_p50_s", "Time to first token, p50", "s"),
    ("ttft_p95_s", "Time to first token, p95", "s"),
    ("tpot_p50_ms", "Time per output token, p50", "ms"),
    ("itl_p95_ms", "Gap between streamed tokens (inter-token latency), p95", "ms"),
    ("itl_max_ms", "Longest gap between streamed tokens (decode stalls)", "ms"),
    ("max_step_tokens", "Largest forward pass (tokens in one step)", "tokens"),
    ("peak_gpu_mb", "Peak GPU memory", "MB"),
    ("extra_gpu_mb", "GPU memory above idle (activations + KV cache)", "MB"),
    ("peak_rss_mb", "Peak server RAM", "MB"),
    ("batch_size_avg", "Average batch size (requests per forward pass)", "requests"),
    ("kv_reserved_pct_max", "KV cache handed out, peak", "% of the KV pool"),
    ("kv_waste_pct_avg", "KV cache handed out but holding nothing, average", "% of handed-out KV"),
    ("kv_fragmentation_pct_max", "Free KV memory unusable for a new request (fragmentation), peak", "%"),
    ("preemptions", "Preemptions (requests evicted and recomputed)", "count"),
    ("prefix_hit_pct", "Prompt tokens served from the prefix cache", "% of prompt tokens"),
    ("kv_capacity_tokens", "KV cache capacity", "tokens"),
    ("padding_waste_pct", "Compute spent on padding (static batching)", "% of decode tokens"),
    ("gpu_util_avg", "GPU utilization, average", "%"),
    ("gpu_bw_util_avg", "GPU memory-bandwidth utilization, average", "%"),
    ("power_avg_w", "GPU power, average", "W"),
    ("joules_per_token", "Energy per generated token", "J"),
    ("cpu_avg_pct", "Server CPU use, average", "% of one core"),
]


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def setup_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)


def label_ends(ax, ends, min_gap_px=12, near_px=70):
    """Direct labels at line ends, in text ink, nudged apart vertically when ends are close.
    Call after the axis limits are final."""
    to_px = ax.transData.transform
    items = sorted(([*to_px((x, y)), x, y, text] for x, y, text in ends), key=lambda i: i[1])
    for i, item in enumerate(items):
        item.append(item[1])
        for other in items[:i]:
            if abs(other[0] - item[0]) < near_px and item[5] - other[5] < min_gap_px:
                item[5] = other[5] + min_gap_px
    for x_px, y_px, x, y, text, target in items:
        ax.annotate(text, (x, y), xytext=(7, target - y_px), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK_2, annotation_clip=False)


TITLE_IN, LEGEND_IN = 0.45, 0.35       # vertical space above the panels for title and legend


def label_room_in(layout):
    """Inches needed to the right of a panel for the longest direct label."""
    longest = max(len(layout.name(layout.line_key, v)) for v in layout.lines)
    return 0.15 + 0.068 * longest


def finish_figure(fig, axes_list, all_ends, title, layout, height_in, width_in, extra_right=0.0):
    """Title on top, legend on its own line below it, room on the right of every panel for the
    direct labels, then place the labels (after the layout is final)."""
    room = label_room_in(layout)
    top = 1 - (TITLE_IN + LEGEND_IN) / height_in
    fig.tight_layout(rect=(0, 0, 1 - (room + extra_right) / width_in, top),
                     w_pad=room * 72 / 10 + 1.0)
    for ax, ends in all_ends:
        label_ends(ax, ends)
    handles, labels = [], []
    for ax in axes_list:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in labels:
                handles.append(h)
                labels.append(lab)
    fig.suptitle(title, x=0.01, y=1 - 0.12 / height_in, ha="left", va="top", color=INK,
                 fontsize=12, fontweight="bold")
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.005, 1 - TITLE_IN / height_in),
               ncol=len(labels), frameon=False, fontsize=9, labelcolor=INK, handlelength=2.2,
               columnspacing=1.4)


class Layout:
    """Which column is a line and which is a row of panels, plus a fixed style per line."""

    def __init__(self, rows):
        engines = list(dict.fromkeys(r["engine"] for r in rows))
        quants = [q for q in QUANT_ORDER if any(r["quant"] == q for r in rows)] + \
                 sorted({r["quant"] for r in rows} - set(QUANT_ORDER))
        if len(engines) > 1:
            self.line_key, self.lines, self.row_key, self.panel_rows = "engine", engines, "quant", quants
        else:
            self.line_key, self.lines, self.row_key, self.panel_rows = "quant", quants, "engine", engines
        if len(self.lines) > len(PALETTE):
            sys.exit(f"{len(self.lines)} lines per chart is too many to read; "
                     f"plot at most {len(PALETTE)} engine variants at once.")
        self.style = {v: (PALETTE[i], MARKERS[i]) for i, v in enumerate(self.lines)}

    def name(self, key, value):
        return QUANT_NAMES.get(value, value) if key == "quant" else value

    def row_title(self, value):
        return self.name(self.row_key, value)


def draw_line(ax, layout, value, xs, ys, markers=True):
    color, marker = layout.style[value]
    label = layout.name(layout.line_key, value)
    ax.plot(xs, ys, color=color, linewidth=2, marker=marker if markers else None, markersize=7,
            markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
    return (xs[-1], ys[-1], label)


def plot_metric(rows, layout, col, title, unit, out_path):
    data = [r for r in rows if num(r.get(col)) is not None]
    if not data:
        return False
    workloads = sorted({(int(r["out_tok"]), int(r["in_words"])) for r in rows})
    n_rows, n_cols = len(layout.panel_rows), len(workloads)
    room = label_room_in(layout)
    width_in = (4.4 + room) * n_cols + 0.6
    height_in = 3.2 * n_rows + TITLE_IN + LEGEND_IN + 0.3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(width_in, height_in), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    top = (max(num(r[col]) for r in data) * 1.1) or 1
    users = sorted({int(r["users"]) for r in rows})
    all_ends = []

    for i, row_value in enumerate(layout.panel_rows):
        for j, (out_tok, in_words) in enumerate(workloads):
            ax = axes[i][j]
            setup_axes(ax)
            ends = []
            for line in layout.lines:
                pts = sorted((int(r["users"]), num(r[col])) for r in data
                             if r[layout.line_key] == line and r[layout.row_key] == row_value
                             and int(r["out_tok"]) == out_tok and int(r["in_words"]) == in_words)
                if pts:
                    ends.append(draw_line(ax, layout, line, [p[0] for p in pts], [p[1] for p in pts]))
            all_ends.append((ax, ends))
            ax.set_xticks(users)
            ax.set_ylim(0, top)                     # same scale in every panel
            ax.margins(x=0.08)
            if i == n_rows - 1:
                ax.set_xlabel("simultaneous users", color=INK_2, fontsize=9)
            workload = f"{out_tok} output tokens, " + (f"+{in_words} prompt words" if in_words else "short prompt")
            heading = workload if n_rows == 1 else f"{layout.row_title(row_value)} · {workload}"
            ax.set_title(heading, color=INK_2, fontsize=9.5)
            if j == 0:
                ax.set_ylabel(unit, color=INK_2, fontsize=9)
            else:
                ax.tick_params(labelleft=False)

    finish_figure(fig, [ax for row in axes for ax in row], all_ends, title, layout, height_in, width_in)
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return True


def plot_timeline(sweep_dir, rows, layout, out_path):
    """Memory above idle over time, for the busiest run of each line, one panel per row value.
    Shows what the requests themselves add: activations and KV cache."""
    n = len(layout.panel_rows)
    room = label_room_in(layout)
    width_in, height_in = (6.4 + room) * n + 0.6, 3.4 + TITLE_IN + LEGEND_IN + 0.3
    fig, axes = plt.subplots(1, n, figsize=(width_in, height_in), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    unit_key, drew, all_ends = None, False, []
    for ax, row_value in zip(axes[0], layout.panel_rows):
        setup_axes(ax)
        ends, users = [], "?"
        for line in layout.lines:
            mine = [r for r in rows if r[layout.line_key] == line and r[layout.row_key] == row_value
                    and r.get("run_file")]
            if not mine:
                continue
            r = max(mine, key=lambda r: (int(r["users"]), int(r["in_words"]), int(r["out_tok"])))
            path = os.path.join(sweep_dir, r["run_file"])
            if not os.path.exists(path):
                continue
            data = json.load(open(path))
            timeline = data.get("memory_timeline", [])
            baseline = data.get("summary", {}).get("memory", {}).get("baseline", {})
            if not timeline:
                continue
            key = "gpu_allocated_mb" if timeline[0].get("gpu_allocated_mb") is not None else "process_rss_mb"
            unit_key = unit_key or key
            base = baseline.get(key) if baseline.get(key) is not None else timeline[0].get(key)
            pts = [(s["t_s"], s[key] - base) for s in timeline if s.get(key) is not None]
            if pts:
                ends.append(draw_line(ax, layout, line, [p[0] for p in pts], [p[1] for p in pts],
                                      markers=False))
                users, drew = r["users"], True
        all_ends.append((ax, ends))
        ax.axhline(0, color=INK_2, linewidth=0.8)
        ax.set_xlabel("seconds since the run started", color=INK_2, fontsize=9)
        ax.margins(x=0.02)
        prefix = f"{layout.row_title(row_value)} · " if n > 1 else ""
        ax.set_title(f"{prefix}busiest run ({users} simultaneous users)", color=INK_2, fontsize=9.5)
    if not drew:
        plt.close(fig)
        return False
    axes[0][0].set_ylabel("MB above idle", color=INK_2, fontsize=9)
    what = "GPU memory" if unit_key == "gpu_allocated_mb" else "Server RAM"
    finish_figure(fig, list(axes[0]), all_ends, f"{what} above idle while requests run",
                  layout, height_in, width_in)
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return True


def main():
    if len(sys.argv) > 1:
        sweep_dir = sys.argv[1]
    else:
        dirs = sorted(glob.glob(os.path.join(ROOT, "sweeps", "*", "summary.csv")))
        if not dirs:
            sys.exit("No sweeps found. Run bench/sweep.py first.")
        sweep_dir = os.path.dirname(dirs[-1])

    rows = [r for r in csv.DictReader(open(os.path.join(sweep_dir, "summary.csv")))]
    for r in rows:
        r.setdefault("engine", "hf")               # sweeps from before engine variants existed
        r["engine"] = r["engine"] or "hf"
    rows = [r for r in rows if num(r.get("throughput_tok_s")) is not None]   # skip failed runs
    if not rows:
        sys.exit("No successful runs in this sweep.")

    layout = Layout(rows)
    out_dir = os.path.join(sweep_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)
    made = []
    for col, title, unit in METRICS:
        path = os.path.join(out_dir, f"{col}.png")
        if plot_metric(rows, layout, col, title, unit, path):
            made.append(path)
    path = os.path.join(out_dir, "memory_timeline.png")
    if plot_timeline(sweep_dir, rows, layout, path):
        made.append(path)

    print(f"Wrote {len(made)} charts to {out_dir}:")
    for p in made:
        print("  " + os.path.basename(p))


if __name__ == "__main__":
    main()