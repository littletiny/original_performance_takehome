"""Profile the submitted kernel's executed VLIW bundles on the frozen machine.

Usage: python3 plot_utilization.py [--seed 0] [--output utilization_profile.png]

Writes a PNG, a JSON summary, and a per-cycle CSV. All utilization denominators
use measured dynamic cycles, including setup and cleanup. Vector instructions
occupy one port slot; debug instructions are excluded. No saved schedule or
historical round-activity data is used.
"""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "tests"))
from frozen_problem import (
    Input, Machine, N_CORES, SLOT_LIMITS, Tree, build_mem_image, reference_kernel2,
)
from perf_takehome import KernelBuilder

ENGINES = ("alu", "valu", "load", "store", "flow")
COLORS = ("#2878b5", "#df504b", "#23976b", "#ba8436", "#8554bc")


class ProfileMachine(Machine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rows = []
        self.opcodes = {engine: Counter() for engine in ENGINES}

    def step(self, instr, core):
        # Match Machine.run's cycle accounting, including non-debug no-ops.
        if any(engine != "debug" for engine in instr):
            assert core.id == 0 and self.cycle == len(self.rows)
            self.rows.append((self.cycle, core.pc - 1, *(
                len(instr.get(engine, ())) for engine in ENGINES
            )))
            for engine in ENGINES:
                self.opcodes[engine].update(slot[0] for slot in instr.get(engine, ()))
        return super().step(instr, core)


def profile(seed):
    assert N_CORES == 1, "This report profiles the single-core workload."
    random.seed(seed)
    forest = Tree.generate(10)
    inputs = Input.generate(forest, 256, 16)
    memory = build_mem_image(forest, inputs)
    builder = KernelBuilder()
    builder.build_kernel(10, 2047, 256, 16)
    machine = ProfileMachine(memory, builder.instrs, builder.debug_info(), n_cores=1)
    machine.enable_pause = False
    machine.enable_debug = False
    machine.run()

    for expected in reference_kernel2(memory.copy()):
        pass
    output_start = memory[6]
    output_end = output_start + len(inputs.values)
    assert machine.mem[output_start:output_end] == expected[output_start:output_end]
    assert machine.mem[:output_start] == memory[:output_start]
    assert machine.mem[output_end:] == memory[output_end:]
    assert machine.cycle == len(machine.rows)

    rows = np.asarray(machine.rows, dtype=np.int64)
    counts = rows[:, 2:]
    capacities = np.asarray([SLOT_LIMITS[engine] for engine in ENGINES])
    assert np.all((counts >= 0) & (counts <= capacities))
    cycles = machine.cycle
    ports = {}
    for i, engine in enumerate(ENGINES):
        used = int(counts[:, i].sum())
        available = cycles * SLOT_LIMITS[engine]
        ports[engine] = {
            "slots_per_cycle": SLOT_LIMITS[engine],
            "used_slots": used,
            "available_slots": available,
            "idle_slots": available - used,
            "utilization_percent": 100 * used / available,
            "full_cycles": int(np.count_nonzero(counts[:, i] == capacities[i])),
            "active_cycles": int(np.count_nonzero(counts[:, i])),
            "opcodes": dict(sorted(machine.opcodes[engine].items())),
        }
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "perf_takehome.py",
        "source_sha256": hashlib.sha256((ROOT / "perf_takehome.py").read_bytes()).hexdigest(),
        "simulator": "tests/frozen_problem.py",
        "workload": {"forest_height": 10, "n_nodes": 2047, "batch_size": 256, "rounds": 16},
        "seed": seed,
        "cycles": cycles,
        "static_bundles": len(builder.instrs),
        "static_slot_operations": sum(
            len(slots) for bundle in builder.instrs for engine, slots in bundle.items()
            if engine != "debug"
        ),
        "dynamic_slot_operations": int(counts.sum()),
        "output_verified": True,
        "non_output_memory_preserved": True,
        "ports": ports,
        "notes": [
            "Counts are from the actual executed path, not the static dispatch tables.",
            "Utilization = occupied slots / (dynamic cycles * engine slots per cycle).",
            "Setup, cleanup, pause, and jumps are included; debug instructions are excluded.",
            "Each vector instruction counts as one slot, not eight scalar lane operations.",
        ],
    }
    return rows, summary


def render(rows, summary, output):
    cycles = summary["cycles"]
    capacities = np.asarray([SLOT_LIMITS[engine] for engine in ENGINES])
    utilization = rows[:, 2:] / capacities * 100
    mean = utilization.mean(axis=0)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig = plt.figure(figsize=(15, 10.5), facecolor="#fbfcfe")
    grid = fig.add_gridspec(3, 2, width_ratios=[1, 0.018], height_ratios=[1.1, 1.6, 1.15])
    bars = fig.add_subplot(grid[0, 0])
    lines = fig.add_subplot(grid[1, 0])
    heat = fig.add_subplot(grid[2, 0], sharex=lines)
    color_axis = fig.add_subplot(grid[2, 1])
    fig.suptitle(f"Port utilization | {cycles:,} dynamic cycles", x=0.06, ha="left",
                 fontsize=21, fontweight="bold", color="#192d44")
    fig.text(0.06, 0.929,
             f"Current submitted kernel  |  {summary['static_bundles']:,} static VLIW bundles"
             f"  |  seed {summary['seed']}  |  frozen-machine output verified",
             color="#52647a", fontsize=11)

    positions = np.arange(len(ENGINES))
    bars.barh(positions, [100] * len(ENGINES), color="#e8edf4", height=0.62)
    bars.barh(positions, mean, color=COLORS, height=0.62)
    for i, engine in enumerate(ENGINES):
        port = summary["ports"][engine]
        bars.text(2, i, f"{mean[i]:.2f}%", va="center", color="white", fontweight="bold")
        bars.text(102, i, f"{port['used_slots']:,} / {port['available_slots']:,} slots"
                  f"    |    {port['idle_slots']:,} idle", va="center", fontsize=9,
                  color="#34485f")
    bars.set_yticks(positions, [f"{engine.upper()}  ({SLOT_LIMITS[engine]}/cycle)" for engine in ENGINES])
    bars.invert_yaxis()
    bars.set_xlim(0, 137)
    bars.set_xticks([])
    bars.tick_params(axis="y", length=0)
    bars.set_title("Average occupied port slots over the complete run", loc="left", fontsize=12, pad=12)

    window = min(15, cycles)
    weights = np.ones(window)
    edge_counts = np.convolve(np.ones(cycles), weights, mode="same")
    for i, engine in enumerate(ENGINES):
        smoothed = np.convolve(utilization[:, i], weights, mode="same") / edge_counts
        lines.plot(rows[:, 0], smoothed, color=COLORS[i], lw=1.6, label=engine.upper(), alpha=0.92)
    lines.axhline(100, color="#52647a", lw=0.7, ls="--", alpha=0.5)
    lines.set_ylim(0, 105)
    lines.set_xlim(-0.5, cycles - 0.5)
    lines.set_ylabel("Slot utilization (%)")
    lines.set_title(f"Utilization through time | {window}-cycle moving average", loc="left", fontsize=12, pad=36)
    lines.legend(ncol=5, loc="lower left", bbox_to_anchor=(0, 1.005), frameon=False)
    lines.grid(alpha=0.18)
    lines.tick_params(labelbottom=False)

    im = heat.imshow(utilization.T, aspect="auto", cmap="inferno", vmin=0, vmax=100,
                     origin="upper", interpolation="nearest",
                     extent=[-0.5, cycles - 0.5, len(ENGINES) - 0.5, -0.5])
    heat.set_yticks(positions, [engine.upper() for engine in ENGINES])
    heat.set_xlabel("Executed cycle (zero-based)")
    heat.set_title("Every executed cycle | dark = idle, bright = full", loc="left", fontsize=12, pad=12)
    colorbar = fig.colorbar(im, cax=color_axis, ticks=[0, 25, 50, 75, 100])
    colorbar.ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    for axis in (bars, lines, heat):
        axis.set_facecolor("#fbfcfe")
        for spine in axis.spines.values():
            spine.set_visible(False)
    fig.text(0.06, 0.022,
             "Actual runtime path; all setup, cleanup and FLOW instructions included. "
             "One vector instruction occupies one slot.\n"
             f"Shape: height 10 / 2,047 nodes / 256 inputs / 16 rounds. "
             f"Source SHA-256: {summary['source_sha256'][:16]}",
             fontsize=9, color="#52647a", linespacing=1.7)
    fig.subplots_adjust(left=0.12, right=0.94, bottom=0.11, top=0.865, hspace=0.48, wspace=0.025)
    fig.savefig(output, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=ROOT / "utilization_profile.png")
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    rows, summary = profile(args.seed)
    render(rows, summary, output)
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    with output.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["cycle", "pc", *ENGINES])
        writer.writerows(rows.tolist())
    print(json.dumps({"image": str(output), "cycles": summary["cycles"],
                      "static_bundles": summary["static_bundles"],
                      "ports": summary["ports"]}, indent=2))


if __name__ == "__main__":
    main()
