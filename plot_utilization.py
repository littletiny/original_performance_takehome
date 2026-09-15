"""Per-cycle engine utilization + round-activity profile of the 985-cycle kernel.

Usage: python3 plot_utilization.py  ->  writes utilization_profile.png

Panels:
  1. per-engine slot-utilization lines (raw + 15-cycle moving average)
  2. engine x cycle utilization heatmap, with a labeled colorbar
  3. round x cycle activity strip (which rounds are being processed when),
     regenerated from the shipped embedded schedule; data in
     round_activity_985.npz (regenerate: rebuild ops from commit 60729b7's
     DAG with EMBEDDED_SCHEDULES cleared, map tags through the winning
     basin's opcycle; HEAD bundle index == schedule cycle).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
from matplotlib.colors import LinearSegmentedColormap

import perf_takehome as pt
from problem import SLOT_LIMITS

kb = pt.KernelBuilder()
kb.build_kernel(10, 2047, 256, 16)
T = len(kb.instrs)
engines = ["valu", "alu", "load", "flow", "store"]
colors = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#8c564b"]
U = np.zeros((len(engines), T))
for t, b in enumerate(kb.instrs):
    for eng, slots in b.items():
        if eng != "debug":
            U[engines.index(eng), t] = len(slots) / SLOT_LIMITS[eng]

d = np.load("round_activity_985.npz")
RA = d["RA"]  # rows 0..15 = rounds, 16 = final stores; per-cycle op counts

fig, (ax1, ax2, ax3) = plt.subplots(
    3, 1, figsize=(15, 11), sharex=True,
    gridspec_kw={"height_ratios": [2.2, 1.5, 1.8]})
x = np.arange(T)

# ---- panel 1: utilization lines ----
W = 15
kernel = np.ones(W) / W
for i, e in enumerate(engines):
    sm = np.convolve(U[i], kernel, mode="same")
    ax1.plot(x, U[i] * 100, color=colors[i], alpha=0.12, lw=0.5)
    ax1.plot(x, sm * 100, color=colors[i], lw=1.8,
             label=f"{e} (avg {U[i].mean()*100:.1f}%)")
ax1.axhline(100, color="k", lw=0.6, ls="--", alpha=0.5)
ax1.set_ylabel("slot utilization (%)")
ax1.set_ylim(0, 112)
ax1.legend(loc="center right", fontsize=9)
ax1.set_title(f"perf_takehome {T}-cycle kernel: engines, colors and rounds")
ax1.grid(alpha=0.25)

# round-entry markers on panel 1 (15-cycle moving sum of round ops >= 12)
entries = {}
for h in range(16):
    ms = np.convolve(RA[h].astype(float), np.ones(W), mode="same")
    idx = np.where(ms >= 12)[0]
    if len(idx):
        entries[h] = int(idx[0])
for j, (h, c) in enumerate(entries.items()):
    ax1.axvline(c, color="gray", lw=0.7, ls=":", alpha=0.7)
    ax1.text(c, 103 + (j % 2) * 4, f"r{h}", rotation=90, fontsize=8,
             va="bottom", ha="right", color="dimgray")
ax1.text(500, 3, "gray dotted lines: cycle where each round enters the pipeline",
         fontsize=8, color="dimgray", ha="center")

# ---- panel 2: engine heatmap with labeled colorbar ----
im = ax2.imshow(U * 100, aspect="auto", cmap="inferno", vmin=0, vmax=100,
                origin="lower", extent=[0, T, -0.5, len(engines) - 0.5],
                interpolation="nearest")
ax2.set_yticks(range(len(engines)))
ax2.set_yticklabels(engines)
ax2.set_ylabel("engine")
cb = fig.colorbar(im, ax=ax2, pad=0.01, ticks=[0, 25, 50, 75, 100])
cb.ax.set_yticklabels(
    ["0% 黑=空闲", "25% 紫", "50% 红", "75% 橙", "100% 黄=打满"], fontsize=8)
ax2.set_title("engine x cycle heatmap (color = slot fill ratio)", fontsize=10)

# ---- panel 3: round activity strip ----
labels = [f"r{h}" for h in range(16)] + ["store"]
im3 = ax3.imshow(RA, aspect="auto", cmap="viridis", vmin=0, vmax=16,
                 origin="lower", extent=[0, T, -0.5, 16.5],
                 interpolation="nearest")
ax3.set_yticks(range(17))
ax3.set_yticklabels(labels, fontsize=8)
ax3.set_ylabel("round")
ax3.set_xlabel("cycle")
cb3 = fig.colorbar(im3, ax=ax3, pad=0.01, ticks=[0, 4, 8, 12, 16])
cb3.ax.set_yticklabels(["0 黑=无活动", "4", "8", "12", "16+ 黄=高峰"],
                       fontsize=8)
ax3.set_title("round x cycle activity (op count per cycle; overlap = software "
              "pipelining)", fontsize=10)
# per-row active range annotations
for h in range(16):
    idx = np.where(RA[h] > 0)[0]
    if len(idx):
        ax3.text(T + 8, h, f"{int(idx[0])}-{int(idx[-1])}", fontsize=7,
                 va="center", color="dimgray")
idx = np.where(RA[16] > 0)[0]
ax3.text(T + 8, 16, f"{int(idx[0])}-{int(idx[-1])}", fontsize=7, va="center",
         color="dimgray")
ax3.text(T + 8, 16.8, "active cycles:", fontsize=7, color="dimgray")

plt.tight_layout()
plt.savefig("utilization_profile.png", dpi=130, bbox_inches="tight")
print(f"saved utilization_profile.png ({T} cycles)")
