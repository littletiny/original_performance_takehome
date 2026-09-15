"""Per-cycle engine utilization profile of the built kernel.

Usage: python3 plot_utilization.py  ->  writes utilization_profile.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import perf_takehome as pt
from problem import SLOT_LIMITS

kb = pt.KernelBuilder()
kb.build_kernel(10, 2047, 256, 16)
bundles = kb.instrs
T = len(bundles)
engines = ["valu", "alu", "load", "flow", "store"]
colors = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#8c564b"]

U = np.zeros((len(engines), T))
for t, b in enumerate(bundles):
    for eng, slots in b.items():
        if eng == "debug":
            continue
        U[engines.index(eng), t] = len(slots) / SLOT_LIMITS[eng]

x = np.arange(T)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                               gridspec_kw={"height_ratios": [2, 1.2]})
W = 15
kernel = np.ones(W) / W
for i, e in enumerate(engines):
    sm = np.convolve(U[i], kernel, mode="same")
    ax1.plot(x, U[i] * 100, color=colors[i], alpha=0.12, lw=0.5)
    ax1.plot(x, sm * 100, color=colors[i], lw=1.8,
             label=f"{e} (avg {U[i].mean()*100:.1f}%)")
ax1.axhline(100, color="k", lw=0.6, ls="--", alpha=0.5)
ax1.set_ylabel("slot utilization (%)")
ax1.set_ylim(0, 108)
ax1.legend(loc="center right", fontsize=9)
ax1.set_title(f"perf_takehome {T}-cycle kernel: per-cycle engine utilization\n"
              "(faint = raw per-cycle, solid = 15-cycle moving average)")
ax1.grid(alpha=0.25)

ax2.imshow(U * 100, aspect="auto", cmap="inferno", vmin=0, vmax=100,
           origin="lower", extent=[0, T, -0.5, len(engines) - 0.5],
           interpolation="nearest")
ax2.set_yticks(range(len(engines)))
ax2.set_yticklabels(engines)
ax2.set_xlabel("cycle")
ax2.set_ylabel("engine")

for ax in (ax1, ax2):
    for xc in (80, 800, 960):
        ax.axvline(xc, color="steelblue", lw=0.8, ls=":", alpha=0.8)
ax1.text(40, 104, "ramp", ha="center", fontsize=9, color="steelblue")
ax1.text(500, 104, "steady state: valu/load/flow pinned at 100%", ha="center",
         fontsize=9, color="steelblue")
ax1.text(925, 104, "drain", ha="center", fontsize=9, color="steelblue")

plt.tight_layout()
plt.savefig("utilization_profile.png", dpi=130)
print(f"saved utilization_profile.png ({T} cycles)")
