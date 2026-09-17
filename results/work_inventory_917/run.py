"""Recount arithmetic work and bounds for the verified 917-cycle checkpoint."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import runpy

from optimize import build, counts
from resource_bounds import analyze


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/compact_917"


def main():
    # Keep category definitions identical to the preceding checkpoint's census.
    classify = runpy.run_path(str(ROOT / "results/work_inventory_918/run.py"))["classify"]
    graph = build(json.loads((SOURCE / "config.json").read_text()))
    categories = defaultdict(Counter)
    details = defaultdict(Counter)
    for name, op in zip(graph.names, graph.ops):
        if op[0] not in ("alu", "valu"):
            continue
        category, detail = classify(name, op)
        weight = 8 if op[0] == "valu" else 1
        categories[category][op[0]] += 1
        categories[category]["W"] += weight
        details[category][detail] += weight

    total = Counter()
    for row in categories.values():
        total.update(row)
    accounting = counts(graph)
    assert total["W"] == accounting["weighted_alu_valu"] == 54652
    for engine in ("alu", "valu"):
        assert total[engine] == accounting["engines"][engine]

    signature = json.dumps([graph.ops, graph.units, graph.control, graph.sizes], separators=(",", ":"))
    digest = hashlib.sha256(signature.encode()).hexdigest()
    bound = analyze(SOURCE)
    saved_bound = json.loads((SOURCE / "bounds.json").read_text())[0]
    assert digest == bound["graph_sha256"] == saved_bound["graph_sha256"]
    assert bound["resource_windows"] == saved_bound["resource_windows"]
    assert bound["bound"] == saved_bound["bound"] == 913
    verification = json.loads((SOURCE / "verification.json").read_text())
    validation = json.loads((SOURCE / "validation.json").read_text())
    source_digest = hashlib.sha256((ROOT / "perf_takehome.py").read_bytes()).hexdigest()
    assert source_digest == validation["source_sha256"]
    assert verification["cycles"] == bound["saved_schedule_cycles"] == 917
    assert verification["static_bundles"] == 10810

    final_checks = [check for check in graph.checks if check[3] == 5]
    deferred = sum(bool(check[4]) for check in final_checks)
    remaining = Counter(check[1] for check in final_checks if not check[4])
    assert len(final_checks) == 512 and deferred == 384
    assert len(graph.checks) * 8 == verification["checkpoints_per_seed"] == 20480
    report = dict(
        source="results/compact_917",
        graph_sha256=digest,
        production_source_sha256=source_digest,
        cycles=917,
        static_bundles=10810,
        static_bundle_cap=12000,
        definition="W = ALU slots + 8 * VALU slots; arithmetic-port work, not FLOPs or static bundles",
        arithmetic_capacity_per_cycle=60,
        fixed_graph_lower_bound=bound["bound"],
        counts_only_minimum_reduction={str(c): total["W"] - 60*c for c in (900, 899)},
        total=dict(total),
        categories={key: dict(**row, details_W=dict(details[key])) for key, row in categories.items()},
        hash_groups=dict(total=512, deferred_h6_constant=deferred, explicit_h6_constant_by_round=dict(remaining)),
        scope="Current category costs are not removable-work claims. Overlapping rewrites cannot be added; startup, tails and other ports still matter.",
    )
    Path(__file__).with_name("inventory.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(cycles=917, static_bundles=10810, W=total["W"], bound=913,
                          strict_sub900_minimum_reduction=report["counts_only_minimum_reduction"]["899"],
                          categories={key: row["W"] for key, row in categories.items()}), indent=2))


if __name__ == "__main__":
    main()
