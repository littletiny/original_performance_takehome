"""Account for arithmetic work in the verified 918-cycle graph."""
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from optimize import build, counts


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/compact_918"


def classify(name, op):
    suffix = re.sub(r"^r\d+\.g\d+\.", "", name)
    stem = re.sub(r"\.lane\d+$", "", suffix)
    if re.fullmatch(r"h[12456](?:\.[ab])?", stem):
        return "hash_internal", stem
    if stem == "mix" or suffix.startswith("dispatch.xor"):
        return "tree_node_mix", "xor"
    if stem == "bit":
        return "path_bits", "mask"
    if stem in ("path", "address"):
        return "path_updates", stem
    if suffix.startswith("dispatch."):
        if op[1][0] == "lookup_copy":
            kind = "grandchildren" if suffix.startswith("dispatch.grand") else "children"
            return "conditional_node_copies", kind
        for kind in ("pack", "targets", "offsets"):
            if suffix.startswith("dispatch." + kind):
                return "dispatch_addressing", kind
    if stem == "prefetched_node":
        return "node_selection_madd", "select"
    for prefix, kind in (
        ("constant.", "constant"),
        ("broadcast.", "broadcast"),
        ("tree.", "tree_bias"),
        ("root.", "root_broadcasts"),
        ("child.diff.", "child_differences"),
        ("table.", "table_broadcasts"),
        ("derive.", "derive"),
        ("restore.", "restore_unbias"),
    ):
        if name.startswith(prefix):
            return "setup_restoration", kind
    raise ValueError(f"Unclassified arithmetic operation: {name}: {op}")


def main():
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
    assert total["W"] == accounting["weighted_alu_valu"] == 54723
    for engine in ("alu", "valu"):
        assert total[engine] == accounting["engines"][engine]
    signature = json.dumps([graph.ops, graph.units, graph.control, graph.sizes], separators=(",", ":"))
    digest = hashlib.sha256(signature.encode()).hexdigest()
    bounds = json.loads((ROOT / "results/resource_windows_918/bounds.json").read_text())[0]
    assert bounds["graph_sha256"] == digest
    verification = json.loads((SOURCE / "verification.json").read_text())
    report = dict(
        source="results/compact_918",
        graph_sha256=digest,
        cycles=verification["cycles"],
        static_bundles=verification["static_bundles"],
        static_bundle_cap=12000,
        definition="W = ALU slots + 8 * VALU slots; throughput weight, not a FLOP count",
        arithmetic_capacity_per_cycle=60,
        fixed_graph_lower_bound=bounds["bound"],
        counts_only_minimum_reduction={str(c): max(0, total["W"] - 60*c) for c in (900, 899)},
        scope="Category sizes are current costs, not proven removable work. Savings from overlapping rewrites cannot be added.",
        total=dict(total),
        categories={key: dict(**row, details_W=dict(details[key])) for key, row in categories.items()},
    )
    output = Path(__file__).with_name("inventory.json")
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(cycles=report["cycles"], total=report["total"], categories=report["categories"]), indent=2))


if __name__ == "__main__":
    main()
