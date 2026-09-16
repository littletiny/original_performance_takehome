"""Offline semantic kernel construction and scheduling experiments.

The submitted KernelBuilder does not import this development tool.  All scores
are measured with the unchanged tests/frozen_problem.py machine.
"""

from collections import Counter
from dataclasses import dataclass, field
import argparse
import importlib.util
import json
from pathlib import Path
import random
import time

import numpy as np

import perf_takehome as pt

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("frozen", ROOT / "tests/frozen_problem.py")
frozen = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = frozen
spec.loader.exec_module(frozen)
MASK = (1 << 32) - 1
ENGINES = ("alu", "valu", "load", "store", "flow")
CAPACITY = np.array([12, 6, 2, 2, 1], dtype=np.int64)


@dataclass
class Graph:
    ops: list = field(default_factory=list)
    names: list = field(default_factory=list)
    sizes: list = field(default_factory=list)
    # Each unit is a list of (op id, cycle relative to unit start).
    units: list = field(default_factory=list)
    regions: list = field(default_factory=list)
    checks: list = field(default_factory=list)
    pc_constants: list = field(default_factory=list)
    control: list = field(default_factory=list)
    lookup_bits: dict = field(default_factory=dict)
    initial_zero: list = field(default_factory=list)
    tag: tuple = (-1, 0)

    def new(self, n=8):
        v = pt.V(len(self.sizes))
        self.sizes.append(n)
        return v

    def emit(self, name, engine, slot, ins, outs):
        i = len(self.ops)
        self.ops.append([engine, tuple(slot), ins, outs, self.tag])
        self.names.append(name)
        self.units.append([(i, 0)])
        return i


def lane(v, j):
    return pt.V(v.vid, v.off + j)


def build(config=None):
    cfg = dict(jump3=0, jump4=32, jump15=32, jump5=0,
               gather3=0, gather14=0, blend4=0, blend15=0,
               leaf_madd=0.0, scalar=0.33, offset_scalar=False,
               prexor=True, fold=True, seed=0, prefetch3=False,
               prefetch4=False, width3=2, width4=2, width5=2,
               last_pair3=False, prefetch_madd=False, dce=True,
               path2_flow=False, fold_path4=False, fuse_tail_pc=False,
               load_vectors=(), synth_scalars=False, synth_vectors=False,
               small_bias_alu=False, path2_valu_groups=(), load_children=False,
               scalar_labels=None, scalar_extra="h2.b", scalar_extra_fraction=0.25)
    cfg.update(tail_gathers=0, load_child_budget=128, oldest_first=False, store_children=False, merge_regions=1, temp_buffers=0, pair_early_end=False, initial_ones=False)
    cfg.update(config or {})
    g = Graph()
    sc, vc = {}, {}
    scalar_tick = 0
    leaf_tick = 0

    def scalar(value):
        value &= MASK
        if value not in sc:
            v = g.new(1)
            expression = None
            if cfg["synth_scalars"] and value not in (2310, 7, 8, 32):
                for a, source in sc.items():
                    delta = (value-a) & MASK
                    if delta in sc:
                        expression = ("+", source, sc[delta])
                        break
                    delta = (a-value) & MASK
                    if delta in sc:
                        expression = ("-", source, sc[delta])
                        break
            if expression:
                code, a, b = expression
                g.emit(f"constant.{value}", "alu", (code, v, a, b), [(a, 1), (b, 1)], [(v, 1)])
            else:
                g.emit(f"constant.{value}", "load", ("const", v, value), [], [(v, 1)])
            sc[value] = v
        return sc[value]

    def bcast(src, name):
        v = g.new()
        g.emit(name, "valu", ("vbroadcast", v, src), [(src, 1)], [(v, 8)])
        return v

    def vector(value):
        value &= MASK
        if value not in vc:
            if value in cfg["load_vectors"]:
                dst = g.new()
                for j in range(8):
                    g.emit(f"vector.const.{value}.{j}", "load", ("const", lane(dst, j), value), [], [(lane(dst, j), 1)])
                vc[value] = dst
            else:
                expression = None
                if cfg["synth_vectors"]:
                    for a, source in vc.items():
                        delta = (value-a) & MASK
                        if delta in vc:
                            expression = ("+", source, vc[delta])
                            break
                        delta = (a-value) & MASK
                        if delta in vc:
                            expression = ("-", source, vc[delta])
                            break
                    if expression is None:
                        for a, source in vc.items():
                            if a and value % a == 0 and value // a in vc:
                                expression = ("*", source, vc[value//a])
                                break
                if expression:
                    code, a, b = expression
                    vc[value] = binary(code, a, b, f"derive.{value}", False)
                else:
                    vc[value] = bcast(scalar(value), f"broadcast.{value}")
        return vc[value]

    def binary(op, a, b, name, flexible=True):
        nonlocal scalar_tick
        dst = g.new()
        scalar_tick += bool(flexible)
        fraction = cfg["scalar"]
        if isinstance(fraction, list):
            fraction = fraction[max(0, g.tag[0])]
        offload = flexible and int(scalar_tick * fraction) != int((scalar_tick - 1) * fraction)
        if flexible and cfg["scalar_labels"] is not None and name.startswith("r"):
            label = ".".join(name.split(".")[2:])
            offload = label in cfg["scalar_labels"]
            index = g.tag[0] * 32 + g.tag[1]
            if label == cfg["scalar_extra"] and label not in cfg["scalar_labels"]:
                f = cfg["scalar_extra_fraction"]
                offload = int((index+1)*f) != int(index*f)
        if offload:
            for j in range(8):
                aa, bb, dd = lane(a, j), lane(b, j), lane(dst, j)
                g.emit(name + f".lane{j}", "alu", (op, dd, aa, bb), [(aa, 1), (bb, 1)], [(dd, 1)])
        else:
            g.emit(name, "valu", (op, dst, a, b), [(a, 8), (b, 8)], [(dst, 8)])
        return dst

    def madd(a, b, c, name):
        dst = g.new()
        g.emit(name, "valu", ("multiply_add", dst, a, b, c), [(a, 8), (b, 8), (c, 8)], [(dst, 8)])
        return dst

    def select(bit, yes, no, name):
        dst = g.new()
        g.emit(name, "flow", ("vselect", dst, bit, yes, no), [(bit, 8), (yes, 8), (no, 8)], [(dst, 8)])
        return dst

    def vload(addr, name, sync=()):
        dst = g.new()
        g.emit(name, "load", ("vload", dst, addr), [(addr, 1), *sync], [(dst, 8)])
        return dst

    if cfg["initial_ones"]:
        zero, one = g.new(), g.new()
        g.initial_zero.append(zero)
        g.emit("constant.ones", "valu", ("==", one, zero, zero), [(zero, 8)], [(one, 8)])
        vc[1] = one
        sc[1] = one

    modes = [["root" if r % 11 == 0 else "blend" if r % 11 <= 3 else "gather" for _ in range(32)] for r in range(16)]
    for r in (3, 4, 5, 14, 15):
        n = cfg["jump3" if r in (3, 14) else f"jump{r}"]
        assert n % 2 == 0
        for k in range(32 - n, 32):
            modes[r][k] = "jump"
    for r, key in ((3, "gather3"), (14, "gather14")):
        for k in range(cfg[key]):
            modes[r][k] = "gather"
    for r, key in ((4, "blend4"), (15, "blend15")):
        for k in range(cfg[key]):
            modes[r][31 - k] = "blend"
    for k in range(32-cfg["tail_gathers"], 32):
        modes[14][k] = modes[15][k] = "gather"
    for r in (3, 4, 14):
        if cfg[f"prefetch{r % 11}"]:
            for k in range(32):
                if modes[r][k] == "jump":
                    modes[r+1][k] = "prefetch"

    dispatch_groups = {}
    for r in (3, 4, 5, 14, 15):
        k = 0
        while k < 32:
            if modes[r][k] != "jump":
                k += 1
                continue
            width = cfg[f"width{r % 11}"]
            if r == 14 and k >= 28 and cfg["last_pair3"]:
                width = 2
            if r == 3 and k >= 28 and cfg["pair_early_end"]:
                width = 2
            width = min(width, 32-k)
            remaining = next((j-k for j in range(k, 32) if modes[r][j] != "jump"), 32-k)
            if width == 3 and remaining == 4:
                width = 4
            width = min(width, remaining)
            assert all(modes[r][j] == "jump" for j in range(k, k+width))
            dispatch_groups[r, k] = width
            k += width

    # Each level's adjusted runtime data is prepared once.  Extra memory uses
    # the specified all-zero input index area and is restored before exit.
    c = [stage[1] for stage in frozen.HASH_STAGES]
    bases = {d: 7 + (1 << d) - 1 for d in range(11)}
    if cfg["prexor"]:
        bases.update({d: 2054 + (1 << d) - 16 for d in range(4, 8)})
        if cfg["tail_gathers"]:
            bases[3] = 2294
    syncs = {}
    raw_nodes, adjusted_nodes = {}, {}
    needed = {0}
    for r in range(16):
        if any(mode in ("blend", "jump", "prefetch") for mode in modes[r]):
            needed.add(r % 11)
    if cfg["prexor"]:
        needed.update(range(4, 8))
    if cfg["synth_scalars"]:
        for value in (1, 2, 4, 8, 32):
            scalar(value)
    for d in sorted(needed):
        raw_nodes[d], adjusted_nodes[d] = [], []
        for off in range(0, 1 << d, 8):
            raw = vload(scalar(7 + (1 << d) - 1 + off), f"tree.d{d}.raw{off}")
            if d <= 2 and cfg["small_bias_alu"]:
                bias = g.new()
                src = scalar(c[5])
                for j in range(1 << d):
                    g.emit(f"tree.d{d}.bias{j}", "alu", ("^", lane(bias, j), lane(raw, j), src), [(lane(raw, j), 1), (src, 1)], [(lane(bias, j), 1)])
            else:
                bias = binary("^", raw, vector(c[5]), f"tree.d{d}.bias{off}", False)
            raw_nodes[d].extend(lane(raw, j) for j in range(min(8, (1 << d) - off)))
            adjusted_nodes[d].extend(lane(bias, j) for j in range(min(8, (1 << d) - off)))
            if cfg["prexor"] and (4 <= d <= 7 or d == 3 and cfg["tail_gathers"]):
                ptr = scalar(bases[d] + off)
                store = g.emit(f"tree.d{d}.store{off}", "store", ("vstore", ptr, bias), [(ptr, 1), (bias, 8)], [])
                syncs.setdefault(d, []).append(store)
    root0 = bcast(raw_nodes[0][0], "root.raw")
    root1 = bcast(adjusted_nodes[0][0], "root.bias")
    child_diffs = {}
    if cfg["prefetch_madd"]:
        for d in (4, 5):
            if d not in needed:
                continue
            children = list(reversed(adjusted_nodes[d]))
            child_diffs[d] = []
            for j in range(0, len(children), 2):
                dst = g.new(1)
                g.emit(f"child.diff.d{d}.{j//2}", "alu", ("-", dst, children[j+1], children[j]), [(children[j+1], 1), (children[j], 1)], [(dst, 1)])
                child_diffs[d].append(dst)
    tables, diffs = {}, {}
    for d in sorted(needed - {0}):
        if not any("blend" in modes[r] for r in range(16) if r % 11 == d):
            continue
        tables[d] = [bcast(x, f"table.d{d}.n{i}") for i, x in enumerate(reversed(adjusted_nodes[d]))]
        if cfg["oldest_first"]:
            tables[d] = [tables[d][int(f"{i:0{d}b}"[::-1], 2)] for i in range(1 << d)]
        if cfg["leaf_madd"]:
            diffs[d] = [binary("-", tables[d][2*i+1], tables[d][2*i], f"diff.d{d}.n{i}", False) for i in range(1 << (d - 1))]

    def blend(d, bits, prefix):
        nonlocal leaf_tick
        level = tables[d]
        order = bits[-d:] if cfg["oldest_first"] else reversed(bits[-d:])
        for step, bit in enumerate(order):
            result = []
            for pair in range(len(level) // 2):
                fraction = cfg["leaf_madd"]
                leaf_tick += 1
                use_madd = step == 0 and fraction and int(leaf_tick * fraction) != int((leaf_tick - 1) * fraction)
                name = prefix + f"select{step}.{pair}"
                if use_madd:
                    value = madd(bit, diffs[d][pair], level[2*pair], name)
                else:
                    value = select(bit, level[2*pair+1], level[2*pair], name)
                result.append(value)
            level = result
        assert len(level) == 1
        return level[0]

    # Each table has dense one-bundle cases.  Offsets for all eight lanes are
    # useful because each case processes the same lane in two vector groups.
    region_counts = Counter((r % 11, width) for (r, k), width in dispatch_groups.items())
    table_positions, total_words = {}, 0
    for key in sorted(region_counts):
        d, width = key
        table_positions[key] = total_words
        total_words += region_counts[key] * 8 * (1 << (width*d))
    next_region = Counter()
    prev_offsets = {}

    prefetched = {}
    previous_temp_loads = {}
    temp_reads = []
    temp_output_reads = {}
    possible_child_loads = sum(8*(2 if width >= 3 else 1) for (r, group), width in dispatch_groups.items() if r == 14)
    child_load_counter = 0

    def dispatch(r, group, values, pointers):
        nonlocal child_load_counter
        d = r % 11
        n = 1 << d
        width = dispatch_groups[r, group]
        key = (d, width)
        cases = n ** width
        prefix = f"r{r}.g{group}.dispatch."
        region_number = next_region[key]
        next_region[key] += 1
        table = table_positions[key] + region_number * 8 * cases
        if region_number == 0:
            offsets = g.new()
            for j in range(8):
                op = g.emit(prefix + f"offset{j}", "load", ("const", lane(offsets, j), table + j*cases), [], [(lane(offsets, j), 1)])
                g.pc_constants.append(op)
        else:
            offsets = binary("+", prev_offsets[key], vector(8*cases), prefix + "offsets", cfg["offset_scalar"])
        prev_offsets[key] = offsets
        fused = r == 14 and cfg["fuse_tail_pc"]
        if fused:
            targets = offsets
            for stream in range(width):
                targets = madd(pointers[group+stream], vector(2*n**(width-1-stream)), targets, prefix+f"prefix{stream}")
            pairs = []
            for stream in range(0, width, 2):
                a = bits[group+stream][-1]
                if stream+1 == width:
                    pairs.append((a, 1))
                else:
                    b = bits[group+stream+1][-1]
                    hi = select(b, vector(n+1), vector(n), prefix+f"bits_high{stream}")
                    pairs.append((select(a, hi, b, prefix+f"bits_pair{stream}"), 2))
            packed_bits = pairs[0][0]
            for stream, (pair, size) in enumerate(pairs[1:], 1):
                packed_bits = madd(packed_bits, vector(n**size), pair, prefix+f"bits_pack{stream}")
            targets = binary("+", targets, packed_bits, prefix+"targets", False)
        else:
            packed = pointers[group]
            for stream in range(1, width):
                packed = madd(packed, vector(n), pointers[group+stream], prefix + f"pack{stream}")
            targets = binary("+", packed, offsets, prefix + "targets", False)
        mixed = [g.new() for _ in range(width)]
        fetch = d in (3, 4) and cfg[f"prefetch{d}"]
        if fetch:
            children = list(reversed(adjusted_nodes[d+1]))
            store_children = cfg["store_children"] and r in (3, 14)
            if store_children:
                buffer = len(g.regions) % cfg["temp_buffers"] if cfg["temp_buffers"] else int(r == 14)
                temp_base = 2310 + 16*buffer
                temp_ptrs = [scalar(temp_base+j) for j in range(16)]
            memory_children = None
            if r == 14 and cfg["load_children"]:
                assert cfg["prexor"] and not cfg["prefetch_madd"]
                memory_children = [scalar(bases[d+1]+j) for j in reversed(range(1 << (d+1)))]
            for stream in range(width):
                prefetched[r+1, group+stream] = [g.new(), g.new()]
        start_unit = len(g.units)
        start = g.emit(prefix + "jump0", "flow", ("jump_indirect", targets),
                       [(targets, 8), *[(values[group+s], 8) for s in range(width)], *[(x, 1) for x in adjusted_nodes[d]],
                        *([(x, 1) for x in children] if fetch else []),
                        *([(x, 1) for x in child_diffs[d+1]] if fetch and cfg["prefetch_madd"] else [])], [])
        if fetch and store_children:
            g.control.extend((op, start, -1) for op in previous_temp_loads.get(buffer, ()))
            for source_group in (2*buffer, 2*buffer+1):
                g.control.append((input_loads[source_group], start, 1))
        parts = []
        temp_stores = [[], []]
        for j in range(8):
            xors = []
            for stream in range(width):
                dst, src, q = lane(mixed[stream], j), lane(values[group+stream], j), lane(pointers[group+stream], j)
                op = g.emit(prefix + f"xor{j}.{stream}", "alu", ("lookup_xor", dst, src, q, *reversed(adjusted_nodes[d])),
                            [(src, 1), (q, 1), *[(x, 1) for x in adjusted_nodes[d]]], [(dst, 1)])
                if fused:
                    g.lookup_bits[op] = lane(bits[group+stream][-1], j)
                    g.ops[op][2].append((g.lookup_bits[op], 1))
                xors.append((op, stream))
                if fetch:
                    for child in range(2):
                        cache = children[child::2] if child == 0 or not cfg["prefetch_madd"] else child_diffs[d+1]
                        dst = lane(prefetched[r+1, group+stream][child], j)
                        load_child = memory_children is not None and (2*stream+child+2*j) % (2*width) < (2 if width >= 3 else 1)
                        if load_child:
                            fraction = min(1, cfg["load_child_budget"] / possible_child_loads)
                            load_child = int((child_load_counter+1)*fraction) != int(child_load_counter*fraction)
                            child_load_counter += 1
                        engine, code = ("load", "lookup_load") if load_child else ("alu", "lookup_copy")
                        if load_child:
                            cache = memory_children[child::2]
                        store_child = store_children and stream == 0
                        if store_child:
                            dst = temp_ptrs[child*8+j]
                            engine, code = "store", "lookup_store"
                        op = g.emit(prefix + f"child{j}.{stream}.{child}", engine, (code, dst, q, *cache),
                                    [(q, 1), *[(x, 1) for x in cache], *([(dst, 1)] if store_child else [])], [] if store_child else [(dst, 1)])
                        if store_child:
                            temp_stores[child].append(op)
                        if load_child:
                            last_gathers.append(op)
                            g.control.extend((store, start, 1) for store in syncs[d+1])
                        if fused:
                            g.lookup_bits[op] = lane(bits[group+stream][-1], j)
                            g.ops[op][2].append((g.lookup_bits[op], 1))
                        xors.append((op, stream))
            slot = ("jump_indirect", lane(targets, j+1)) if j < 7 else ("jump", 0)
            jump = g.emit(prefix + f"jump{j+1}", "flow", slot, [(slot[1], 1)] if j < 7 else [], [])
            parts.append((xors, jump))
        g.units[start_unit:] = [[(start, 0), *[(op, j+1) for j, (xors, jump) in enumerate(parts) for op in [*(op for op, stream in xors), jump]]]]
        g.regions.append(dict(start=start, parts=parts, n=n, width=width, cases=cases, table=table, groups=list(range(group, group+width)), round=r))
        if fetch and store_children:
            loads = []
            for child in range(2):
                dst = prefetched[r+1, group][child]
                addr = temp_ptrs[child*8]
                op = g.emit(prefix+f"child_vector{child}", "load", ("vload", dst, addr), [(addr, 1)], [(dst, 8)])
                g.control.extend((before, op, 1) for before in temp_stores[child])
                loads.append(op)
            previous_temp_loads[buffer] = loads
            temp_reads.extend(loads)
            for output_group in (2*buffer, 2*buffer+1):
                temp_output_reads.setdefault(output_group, []).extend(loads)
            g.regions[-1]["temp_loads"] = loads
        return mixed

    values = []
    io = []
    input_loads = []
    for k in range(32):
        g.tag = (-1, k)
        ptr = scalar(2310 + 8*k)
        io.append(ptr)
        values.append(vload(ptr, f"g{k}.input"))
        input_loads.append(len(g.ops)-1)
    ptrs = [None] * 32
    state = [None] * 32
    bits = [[] for _ in range(32)]
    last_gathers = []
    gather_levels = {}
    for r in range(16):
        depth = r % 11
        old_values = values.copy()
        mixed_pairs = {}
        for k in range(32):
            g.tag = (r, k)
            prefix = f"r{r}.g{k}."
            mode = modes[r][k]
            if mode == "root":
                bits[k] = []
                value = binary("^", values[k], root0 if r == 0 else root1, prefix + "mix")
            elif mode == "blend":
                node = blend(depth, bits[k], prefix)
                value = binary("^", values[k], node, prefix + "mix")
            elif mode == "jump":
                if k not in mixed_pairs:
                    assert all(state[j] == "q" for j in range(k, k+dispatch_groups[r, k]))
                    both = dispatch(r, k, old_values, ptrs)
                    mixed_pairs.update((k+s, v) for s, v in enumerate(both))
                value = mixed_pairs[k]
            elif mode == "prefetch":
                no, yes = prefetched[r, k]
                if cfg["prefetch_madd"]:
                    node = madd(bits[k][-1], yes, no, prefix + "prefetched_node")
                else:
                    node = select(bits[k][-1], yes, no, prefix + "prefetched_node")
                value = binary("^", values[k], node, prefix + "mix")
            else:
                address = ptrs[k]
                assert state[k] == "a"
                node = g.new()
                deps = syncs.get(depth, [])
                if depth <= 3 and not deps:
                    # Rare shallow gathers use raw nodes, so absorb the
                    # previous round's deferred XOR after the load.
                    bias_gather = True
                else:
                    bias_gather = False
                for j in range(8):
                    op = g.emit(prefix + f"load{j}", "load", ("load", lane(node, j), lane(address, j)), [(lane(address, j), 1)], [(lane(node, j), 1)])
                    g.control.extend((before, op, 1) for before in deps)
                    if deps:
                        last_gathers.append(op)
                        gather_levels[op] = depth
                if bias_gather:
                    node = binary("^", node, vector(c[5]), prefix + "load.bias")
                value = binary("^", values[k], node, prefix + "mix")
            defer = r != 15 and ((r+1) % 11 <= 3 or modes[r+1][k] in ("blend", "jump", "prefetch") or cfg["prexor"] and 4 <= (r+1)%11 <= 7)

            def checkpoint(value, stage, biased=False):
                g.checks.append((value, r, k, stage, c[5] if biased else 0))

            value = madd(value, vector(4097), vector(c[0]), prefix + "h1")
            checkpoint(value, 0)
            a = binary("^", value, vector(c[1]), prefix + "h2.a")
            b = binary(">>", value, vector(19), prefix + "h2.b")
            value = binary("^", a, b, prefix + "h2")
            checkpoint(value, 1)
            a = madd(value, vector(33), vector(c[2] + c[3]), prefix + "h4.a")
            b = madd(value, vector(33 << 9), vector(c[2] << 9), prefix + "h4.b")
            value = binary("^", a, b, prefix + "h4")
            checkpoint(value, 3)
            value = madd(value, vector(9), vector(c[4]), prefix + "h5")
            checkpoint(value, 4)
            shifted = binary(">>", value, vector(16), prefix + "h6.b")
            a = value if defer else binary("^", value, vector(c[5]), prefix + "h6.a")
            value = binary("^", a, shifted, prefix + "h6")
            checkpoint(value, 5, defer)
            values[k] = value
            if r == 15 or depth == 10:
                continue
            bit = binary("&", value, vector(1), prefix + "bit")
            bits[k].append(bit)
            next_state = "a" if modes[r+1][k] == "gather" else "q"
            if depth == 0:
                assert defer and next_state == "q"
                ptrs[k], state[k] = bit, "q"
                continue
            if r == 13 and modes[14][k] == "jump" and cfg["fuse_tail_pc"]:
                pass  # q2 and b13 are consumed separately by the PC builder.
            elif depth == 3 and modes[r+1][k] == "prefetch" and cfg["fold_path4"]:
                pass  # q3, b3 and b4 will directly form the depth-5 address.
            elif depth == 4 and modes[r][k] == "prefetch" and cfg["fold_path4"]:
                assert defer
                if next_state == "q":
                    hi = select(bit, vector(3), vector(2), prefix+"address.high")
                    aux = select(bits[k][-2], hi, bit, prefix+"address.aux")
                    ptrs[k] = madd(ptrs[k], vector(4), aux, prefix+"address")
                else:
                    base = bases[5] + 31
                    hi = select(bit, vector(base-3), vector(base-2), prefix+"address.high")
                    lo = select(bit, vector(base-1), vector(base), prefix+"address.low")
                    aux = select(bits[k][-2], hi, lo, prefix+"address.aux")
                    ptrs[k] = madd(ptrs[k], vector(-4), aux, prefix+"address")
            elif depth == 1 and cfg["path2_flow"] and (r, k) not in set(tuple(x) for x in cfg["path2_valu_groups"]) and state[k] == next_state == "q" and defer:
                hi = select(bit, vector(3), vector(2), prefix+"path.high")
                ptrs[k] = select(ptrs[k], hi, bit, prefix+"path")
            elif state[k] == "q" and next_state == "q" and defer:
                ptrs[k] = madd(ptrs[k], vector(2), bit, prefix + "path")
            else:
                # Convert the current coordinate to the next level directly.
                # All bias and base constants become the two select choices.
                if next_state == "q":
                    scale = 2 if state[k] == "q" else -2
                    const = 0 if state[k] == "q" else 2*(bases[depth] + (1 << depth)-1)
                    zero, one = (const, const+1) if defer else (const+1, const)
                else:
                    scale = -2 if state[k] == "q" else 2
                    const = bases[depth+1] + 2*((1 << depth)-1) if state[k] == "q" else bases[depth+1]-2*bases[depth]
                    zero, one = (const+1, const) if defer else (const, const+1)
                aux = select(bit, vector(one), vector(zero), prefix + "address.aux")
                ptrs[k] = madd(ptrs[k], vector(scale), aux, prefix + "address")
            state[k] = next_state
    g.tag = (16, 0)
    for k in range(32):
        op = g.emit(f"g{k}.output", "store", ("vstore", io[k], values[k]), [(io[k], 1), (values[k], 8)], [])
        g.control.extend((before, op, 0) for before in temp_output_reads.get(k, ()))
    # Restore all temporary index words, ordered after every reading gather.
    if cfg["prexor"]:
        for off in range(0, 248 if cfg["tail_gathers"] else 240, 8):
            addr = scalar(2054 + off)
            op = g.emit(f"restore.{off}", "store", ("vstore", addr, vector(0)), [(addr, 1), (vector(0), 8)], [])
            depth = 3 if off >= 240 else (off+16).bit_length()-1
            g.control.extend((before, op, 1) for before in syncs.get(depth, ()))
            g.control.extend((before, op, 1) for before in last_gathers if gather_levels.get(before, 4) == depth)
    g.config = cfg
    g.total_table_words = total_words
    if cfg["dce"]:
        eliminate_dead(g)
    if cfg["merge_regions"] > 1:
        merge_dispatch_regions(g, cfg["merge_regions"])
    return g


def eliminate_dead(g):
    writers = {int(base)+j: i for i, op in enumerate(g.ops) for base, size in op[3] for j in range(size)}
    parents = [set() for _ in g.ops]
    for i, op in enumerate(g.ops):
        parents[i].update(writers[int(base)+j] for base, size in op[2] for j in range(size) if int(base)+j in writers)
    for before, after, lag in g.control:
        parents[after].add(before)
    keep = {i for i, op in enumerate(g.ops) if op[0] in ("store", "flow")}
    # Hash checkpoints intentionally preserve all semantically computed words.
    keep.update(writers[int(v)+j] for v, *_ in g.checks for j in range(8))
    pending = list(keep)
    while pending:
        for p in parents[pending.pop()]:
            if p not in keep:
                keep.add(p)
                pending.append(p)
    ids = {old: new for new, old in enumerate(sorted(keep))}
    g.ops = [g.ops[i] for i in sorted(keep)]
    g.names = [g.names[i] for i in sorted(keep)]
    g.units = [[(ids[i], off) for i, off in unit if i in keep] for unit in g.units]
    g.units = [unit for unit in g.units if unit]
    g.control = [(ids[a], ids[b], lag) for a, b, lag in g.control if a in keep and b in keep]
    g.pc_constants = [ids[i] for i in g.pc_constants if i in keep]
    g.lookup_bits = {ids[i]: bit for i, bit in g.lookup_bits.items() if i in keep}
    for r in g.regions:
        r["start"] = ids[r["start"]]
        r["parts"] = [([(ids[i], stream) for i, stream in slots], ids[jump]) for slots, jump in r["parts"]]
        r["temp_loads"] = [ids[i] for i in r.get("temp_loads", ())]


def merge_dispatch_regions(g, number):
    """One region's last handler directly enters its neighbour's first case."""
    assert 2 <= number <= 4 and g.config["store_children"]
    unit_of = {i:u for u, rows in enumerate(g.units) for i, off in rows}
    aliases, removed_units, replacement_units = {}, set(), {}
    for rnd in sorted({r["round"] for r in g.regions}):
        regions = [r for r in g.regions if r["round"] == rnd]
        for start in range(0, len(regions), number):
            batch = regions[start:start+number]
            if len(batch) == 1:
                continue
            combined = []
            first_unit = unit_of[batch[0]["start"]]
            for j, region in enumerate(batch):
                unit = unit_of[region["start"]]
                removed_units.add(unit)
                for i, off in g.units[unit]:
                    if j and i == region["start"]:
                        continue
                    combined.append((i, off+j*8))
                if j+1 < len(batch):
                    region["chain_exit"] = True
                    next_start = batch[j+1]["start"]
                    previous_exit = region["parts"][-1][1]
                    aliases[next_start] = previous_exit
                    g.ops[previous_exit] = [*g.ops[next_start][:4], g.ops[previous_exit][4]]
                    for i in region.get("temp_loads", ()):
                        removed_units.add(unit_of[i])
                        combined.append((i, j*8+9))
            replacement_units[first_unit] = combined
    units = []
    for u, rows in enumerate(g.units):
        if u in replacement_units:
            units.append(replacement_units[u])
        elif u not in removed_units:
            units.append(rows)
    keep = [i for i in range(len(g.ops)) if i not in aliases]
    ids = {old:new for new, old in enumerate(keep)}
    def mapped(i): return ids[aliases.get(i, i)]
    g.ops = [g.ops[i] for i in keep]
    g.names = [g.names[i] for i in keep]
    g.units = [[(mapped(i), off) for i, off in rows] for rows in units]
    g.control = [(mapped(a), mapped(b), lag) for a, b, lag in g.control]
    g.pc_constants = [mapped(i) for i in g.pc_constants]
    g.lookup_bits = {mapped(i):bit for i, bit in g.lookup_bits.items()}
    for r in g.regions:
        r["start"] = mapped(r["start"])
        r["parts"] = [([(mapped(i), stream) for i, stream in rows], mapped(jump)) for rows, jump in r["parts"]]
        r["temp_loads"] = [mapped(i) for i in r.get("temp_loads", ())]


def counts(g):
    c = Counter(op[0] for op in g.ops)
    return dict(engines=dict(c), weighted_alu_valu=c["alu"]+8*c["valu"],
                bound=max((c[e]+int(cap)-1)//int(cap) for e, cap in zip(ENGINES, CAPACITY)),
                arithmetic_bound=(c["alu"]+8*c["valu"]+59)//60,
                table_bundles=g.total_table_words, regions=len(g.regions), values=len(g.sizes))


class Scheduler:
    def __init__(self, g):
        import ctypes
        import subprocess
        lib_path = ROOT / ".schedule.so"
        if not lib_path.exists() or lib_path.stat().st_mtime < (ROOT / "schedule.cpp").stat().st_mtime:
            subprocess.run(["g++", "-O3", "-std=c++17", "-shared", "-fPIC", str(ROOT/"schedule.cpp"), "-o", str(lib_path)], check=True)
        self.lib = ctypes.CDLL(str(lib_path))
        ptr = np.ctypeslib.ndpointer(dtype=np.int64, flags="C_CONTIGUOUS")
        fptr = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
        self.lib.schedule_search.argtypes = [ctypes.c_int, ctypes.c_int, ptr, ptr, ptr, ptr, ptr, fptr,
                                            ctypes.c_int, ctypes.c_uint64, ctypes.c_double, ptr, ptr]
        self.lib.schedule_search.restype = ctypes.c_int
        self.g = g
        n = len(g.units)
        units = [None] * len(g.ops)
        self.usage = np.zeros((n, 33, 5), dtype=np.int64)
        self.durations = np.zeros(n, dtype=np.int64)
        self.tags = np.zeros((n, 2), dtype=np.int64)
        for u, ops in enumerate(g.units):
            self.tags[u] = g.ops[ops[0][0]][4]
            for i, off in ops:
                units[i] = (u, off)
                self.usage[u, off, ENGINES.index(g.ops[i][0])] += 1
                self.durations[u] = max(self.durations[u], off)
        assert np.all(self.usage <= CAPACITY)
        self.op_units = np.array([row[0] for row in units])
        self.op_offsets = np.array([row[1] for row in units])
        edges = {}
        writers = {int(base)+j:i for i, op in enumerate(g.ops) for base, size in op[3] for j in range(size)}

        def edge(before, after, lag):
            a, ao = units[before]
            b, bo = units[after]
            if a != b:
                edges[a, b] = max(edges.get((a, b), -999), lag + ao - bo)
            else:
                assert bo >= ao + lag, (g.names[before], g.names[after], lag, ao, bo)

        for i, op in enumerate(g.ops):
            for base, size in op[2]:
                for j in range(size):
                    if int(base)+j in writers:
                        edge(writers[int(base)+j], i, 1)
                    else:
                        assert base.vid in {v.vid for v in g.initial_zero}
        for before, after, lag in g.control:
            edge(before, after, lag)
        self.sources = np.array([a for a, b in edges], dtype=np.int64)
        self.dests = np.array([b for a, b in edges], dtype=np.int64)
        self.lags = np.array(list(edges.values()), dtype=np.int64)
        self.tail = np.zeros(n, dtype=np.int64)
        children = [[] for _ in range(n)]
        for (a, b), lag in edges.items():
            children[a].append((b, lag))
        pending = np.bincount(self.dests, minlength=n).tolist()
        ready = [u for u in range(n) if pending[u] == 0]
        order = []
        while ready:
            u = ready.pop()
            order.append(u)
            for child, lag in children[u]:
                pending[child] -= 1
                if not pending[child]: ready.append(child)
        assert len(order) == n, "Cyclic compound schedule"
        self.order = order
        for a in reversed(order):
            self.tail[a] = max([self.durations[a], *[lag+self.tail[b] for b, lag in children[a]]])
        self.children = children

    def consumer_setup(self, keys):
        keys = np.array(keys, dtype=np.float64)
        for u in reversed(self.order):
            name = self.g.names[self.g.units[u][0][0]]
            if name.startswith(("constant.", "broadcast.", "derive.", "tree.", "root.", "table.", "diff.", "vector.const.")) and self.children[u]:
                keys[u] = min(keys[child]-lag for child, lag in self.children[u])
        return keys

    def search(self, keys=None, iterations=100, seed=0, noise=1.0):
        n = len(self.g.units)
        if keys is None:
            keys = self.consumer_setup(self.tags[:, 0] * 2.0 + self.tags[:, 1])
        keys = np.ascontiguousarray(keys, dtype=np.float64)
        best = np.empty(n, dtype=np.int64)
        current = np.empty(n, dtype=np.int64)
        score = self.lib.schedule_search(n, len(self.lags), self.sources, self.dests, self.lags,
            self.durations, self.usage, keys, iterations, seed, noise, best, current)
        assert score > 0, score
        assert np.all(best[self.dests] >= best[self.sources] + self.lags)
        return score, best, current

    def op_times(self, times):
        return times[self.op_units] + self.op_offsets


def allocate(g, times, policy=0):
    import bisect
    first, last_read, last_write = {}, {}, {}
    for v in g.initial_zero:
        first[v.vid] = last_write[v.vid] = -1
    for i, op in enumerate(g.ops):
        t = int(times[i])
        for base, length in op[3]:
            first[base.vid] = min(first.get(base.vid, t), t)
            last_write[base.vid] = max(last_write.get(base.vid, t), t)
        for base, length in op[2]:
            last_read[base.vid] = max(last_read.get(base.vid, t), t)
    end = {v: max(last_read.get(v, -1), last_write[v]+1) for v in first}
    order = sorted(first, key=lambda v: (first[v], -g.sizes[v] if policy < 6 else g.sizes[v], end[v] if policy % 2 == 0 else -end[v], v))
    free = [(0, 1536)]
    active = []
    bases = {}
    peak = 0
    for v in order:
        start, size = first[v], g.sizes[v]
        keep = []
        for until, addr, sz in active:
            if until <= start:
                bisect.insort(free, (addr, addr+sz))
            else:
                keep.append((until, addr, sz))
        active = keep
        merged = []
        for a, b in free:
            if merged and a == merged[-1][1]:
                merged[-1] = (merged[-1][0], b)
            else:
                merged.append((a, b))
        free = merged
        choices = []
        for j, (a, b) in enumerate(free):
            if b-a < size:
                continue
            if policy % 6 in (2, 3) and size == 1:
                key = -b
            elif policy % 6 in (4, 5) and size == 1:
                key = a
            elif policy >= 6 and size >= 8:
                key = a
            else:
                key = b-a
            choices.append((key, j))
        if not choices:
            failed = dict(cycle=start, live=sum(sz for _, _, sz in active), requested=size, largest_hole=max((b-a for a, b in free), default=0))
            if policy == 0:
                for alternative in range(1, 12):
                    solution, report = allocate(g, times, alternative)
                    if solution is not None:
                        return solution, report
            return None, failed
        _, j = min(choices)
        a, b = free.pop(j)
        addr = a if size >= 8 else b-size
        if b-a > size:
            bisect.insort(free, (a+size, b) if size >= 8 else (a, b-size))
        bases[v] = addr
        active.append((end[v], addr, size))
        peak = max(peak, addr+size)
    return bases, dict(scratch_words=peak, allocation_policy=policy)


def lower(g, times, bases):
    """Expand dense cases only after all scratch and schedule checks pass."""
    total = int(max(times)) + 1

    def word(v):
        return bases[v.vid] + v.off if isinstance(v, pt.V) else v

    slots = [tuple(word(v) for v in op[1]) for op in g.ops]
    for i in g.pc_constants:
        code, dst, immediate = slots[i]
        slots[i] = (code, dst, immediate+total)
    logical = [{} for _ in range(total)]
    for i, op in enumerate(g.ops):
        logical[int(times[i])].setdefault(op[0], []).append(slots[i])
    # Falling off main code must skip the out-of-line case bodies.
    # A final jump is packed with a last store if its FLOW slot is free.
    assert not logical[-1].get("flow")
    logical[-1]["flow"] = [("jump", total + g.total_table_words)]
    program = list(logical) + [{} for _ in range(g.total_table_words)]
    origins = np.full(len(program), -1, dtype=np.int64)
    origins[:total] = np.arange(total)
    for region in g.regions:
        entry = int(times[region["start"]])
        n, width, cases = region["n"], region["width"], region["cases"]
        for part, (lookups, jump) in enumerate(region["parts"]):
            cycle = int(times[jump])
            assert cycle == entry+part+1
            pos = total + region["table"] + part*cases
            replacements = {slots[i]: (i, stream) for i, stream in lookups}
            for choice in range(cases):
                indices = [(choice//(n**(width-1-stream))) % n for stream in range(width)]
                bundle = {}
                for engine, instructions in logical[cycle].items():
                    out = []
                    for slot in instructions:
                        if slot in replacements:
                            i, stream = replacements[slot]
                            q = indices[stream]
                            if slot[0] == "lookup_xor":
                                slot = ("^", slot[1], slot[2], slot[4+q])
                            elif slot[0] == "lookup_load":
                                slot = ("load", slot[1], slot[3+q])
                            elif slot[0] == "lookup_store":
                                slot = ("store", slot[1], slot[3+q])
                            else:
                                assert slot[0] == "lookup_copy"
                                slot = ("|", slot[1], slot[3+q], slot[3+q])
                        elif engine == "flow" and slot == slots[jump] and part == 7 and not region.get("chain_exit"):
                            slot = ("jump", cycle+1)
                        out.append(slot)
                    bundle[engine] = out
                program[pos+choice] = bundle
                origins[pos+choice] = cycle
            program[cycle] = {}
            origins[cycle] = -1
    assert len(program) < 500_000, len(program)
    return program, origins, logical


def verify_frozen(g, times, bases, program, origins, seeds=(0, 1)):
    checks = {}
    last_write = {}
    for i, op in enumerate(g.ops):
        for base, size in op[3]:
            for j in range(size): last_write[int(base)+j] = int(times[i])
    for v, r, k, stage, bias in g.checks:
        for j in range(8):
            cycle = last_write[int(v)+j]
            checks.setdefault(cycle, []).append((bases[v.vid]+v.off+j, (r, k*8+j, "hash_stage", stage), bias))

    class Checked(frozen.Machine):
        def step(self, instr, core):
            pc = core.pc - 1
            logical_cycle = int(origins[pc])
            assert logical_cycle == self.cycle, (pc, logical_cycle, self.cycle)
            super().step(instr, core)
            for addr, key, bias in checks.get(logical_cycle, []):
                assert core.scratch[addr] ^ bias == self.value_trace[key], (self.cycle, addr, key, core.scratch[addr]^bias, self.value_trace[key])

    for seed in seeds:
        random.seed(seed)
        tree = frozen.Tree.generate(10)
        inp = frozen.Input.generate(tree, 256, 16)
        mem = frozen.build_mem_image(tree, inp)
        trace = {}
        for expected in frozen.reference_kernel2(mem.copy(), trace): pass
        machine = Checked(mem, program, frozen.DebugInfo({}), value_trace=trace)
        machine.enable_debug = False
        machine.enable_pause = False
        machine.run()
        assert machine.cycle == int(max(times))+1
        assert machine.mem[2310:] == expected[2310:]
        assert machine.mem[:2310] == mem[:2310]
    return dict(cycles=machine.cycle, seeds=list(seeds), checkpoints_per_seed=len(g.checks)*8,
                static_bundles=len(program), static_slots=sum(len(slots) for b in program for slots in b.values()))


def search_config(cfg, directory, trials=100, iterations=100, seed=0):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    g = build(cfg)
    print(json.dumps(dict(config=g.config, **counts(g))), flush=True)
    verify_semantics(g, (0,))
    s = Scheduler(g)
    rng = random.Random(seed)
    best_score, best_feasible = 100000, 100000
    incumbent = None
    start = time.monotonic()
    for trial in range(trials):
        if incumbent is not None and trial % 4:
            keys = incumbent.copy()
        else:
            ka = rng.choice([0.5, 1, 2, 4, 8, 12])
            kb = rng.choice([0.5, 1, 2, 4, 8])
            keys = s.tags[:, 0] * ka + s.tags[:, 1] * kb
            keys = keys - s.tail * rng.choice([0, .05, .2, 1.0])
        noise = rng.choice([.001, .1, .5, 1, 2, 4, 8, 16, 32])
        score, best, last = s.search(keys, iterations, rng.randrange(1<<32), noise)
        if score < best_score:
            best_score = score
            incumbent = best
            np.savez_compressed(directory/"best_ir.npz", times=s.op_times(best), unit_times=best)
        elif rng.random() < .10:
            incumbent = last
        if score >= best_feasible:
            continue
        op_times = s.op_times(best)
        bases, allocation = allocate(g, op_times)
        row = dict(trial=trial, cycles=score, seconds=time.monotonic()-start, **allocation)
        if bases is not None:
            best_feasible = score
            np.savez_compressed(directory/"best.npz", times=op_times, unit_times=best)
            (directory/"config.json").write_text(json.dumps(g.config, indent=2)+"\n")
            (directory/"result.json").write_text(json.dumps(row, indent=2)+"\n")
        print(json.dumps(row), flush=True)
    return best_feasible


def verify_semantics(g, seeds=(0, 1)):
    for seed in seeds:
        random.seed(seed)
        tree = frozen.Tree.generate(10)
        inp = frozen.Input.generate(tree, 256, 16)
        mem = frozen.build_mem_image(tree, inp)
        initial = mem.copy()
        trace = {}
        for ref in frozen.reference_kernel2(mem.copy(), trace):
            pass
        values = {int(v)+j:0 for v in g.initial_zero for j in range(g.sizes[v.vid])}

        def binary(op, a, b):
            if op == "+": return (a+b)&MASK
            if op == "-": return (a-b)&MASK
            if op == "*": return a*b&MASK
            if op == "^": return a^b
            if op == "&": return a&b
            if op == ">>": return a>>b
            if op == "==": return int(a == b)
            raise ValueError(op)

        for op_id, (engine, slot, *_) in enumerate(g.ops):
            op, *args = slot
            if op == "lookup_store":
                dst, q, *cache = args
                index = values[int(q)]
                if op_id in g.lookup_bits:
                    index = 2*index + values[int(g.lookup_bits[op_id])]
                mem[values[int(dst)]] = values[int(cache[index])]
            elif op == "lookup_load":
                dst, q, *cache = args
                index = values[int(q)]
                if op_id in g.lookup_bits:
                    index = 2*index + values[int(g.lookup_bits[op_id])]
                values[int(dst)] = mem[values[int(cache[index])]]
            elif engine == "load":
                dst = args[0]
                if op == "const": values[int(dst)] = args[1]
                elif op == "load": values[int(dst)] = mem[values[int(args[1])]]
                elif op == "vload":
                    addr = values[int(args[1])]
                    for j in range(8): values[int(dst)+j] = mem[addr+j]
            elif op == "lookup_xor":
                dst, src, q, *cache = args
                index = values[int(q)]
                if op_id in g.lookup_bits:
                    index = 2*index + values[int(g.lookup_bits[op_id])]
                assert 0 <= index < len(cache)
                values[int(dst)] = values[int(src)] ^ values[int(cache[index])]
            elif op == "lookup_copy":
                dst, q, *cache = args
                index = values[int(q)]
                if op_id in g.lookup_bits:
                    index = 2*index + values[int(g.lookup_bits[op_id])]
                assert 0 <= index < len(cache)
                values[int(dst)] = values[int(cache[index])]
            elif engine in ("alu", "valu"):
                dst = args[0]
                for j in range(8 if engine == "valu" else 1):
                    if op == "vbroadcast": val = values[int(args[1])]
                    elif op == "multiply_add": val = (values[int(args[1])+j]*values[int(args[2])+j]+values[int(args[3])+j])&MASK
                    else: val = binary(op, values[int(args[1])+j], values[int(args[2])+j])
                    values[int(dst)+j] = val
            elif op == "vselect":
                dst, bit, yes, no = args
                for j in range(8): values[int(dst)+j] = values[int(yes if values[int(bit)+j] else no)+j]
            elif engine == "store":
                addr, src = args
                for j in range(8): mem[values[int(addr)]+j] = values[int(src)+j]
            else:
                assert op in ("jump", "jump_indirect")
        for value, r, k, stage, bias in g.checks:
            for j in range(8):
                actual = values[int(value)+j] ^ bias
                expected = trace[r, 8*k+j, "hash_stage", stage]
                assert actual == expected, (seed, r, k, stage, j, actual, expected)
        assert mem[2310:] == ref[2310:]
        assert mem[:2310] == initial[:2310]
    return dict(seeds=list(seeds), hash_checkpoints_per_seed=len(g.checks)*8, preserved_non_output_memory=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="{}")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--search", type=Path)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.search:
        search_config(json.loads(args.config), args.search, args.trials, args.iterations, args.seed)
    else:
        graph = build(json.loads(args.config))
        print(json.dumps(counts(graph)))
        if args.verify:
            print(json.dumps(verify_semantics(graph)))
