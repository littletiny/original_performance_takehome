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
RESOURCE_CAPACITY = np.append(CAPACITY, 1)  # exclusive dispatch execution


class WideV(pt.V):
    """Research-only virtual spans; the real machine still has eight lanes."""
    STRIDE = 16

    def __new__(cls, vid, off=0):
        assert 0 <= off < cls.STRIDE
        obj = int.__new__(cls, pt.VA_BASE + vid*cls.STRIDE + off)
        obj.vid, obj.off = vid, off
        return obj

    def __reduce__(self):
        return type(self),(self.vid,self.off)


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
    ref_type: type = pt.V

    def new(self, n=8):
        assert 1 <= n <= getattr(self.ref_type, 'STRIDE', 8)
        v = self.ref_type(len(self.sizes))
        self.sizes.append(n)
        return v

    def emit(self, name, engine, slot, ins, outs):
        i = len(self.ops)
        self.ops.append([engine, tuple(slot), ins, outs, self.tag])
        self.names.append(name)
        self.units.append([(i, 0)])
        return i


def lane(v, j):
    return type(v)(v.vid, v.off + j)


def scalar_root_group(config,r,k):
    groups=config.get('scalar_root_groups',())
    return r in (0,11) and (groups=='all' or (r,k) in groups or [r,k] in groups)


def late_pair_member(config, k):
    return any(k in (group,group+1) for group in config.get('late_pair_groups',()))


def overfetch_member(config,r,k):
    groups=config.get('overfetch_groups',())
    return r%11 in config.get('overfetch_levels',(8,9,10)) and (groups=='all' or k in groups)


def forced_scalar_pack(config, r, k, label):
    return (label=='mix' and scalar_root_group(config,r,k) or
            label=='mix' and overfetch_member(config,r,k) or
            ((r==14 and label=='bit') or (r==15 and label=='mix')) and late_pair_member(config,k))


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
    cfg.update(tail_gathers=0, load_child_budget=128, oldest_first=False,
               store_children=False, merge_regions=1, temp_buffers=0,
               pair_early_end=False, initial_ones=False, dispatch_widths=(),
               compact_main=False, flow_constants=(), ahead_bits=(),
               prefetch5_groups=(), prefetch_madd_groups=(), fold_path4_groups=(),
               compact_heap=False, heap_keep_levels=(), heap_reuse_levels=(),
               heap_backup=False, heap_restore_window=0,
               output_address_block=0, output_address_window=32,
               alias_constants=(), lane_allocation=False,
               merge_chains=(), precise_dispatch_inputs=False,
               heap_backup_before_bias=False, share_shallow_loads=False,
               share_shallow_bias=False, lane_allocation_trials=8,
               scalar_overrides=None, dispatch_spans=(), load_temp_addresses=False,
               header_constants=False, initial_zero_vector=False, heap_io_backup=False,
               early_gather_groups=(), temp_region_order=(), pc_address_pools=False,
               dispatch5_groups=(), grand_row_groups=(), grand_row_buffers=3,
               dispatch4_groups=(), gather_dispatch4=False, precise_restore=False,
               header_root=False, scalar_root_groups=(), force_load_scalars=(),
               dispatch2_groups=(), dispatch13_groups=(), width2=3, prefetch2=True,
               fold_path3_groups=(), pc_bit_pools=(),
               late_pair_groups=(), late_pair_buffers=1, late_pair_order=(),
               memory_vectors=(), memory_vector_buffers=1, memory_vector_order=(),
               overfetch_groups=(), overfetch_levels=(8,9,10))
    cfg.update(config or {})
    late_pairs=set(cfg['late_pair_groups'])
    g = Graph(ref_type=WideV if late_pairs else pt.V)
    g.overfetch_values=[]
    if cfg['overfetch_groups']:
        assert cfg['compact_heap']
        assert set(cfg['overfetch_levels'])<={8,9,10}
        assert cfg['overfetch_groups']=='all' or set(cfg['overfetch_groups'])<=set(range(32))
    if late_pairs:
        assert cfg['compact_heap'] and cfg['store_children'] and cfg['prefetch3']
        assert cfg['heap_io_backup'] and not cfg['grand_row_groups']
        assert not cfg['dispatch_spans']
        assert 1 <= cfg['late_pair_buffers'] <= 4
    sc, vc = {}, {}
    # Fixed table addresses can share the scalar pointers already required by
    # tree loads and I/O. Allocate those words together before emitting them.
    address_vectors = {}
    address_lanes = {}
    if cfg["pc_address_pools"]:
        for base in (14, 78, 142, 206, 2318, 2382, 2446, 2510):
            value = g.new()
            address_vectors[base] = value
            address_lanes.update((base+8*j, lane(value,j)) for j in range(8))
    pc_bit_vectors={}
    if cfg['pc_bit_pools']:
        assert cfg['pc_address_pools']
        for base in cfg['pc_bit_pools']:
            assert base in address_vectors and base not in pc_bit_vectors
            value=g.new()
            pc_bit_vectors[base]=value
            for j in range(8):
                address=base+1+8*j
                assert address not in address_lanes
                address_lanes[address]=lane(value,j)
    row_groups=set(cfg["grand_row_groups"])
    row_pointers={}
    if row_groups:
        assert cfg["compact_heap"] and cfg["store_children"] and cfg["prefetch3"]
        assert cfg["heap_io_backup"], "Row buffers share the otherwise unused index area"
        assert row_groups <= set(cfg["prefetch5_groups"])
        assert 1 <= cfg["grand_row_buffers"] <= 3
        for buffer in range(cfg["grand_row_buffers"]):
            pointers=[]
            for shift in (0,-4):
                base=2062+72*buffer+shift
                value=g.new()
                pointers.append(value)
                for j in range(8):
                    address=base+8*j
                    assert address not in address_lanes
                    address_lanes[address]=lane(value,j)
            row_pointers[buffer]=pointers
    scalar_tick = 0
    leaf_tick = 0
    constant_zero = None
    header_root = None
    ahead_bits = set(tuple(x) for x in cfg["ahead_bits"])
    fold_path3 = set(tuple(x) for x in cfg['fold_path3_groups'])
    prefetch_madd_groups = set(tuple(x) for x in cfg["prefetch_madd_groups"])
    if cfg["compact_heap"]:
        assert cfg["prexor"]
        assert not cfg["ahead_bits"]
    if cfg["share_shallow_bias"]:
        assert cfg["share_shallow_loads"]
    if cfg["share_shallow_loads"]:
        assert cfg["small_bias_alu"] or cfg["share_shallow_bias"]

    def prefetch_madd(r, k):
        return cfg["prefetch_madd"] or (r, k) in prefetch_madd_groups

    def fold_path4(k):
        return cfg["fold_path4"] or k in cfg["fold_path4_groups"]

    def scalar(value, force_load=False):
        nonlocal constant_zero
        value &= MASK
        force_load = force_load or value in cfg['force_load_scalars']
        if value not in sc:
            v = address_lanes[value] if value in address_lanes else g.new(1)
            if cfg["flow_constants"] == "all" or value in cfg["flow_constants"]:
                if constant_zero is None:
                    constant_zero = g.new(1)
                    g.initial_zero.append(constant_zero)
                g.emit(f"constant.{value}", "flow", ("add_imm", v, constant_zero, value), [(constant_zero, 1)], [(v, 1)])
                sc[value] = v
                return v
            expression = None
            if cfg["synth_scalars"] and not force_load and value not in (2310, 7, 8, 32):
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
        if cfg["alias_constants"] == "all" or value in cfg["alias_constants"]:
            sc[value] = vc[value]
        return vc[value]

    def binary(op, a, b, name, flexible=True):
        nonlocal scalar_tick
        dst = g.new()
        scalar_tick += bool(flexible)
        fraction = cfg["scalar"]
        if isinstance(fraction, list):
            fraction = fraction[min(len(fraction)-1, max(0, g.tag[0]))]
        offload = flexible and int(scalar_tick * fraction) != int((scalar_tick - 1) * fraction)
        if flexible and cfg["scalar_labels"] is not None and name.startswith("r"):
            label = ".".join(name.split(".")[2:])
            offload = label in cfg["scalar_labels"]
            index = g.tag[0] * 32 + g.tag[1]
            if label == cfg["scalar_extra"] and label not in cfg["scalar_labels"]:
                f = cfg["scalar_extra_fraction"]
                offload = int((index+1)*f) != int(index*f)
        if flexible and cfg["scalar_overrides"] and name in cfg["scalar_overrides"]:
            offload = cfg["scalar_overrides"][name]
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

    def load_inputs():
        old_tag=g.tag
        values,addresses,loads=[],[],[]
        for k in range(32):
            g.tag=(-1,k)
            ptr=scalar(2310+8*k)
            addresses.append(ptr)
            values.append(vload(ptr,f"g{k}.input"))
            loads.append(len(g.ops)-1)
        g.tag=old_tag
        return values,addresses,loads

    if cfg["initial_ones"] or cfg["header_constants"] or cfg["initial_zero_vector"]:
        zero = g.new(8 if cfg["initial_ones"] or cfg["initial_zero_vector"] else 1)
        g.initial_zero.append(zero)
        if cfg["initial_ones"]:
            one = g.new()
            g.emit("constant.ones", "valu", ("==", one, zero, zero), [(zero, 8)], [(one, 8)])
            vc[1] = one
            sc[1] = one
        if cfg["initial_zero_vector"]:
            vc[0] = sc[0] = zero
        if cfg["header_constants"]:
            header = vload(zero,"header.constants")
            header_root = lane(header,7)
            for j,value in enumerate((16,2047,256,10,7,2054,2310)):
                sc[value] = lane(header,j)

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
    for k in cfg["early_gather_groups"]:
        assert 0 <= k < 32 and k not in cfg["prefetch5_groups"]
        modes[3][k] = modes[4][k] = "gather"
    for r in (2,13):
        for k in cfg[f'dispatch{r}_groups']:
            assert 0 <= k < 32
            assert r==13 or k not in cfg['prefetch5_groups']
            modes[r][k]='jump'
            modes[r+2][k]='gather'
    for r in (2, 3, 4, 13, 14):
        if r==3 and cfg['gather_dispatch4']:
            for k in cfg['dispatch4_groups']:
                assert 0 <= k < 32 and k not in cfg['prefetch5_groups']
                modes[3][k]='gather'
        if r==4:
            for k in cfg['dispatch4_groups']:
                assert 0 <= k < 32 and k not in cfg['prefetch5_groups']
                modes[4][k]='jump'
        if cfg[f"prefetch{r % 11}"]:
            for k in range(32):
                if modes[r][k] == "jump":
                    modes[r+1][k] = "prefetch"
    for k in cfg["prefetch5_groups"]:
        assert modes[3][k] == "jump" and modes[4][k] == "prefetch"
        modes[5][k] = "grand"
    for k in cfg["dispatch5_groups"]:
        assert 0 <= k < 32 and k not in cfg["prefetch5_groups"]
        modes[5][k] = "jump"

    dispatch_groups = {}
    width_overrides = {(r, k): width for r, k, width in cfg["dispatch_widths"]}
    for r in (2, 3, 4, 5, 13, 14, 15):
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
            width = width_overrides.get((r, k), width)
            assert 1 <= width <= 4
            width = min(width, 32-k)
            remaining = next((j-k for j in range(k, 32) if modes[r][j] != "jump"), 32-k)
            if width == 3 and remaining == 4:
                width = 4
            width = min(width, remaining)
            assert all(modes[r][j] == "jump" for j in range(k, k+width))
            dispatch_groups[r, k] = width
            k += width

    for k in late_pairs:
        assert dispatch_groups.get((14,k))==2
        assert all(modes[15][j]=='prefetch' and not prefetch_madd(15,j) for j in (k,k+1))

    spans = {(r,k):span for r,k,span in cfg["dispatch_spans"]}
    assert set(spans) <= set(dispatch_groups)
    assert all(1 <= span <= 4 for span in spans.values())
    field_plans = {}
    for (r,k), width in dispatch_groups.items():
        fields = []
        if cfg["store_children"] and r in (2,3,4,13,14) and cfg[f"prefetch{r%11}"]:
            fields = [("child",s,c) for s in range(width)
                      if modes[r+1][k+s]=='prefetch' for c in range(2)]
            fields += [("grand",s,c) for s in range(width) if r==3 and k+s in cfg["prefetch5_groups"] for c in range(4)]
            fields = fields[:2*spans.get((r,k),1)]
            rows=sum(r==3 and k+s in row_groups for s in range(width))
            if rows:
                assert spans.get((r,k),1)==1 and rows <= cfg["grand_row_buffers"]
                fields=fields[:2-rows]
        if r==14 and k in late_pairs:
            fields=[]
        field_plans[r,k] = fields
    regular_buffers = cfg["temp_buffers"] or 2
    extended_words = 8*max([0,*[len(fields) for key,fields in field_plans.items() if spans.get(key,1)>1]])
    reserved_io_words = 16*regular_buffers+extended_words
    assert reserved_io_words <= 256
    temp_rank=None
    if cfg["temp_region_order"]:
        order=[tuple(key) for key in cfg["temp_region_order"] if tuple(key) in dispatch_groups]
        assert len(order)==len(set(order)) and set(order)==set(dispatch_groups)
        temp_rank={key:i for i,key in enumerate(order)}
    pair_order=list(cfg['late_pair_order'])
    if pair_order:
        assert temp_rank is None and len(pair_order)==len(set(pair_order)) and set(pair_order)==late_pairs
    else:
        pair_order=sorted(late_pairs,key=lambda k:temp_rank[14,k] if temp_rank else k)
    pair_rank={k:i for i,k in enumerate(pair_order)}
    values=io=input_loads=None
    io_backup_groups={}
    next_io_backup=reserved_io_words//8
    if cfg["heap_io_backup"]:
        assert cfg["compact_heap"] and cfg["heap_backup"]
        backup_words=sum(1<<d for d in range(4,8)
                         if d not in cfg["heap_keep_levels"] and d not in cfg["heap_reuse_levels"])
        assert reserved_io_words+backup_words<=256, "Backups overlap the lookup buffers"
        values,io,input_loads=load_inputs()

    # Prepare runtime tree tables once. Tree/index contents are restored;
    # overwritten input words receive their final output values.
    c = [stage[1] for stage in frozen.HASH_STAGES]
    cache_gather3 = bool(cfg["tail_gathers"] or cfg["early_gather_groups"] or
                         cfg['gather_dispatch4'] and cfg['dispatch4_groups'])
    bases = {d: 7 + (1 << d) - 1 for d in range(11)}
    if cfg["prexor"]:
        bases.update({d: 2054 + (1 << d) - 16 for d in range(4, 8)})
        if cfg["compact_heap"]:
            bases.update({d: 1 << d for d in range(4, 8)})
        if cache_gather3:
            bases[3] = 2294
    syncs = {}
    raw_nodes, adjusted_nodes = {}, {}
    raw_blocks, pending_stores, tree_loads = {}, [], []
    heap_backups = {}
    shallow_raw = shallow_biased = None
    needed = {0}
    for r in range(16):
        if any(mode in ("blend", "jump", "prefetch", "grand") for mode in modes[r]):
            needed.add(r % 11)
    if cfg["prexor"]:
        needed.update(range(4, 8))
        if cache_gather3:
            needed.add(3)
    if cfg["synth_scalars"]:
        for value in (1, 2, 4, 8, 32):
            scalar(value)
    for d in sorted(needed):
        raw_nodes[d], adjusted_nodes[d] = [], []
        raw_blocks[d] = []
        for off in range(0, 1 << d, 8):
            if cfg["share_shallow_loads"] and d <= 2:
                if shallow_raw is None:
                    shallow_raw = vload(scalar(7), "tree.d0.raw0")
                    tree_loads.append((len(g.ops)-1, 7))
                raw = lane(shallow_raw, (1 << d)-1)
            else:
                raw = vload(scalar(7 + (1 << d) - 1 + off), f"tree.d{d}.raw{off}")
                tree_loads.append((len(g.ops)-1, 6+(1 << d)+off))
            raw_blocks[d].append(raw)
            if (cfg["compact_heap"] and cfg["heap_backup"] and 4 <= d <= 7
                    and d not in cfg["heap_keep_levels"] and d not in cfg["heap_reuse_levels"]):
                if cfg["heap_io_backup"]:
                    io_backup_groups[d,off]=next_io_backup
                    backup=io[next_io_backup]
                    next_io_backup+=1
                else:
                    backup = scalar(2054+(1 << d)-16+off)
                op = g.emit(f"tree.d{d}.backup{off}", "store", ("vstore", backup, raw), [(backup, 1), (raw, 8)], [])
                if cfg["heap_io_backup"]:
                    g.control.append((input_loads[io_backup_groups[d,off]],op,0))
                heap_backups[d, off] = backup, op
            mirrored = cfg["compact_heap"] and 4 <= d <= 7
            if cfg["share_shallow_bias"] and d <= 2:
                if shallow_biased is None:
                    shallow_biased = binary("^", shallow_raw, vector(c[5]), "tree.shallow.bias", False)
                bias = lane(shallow_biased, (1 << d)-1)
            elif mirrored:
                bias = g.new(14 if d==4 and late_pairs else 8)
                src = scalar(c[5])
                for j in range(8):
                    dst, source = lane(bias, 7-j), lane(raw, j)
                    op = g.emit(f"tree.d{d}.bias{off}.lane{j}", "alu", ("^", dst, source, src), [(source, 1), (src, 1)], [(dst, 1)])
                    if cfg["heap_backup_before_bias"] and (d, off) in heap_backups:
                        g.control.append((heap_backups[d, off][1], op, 1))
            elif d <= 2 and cfg["small_bias_alu"]:
                bias = g.new()
                src = scalar(c[5])
                for j in range(1 << d):
                    g.emit(f"tree.d{d}.bias{j}", "alu", ("^", lane(bias, j), lane(raw, j), src), [(lane(raw, j), 1), (src, 1)], [(lane(bias, j), 1)])
            else:
                bias = binary("^", raw, vector(c[5]), f"tree.d{d}.bias{off}", False)
            raw_nodes[d].extend(lane(raw, j) for j in range(min(8, (1 << d) - off)))
            adjusted_nodes[d].extend(lane(bias, 7-j if mirrored else j) for j in range(min(8, (1 << d) - off)))
            if cfg["prexor"] and (4 <= d <= 7 or d == 3 and cache_gather3):
                ptr = scalar(bases[d] + ((1 << d)-8-off if mirrored else off))
                pending_stores.append((d, off, ptr, bias))
                if not cfg["compact_heap"]:
                    store = g.emit(f"tree.d{d}.store{off}", "store", ("vstore", ptr, bias), [(ptr, 1), (bias, 8)], [])
                    syncs.setdefault(d, []).append(store)
    if cfg["compact_heap"]:
        # The shifted table overlaps the original tree. Protect exactly the
        # raw blocks each store can overwrite, including shallow cached nodes.
        for d, off, ptr, bias in pending_stores:
            store = g.emit(f"tree.d{d}.store{off}", "store", ("vstore", ptr, bias), [(ptr, 1), (bias, 8)], [])
            address = bases[d]+((1 << d)-8-off if 4 <= d <= 7 else off)
            g.control.extend((before, store, 1) for before, source in tree_loads if source < address+8 and address < source+8)
            syncs.setdefault(d, []).append(store)
    if cfg['header_root']:
        assert header_root is not None
    root0 = bcast(header_root if cfg['header_root'] else raw_nodes[0][0], "root.raw")
    root1 = bcast(adjusted_nodes[0][0], "root.bias")
    child_diffs = {}
    if cfg["prefetch_madd"] or prefetch_madd_groups:
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
                if isinstance(fraction, list):
                    fraction = fraction[max(0, g.tag[0])]
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

    # Each choice occupies a contiguous sequence of one to four bundles.
    region_counts = Counter((r % 11, width, spans.get((r,k),1)) for (r, k), width in dispatch_groups.items())
    table_offsets, total_words = {}, 0
    if cfg["pc_address_pools"]:
        assert not spans, "Address pools currently require single-bundle cases"
        assert 8 <= region_counts[3,1,1] <= 40
    for key in sorted(region_counts):
        d, width, span = key
        table_offsets[key] = []
        for number in range(region_counts[key]):
            if cfg["pc_address_pools"] and key == (3,1,1) and number == region_counts[key]-4:
                assert total_words <= 2318-14
                total_words = 2318-14
            table_offsets[key].append(total_words)
            total_words += 8 * span * (1 << (width*d))
    next_region = Counter()
    prev_offsets = {}
    pc_bit_groups={}
    if pc_bit_vectors:
        numbered=Counter()
        used=set()
        for (r,k),width in dispatch_groups.items():
            key=(r%11,width,spans.get((r,k),1))
            table=table_offsets[key][numbered[key]]
            numbered[key]+=1
            if r==14 and key==(3,1,1) and table+14 in pc_bit_vectors:
                pc_bit_groups[k]=pc_bit_vectors[table+14]
                used.add(table+14)
        assert used==set(pc_bit_vectors), 'Each bit pool must serve a final depth-three singleton'

    prefetched = {}
    grandchildren = {}
    row_columns = {}
    row_records = {}
    pair_nodes={}
    pair_bits={}
    pair_loads=[]
    g.pair_rows=[]
    previous_temp_loads = {}
    temp_reads = []
    temp_output_reads = {}
    if cfg["compact_heap"]:
        possible_child_loads = sum(
            8 * sum(("child",int(cfg["store_children"]),c) not in field_plans[r,group]
                    for c in range(2) if int(cfg["store_children"]) < width)
            for (r, group), width in dispatch_groups.items() if r == 14)
    else:
        possible_child_loads = sum(8*(2 if width >= 3 else 1) for (r, group), width in dispatch_groups.items() if r == 14)
    child_load_counter = 0

    def dispatch(r, group, values, pointers):
        nonlocal child_load_counter
        d = r % 11
        n = 1 << d
        width = dispatch_groups[r, group]
        span = spans.get((r,group),1)
        fields = field_plans[r,group]
        field_index = {field:i for i,field in enumerate(fields)}
        key = (d, width, span)
        cases = n ** width
        prefix = f"r{r}.g{group}.dispatch."
        region_number = next_region[key]
        next_region[key] += 1
        table = table_offsets[key][region_number]
        if cfg["pc_address_pools"] and key == (3,1,1) and table+14 in address_vectors:
            offsets = address_vectors[table+14]
            for j in range(8):
                assert scalar(table+14+8*j) == lane(offsets,j)
        elif region_number == 0:
            offsets = g.new()
            for j in range(8):
                op = g.emit(prefix + f"offset{j}", "load", ("const", lane(offsets, j), table + j*cases*span), [], [(lane(offsets, j), 1)])
                g.pc_constants.append(op)
        else:
            previous, previous_table = prev_offsets[key]
            offsets = binary("+", previous, vector(table-previous_table), prefix + "offsets", cfg["offset_scalar"])
        prev_offsets[key] = offsets, table
        bit_pool = r==14 and group in pc_bit_groups
        fused = r == 14 and (cfg["fuse_tail_pc"] or bit_pool)
        if bit_pool:
            assert width==span==1
            plus_one=pc_bit_groups[group]
            for j in range(8):
                assert scalar(table+15+8*j)==lane(plus_one,j)
            choice=select(bits[group][-1],plus_one,offsets,prefix+'bit_offset')
            targets=madd(pointers[group],vector(2),choice,prefix+'targets')
        elif fused:
            targets = offsets
            for stream in range(width):
                targets = madd(pointers[group+stream], vector(2*span*n**(width-1-stream)), targets, prefix+f"prefix{stream}")
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
            targets = (binary("+", targets, packed_bits, prefix+"targets", False) if span==1 else
                       madd(packed_bits,vector(span),targets,prefix+"targets"))
        else:
            packed = pointers[group]
            for stream in range(1, width):
                packed = madd(packed, vector(n), pointers[group+stream], prefix + f"pack{stream}")
            targets = (binary("+", packed, offsets, prefix + "targets", False) if span==1 else
                       madd(packed,vector(span),offsets,prefix+"targets"))
        mixed = [g.new() for _ in range(width)]
        fetch_streams=tuple(s for s in range(width) if d in (2,3,4) and cfg[f"prefetch{d}"]
                            and modes[r+1][group+s]=='prefetch')
        fetch = bool(fetch_streams)
        pair_row = r==14 and group in late_pairs
        pair_pointers=None
        if fetch:
            children = list(reversed(adjusted_nodes[d+1]))
            store_children = cfg["store_children"] and r in (2, 3, 4, 13, 14) and not pair_row
            if store_children:
                ordinal=temp_rank[r,group] if temp_rank is not None else len(g.regions)
                buffer = ordinal % cfg["temp_buffers"] if cfg["temp_buffers"] else int(r == 14)
                if span > 1:
                    buffer = regular_buffers
                temp_base = 2310 + 16*buffer
                temp_ptrs = [scalar(temp_base+j,force_load=span>1 and cfg["load_temp_addresses"])
                             for j in range(8*len(fields))]
            memory_children = None
            if r == 14 and cfg["load_children"]:
                assert cfg["prexor"] and not any(prefetch_madd(r+1, group+s) for s in range(width))
                indices = range(1 << (d+1))
                if not cfg["compact_heap"]:
                    indices = reversed(indices)
                memory_children = [scalar(bases[d+1]+j) for j in indices]
            for stream in fetch_streams:
                if pair_row:
                    pair_nodes[group+stream]=[g.new(9),g.new(9)]
                else:
                    prefetched[r+1, group+stream] = [g.new(), g.new()]
        if pair_row:
            assert fetch_streams==(0,1) and span==1
            pair_buffer=pair_rank[group]%cfg['late_pair_buffers']
            pair_base=2054+48*pair_buffer
            pair_buffer_key=regular_buffers+pair_buffer
            pair_pointers=[[scalar(pair_base+24*s+2*j) for j in range(8)] for s in range(2)]
            pair_stores=[[],[]]
        grand_streams = [s for s in range(width) if r == 3 and group+s in cfg["prefetch5_groups"]]
        if grand_streams:
            assert width <= 2
            grand_nodes = list(reversed(adjusted_nodes[5]))
            for stream in grand_streams:
                which=group+stream
                if which in row_groups:
                    row_buffer=len(row_records)%cfg["grand_row_buffers"]
                    positive,negative=row_pointers[row_buffer]
                    if row_buffer not in row_columns:
                        for j in range(8):
                            assert scalar(2062+72*row_buffer+8*j)==lane(positive,j)
                            assert scalar(2058+72*row_buffer+8*j)==lane(negative,j)
                        row_columns[row_buffer]=[positive]+[
                            binary("+",positive,vector(c),f"grand_row.buffer{row_buffer}.column{c}",False)
                            for c in range(1,4)]
                    grandchildren[which]=row_columns[row_buffer]
                    row_records[which]=dict(buffer=row_buffer,stores=[],loads=[])
                else:
                    grandchildren[which] = [g.new() for _ in range(4)]
        start_unit = len(g.units)
        start = g.emit(prefix + "jump0", "flow", ("jump_indirect", targets),
                       [(targets, 8), *[(values[group+s], 8) for s in range(width)], *[(x, 1) for x in adjusted_nodes[d]],
                        *([(x, 1) for x in children] if fetch else []),
                        *([(x, 1) for x in child_diffs[d+1]] if fetch and any(prefetch_madd(r+1, group+s) for s in range(width)) else [])], [])
        if cfg["precise_dispatch_inputs"]:
            # Entry reads only PC lane zero. The handlers carry their own
            # data dependencies at the cycle where each word is consumed.
            g.ops[start][2] = [(targets, 1)]
        if fetch and store_children:
            if temp_rank is None:
                g.control.extend((op, start, -1-i//2)
                                 for i,op in enumerate(previous_temp_loads.get(buffer, ())) if i<len(fields))
            for source_group in range((temp_base-2310)//8,(temp_base-2310)//8+len(fields)):
                g.control.append((input_loads[source_group], start, 1))
        if pair_row and temp_rank is None and not cfg['late_pair_order']:
            g.control.extend((op,start,-1-i//2)
                             for i,op in enumerate(previous_temp_loads.get(pair_buffer_key,())))
        parts = []
        temp_stores = [[] for _ in fields]
        relative_times = {start:0}
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
                relative_times[op] = 1+j*span
                if stream in fetch_streams:
                    if pair_row:
                        cache=children[::2]
                        assert all(children[2*q+1]==lane(src,1) and src.off+8<=g.sizes[src.vid]
                                   for q,src in enumerate(cache))
                        dst=pair_pointers[stream][j]
                        op=g.emit(prefix+f'pair_row{j}.{stream}','store',
                                  ('lookup_pair_store',dst,q,*cache),
                                  [(dst,1),(q,1),*[(src,2) for src in cache]],[])
                        if fused:
                            g.lookup_bits[op]=lane(bits[group+stream][-1],j)
                            g.ops[op][2].append((g.lookup_bits[op],1))
                        pair_stores[stream].append(op)
                        xors.append((op,stream))
                        relative_times[op]=1+j
                    for child in (() if pair_row else range(2)):
                        cache = children[child::2] if child == 0 or not prefetch_madd(r+1, group+stream) else child_diffs[d+1]
                        dst = lane(prefetched[r+1, group+stream][child], j)
                        field = ("child",stream,child)
                        store_child = field in field_index
                        eligible_child = (stream == int(store_children) if cfg["compact_heap"] else
                                          (2*stream+child+2*j) % (2*width) < (2 if width >= 3 else 1))
                        load_child = not store_child and memory_children is not None and eligible_child
                        if load_child:
                            fraction = min(1, cfg["load_child_budget"] / possible_child_loads)
                            load_child = int((child_load_counter+1)*fraction) != int(child_load_counter*fraction)
                            child_load_counter += 1
                        engine, code = ("load", "lookup_load") if load_child else ("alu", "lookup_copy")
                        if load_child:
                            cache = memory_children[child::2]
                        if store_child:
                            dst = temp_ptrs[field_index[field]*8+j]
                            engine, code = "store", "lookup_store"
                        op = g.emit(prefix + f"child{j}.{stream}.{child}", engine, (code, dst, q, *cache),
                                    [(q, 1), *[(x, 1) for x in cache], *([(dst, 1)] if store_child else [])], [] if store_child else [(dst, 1)])
                        if store_child:
                            temp_stores[field_index[field]].append(op)
                        if load_child:
                            last_gathers.append(op)
                            gather_levels[op] = d+1
                            g.control.extend((store, start, 1) for store in syncs[d+1])
                        if fused:
                            g.lookup_bits[op] = lane(bits[group+stream][-1], j)
                            g.ops[op][2].append((g.lookup_bits[op], 1))
                        xors.append((op, stream))
                        relative_times[op] = 1+j*span+(field_index[field]//2 if store_child else 0)
                if stream in grand_streams:
                    if group+stream in row_groups:
                        record=row_records[group+stream]
                        positive,negative=row_pointers[record['buffer']]
                        # Both four-node halves share one contiguous eight-word
                        # source. Shift the destination back four words for an
                        # odd choice, keeping the useful quartet at a fixed row.
                        cache=[grand_nodes[(4*q)//8*8] for q in range(n)]
                        assert all(v.off==0 for v in cache)
                        op=g.emit(prefix+f"grand_row{j}.{stream}","store",
                                  ("lookup_vstore",q,lane(positive,j),lane(negative,j),*cache),
                                  [(q,1),(lane(positive,j),1),(lane(negative,j),1),
                                   *[(v,8) for v in cache]],[])
                        record['stores'].append(op)
                        xors.append((op,stream))
                        relative_times[op]=1+j*span
                        continue
                    for child in range(4):
                        dst = lane(grandchildren[group+stream][child], j)
                        cache = grand_nodes[child::4]
                        field = ("grand",stream,child)
                        stored = field in field_index
                        if stored:
                            dst = temp_ptrs[field_index[field]*8+j]
                        op = g.emit(prefix + f"grand{j}.{stream}.{child}", "store" if stored else "alu",
                                    ("lookup_store" if stored else "lookup_copy", dst, q, *cache),
                                    [(q, 1), *[(x, 1) for x in cache], *([(dst,1)] if stored else [])],
                                    [] if stored else [(dst, 1)])
                        if stored:
                            temp_stores[field_index[field]].append(op)
                        xors.append((op, stream))
                        relative_times[op] = 1+j*span+(field_index[field]//2 if stored else 0)
            slot = ("jump_indirect", lane(targets, j+1)) if j < 7 else ("jump", 0)
            jump = g.emit(prefix + f"jump{j+1}", "flow", slot, [(slot[1], 1)] if j < 7 else [], [])
            relative_times[jump] = (j+1)*span
            parts.append((xors, jump))
        g.units[start_unit:] = [[(op,relative_times[op]) for op in relative_times]]
        g.regions.append(dict(start=start, parts=parts, n=n, width=width, span=span, cases=cases, table=table, groups=list(range(group, group+width)), round=r))
        if fetch and store_children and fields:
            loads = []
            for i,(kind,stream,child) in enumerate(fields):
                dst = prefetched[r+1,group+stream][child] if kind=="child" else grandchildren[group+stream][child]
                addr = temp_ptrs[i*8]
                name = f"child_vector{child}" if kind=="child" and stream==0 else f"{kind}_vector{stream}.{child}"
                op = g.emit(prefix+name, "load", ("vload", dst, addr), [(addr, 1)], [(dst, 8)])
                g.control.extend((before, op, 1) for before in temp_stores[i])
                loads.append(op)
            previous_temp_loads[buffer] = loads+previous_temp_loads.get(buffer,[])[len(loads):]
            temp_reads.extend(loads)
            for output_group in range((temp_base-2310)//8,(temp_base-2310)//8+len(fields)):
                temp_output_reads.setdefault(output_group, []).extend(loads)
            g.regions[-1]["temp_loads"] = loads
            g.regions[-1]["temp_buffer"] = buffer
            g.regions[-1]["temp_fields"] = len(fields)
        if pair_row:
            loads=[]
            for block in range(2):
                for stream in range(2):
                    dst=pair_nodes[group+stream][block]
                    address=pair_pointers[stream][4*block]
                    op=g.emit(prefix+f'pair_load{block}.{stream}','load',
                              ('vload',dst,address),[(address,1)],[(dst,8)])
                    g.control.extend((before,op,1) for j,before in enumerate(pair_stores[stream])
                                     if 2*j<8*block+8 and 8*block<2*j+8)
                    loads.append(op)
            previous_temp_loads[pair_buffer_key]=loads
            pair_loads.extend(loads)
            g.regions[-1].update(temp_loads=loads,temp_buffer=pair_buffer_key,temp_fields=4)
            g.pair_rows.append(dict(group=group,base=pair_base,
                                    stores=[[g.names[i] for i in row] for row in pair_stores],
                                    loads=[g.names[i] for i in loads]))
        return mixed

    if values is None:
        values,io,input_loads=load_inputs()
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
                if scalar_root_group(cfg,r,k):
                    scalar_tick+=1
                    root=(header_root if cfg['header_root'] else raw_nodes[0][0]) if r==0 else adjusted_nodes[0][0]
                    value=g.new()
                    for j in range(8):
                        dst,src=lane(value,j),lane(values[k],j)
                        g.emit(prefix+f'mix.lane{j}','alu',('^',dst,src,root),
                               [(src,1),(root,1)],[(dst,1)])
                else:
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
                if r==15 and k in pair_nodes:
                    selected=[]
                    for block in range(2):
                        no=pair_nodes[k][block]
                        yes=lane(no,1)
                        cond=pair_bits[k][block]
                        node=g.new()
                        inputs=[(lane(v,j),1) for v in (cond,yes,no) for j in (0,2,4,6)]
                        g.emit(prefix+f'prefetched_node.half{block}','flow',
                               ('vselect_even',node,cond,yes,no),inputs,[(node,8)])
                        selected.extend(lane(node,2*j) for j in range(4))
                    scalar_tick+=1
                    value=g.new()
                    for j in range(8):
                        dst,src,node=lane(value,j),lane(values[k],j),selected[j]
                        g.emit(prefix+f'mix.lane{j}','alu',('^',dst,src,node),
                               [(src,1),(node,1)],[(dst,1)])
                else:
                    no, yes = prefetched[r, k]
                    if prefetch_madd(r, k):
                        node = madd(bits[k][-1], yes, no, prefix + "prefetched_node")
                    else:
                        node = select(bits[k][-1], yes, no, prefix + "prefetched_node")
                    value = binary("^", values[k], node, prefix + "mix")
            elif mode == "grand":
                nodes = grandchildren[k]
                left = select(bits[k][-2], nodes[2], nodes[0], prefix + "grand_left")
                right = select(bits[k][-2], nodes[3], nodes[1], prefix + "grand_right")
                node = select(bits[k][-1], right, left, prefix + "grand_node")
                if k in row_groups:
                    address=node
                    node=g.new()
                    for j in range(8):
                        op=g.emit(prefix+f"grand_load{j}","load",
                                  ("load",lane(node,j),lane(address,j)),
                                  [(lane(address,j),1)],[(lane(node,j),1)])
                        g.control.append((row_records[k]['stores'][j],op,1))
                        row_records[k]['loads'].append(op)
                value = binary("^", values[k], node, prefix + "mix")
            else:
                address = ptrs[k]
                assert state[k] == "a"
                overfetch=overfetch_member(cfg,r,k)
                node = None if overfetch else g.new()
                node_parts=[]
                deps = syncs.get(depth, [])
                if depth <= 3 and not deps:
                    # Rare shallow gathers use raw nodes, so absorb the
                    # previous round's deferred XOR after the load.
                    bias_gather = True
                else:
                    bias_gather = False
                for j in range(8):
                    if overfetch:
                        # The original node is at 6+q. VLOAD's implicit lane
                        # offset supplies +6 while the recurrence keeps q.
                        part=g.new()
                        g.overfetch_values.append(part)
                        node_parts.append(lane(part,6))
                        op=g.emit(prefix+f'load{j}','load',('vload',part,lane(address,j)),
                                  [(lane(address,j),1)],[(part,8)])
                    else:
                        op = g.emit(prefix + f"load{j}", "load", ("load", lane(node, j), lane(address, j)), [(lane(address, j), 1)], [(lane(node, j), 1)])
                    g.control.extend((before, op, 1) for before in deps)
                    if deps:
                        last_gathers.append(op)
                        gather_levels[op] = depth
                if bias_gather:
                    node = binary("^", node, vector(c[5]), prefix + "load.bias")
                if overfetch:
                    assert not bias_gather
                    value=g.new()
                    scalar_tick+=1
                    for j,source in enumerate(node_parts):
                        dst,old=lane(value,j),lane(values[k],j)
                        g.emit(prefix+f'mix.lane{j}','alu',('^',dst,old,source),
                               [(old,1),(source,1)],[(dst,1)])
                else:
                    value = binary("^", values[k], node, prefix + "mix")
            defer = r != 15 and ((r+1) % 11 <= 3 or modes[r+1][k] in ("blend", "jump", "prefetch", "grand") or cfg["prexor"] and 4 <= (r+1)%11 <= 7)

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
            before_h5 = value
            value = madd(value, vector(9), vector(c[4]), prefix + "h5")
            checkpoint(value, 4)
            shifted = binary(">>", value, vector(16), prefix + "h6.b")
            a = value if defer else binary("^", value, vector(c[5]), prefix + "h6.a")
            value = binary("^", a, shifted, prefix + "h6")
            checkpoint(value, 5, defer)
            values[k] = value
            if r == 15 or depth == 10:
                continue
            if r==14 and k in pair_nodes:
                # This parity is only consumed by the final node selection.
                # Its former coordinate update has no live consumer.
                scalar_tick+=1
                pair_bits[k]=[g.new(),g.new()]
                one=scalar(1)
                for j in range(8):
                    dst=lane(pair_bits[k][j//4],2*(j%4))
                    src=lane(value,j)
                    g.emit(prefix+f'bit.lane{j}','alu',('&',dst,src,one),
                           [(src,1),(one,1)],[(dst,1)])
                continue
            next_state = "a" if modes[r+1][k] == "gather" else "q"
            if (r, k) in ahead_bits:
                assert next_state == "a" and 4 <= depth <= 9
                # bit16(x * 65537) = bit16(x) XOR bit0(x). Fold the
                # multiplication into h5 and use the unshifted bit as a
                # vselect condition, two cycles before the normal parity.
                bias = (c[5] & 1) * 65536 if not defer else 0
                ahead = madd(before_h5, vector(9*65537), vector(c[4]*65537+bias), prefix + "bit.lookahead")
                bit = binary("&", ahead, vector(65536), prefix + "bit")
            else:
                bit = binary("&", value, vector(1), prefix + "bit")
            bits[k].append(bit)
            if depth == 0:
                assert defer and next_state == "q"
                ptrs[k], state[k] = bit, "q"
                continue
            if (r,k) in fold_path3:
                assert depth==2 and modes[r+1][k]=='prefetch'
                pass  # The next round forms the address from q2, b2, and b3.
            elif (r-1,k) in fold_path3:
                assert depth==3 and mode=='prefetch' and next_state=='a' and defer
                base=bases[4] if cfg['compact_heap'] else bases[4]+15
                step=1 if cfg['compact_heap'] else -1
                hi=select(bit,vector(base+3*step),vector(base+2*step),prefix+'address.high')
                lo=select(bit,vector(base+step),vector(base),prefix+'address.low')
                aux=select(bits[k][-2],hi,lo,prefix+'address.aux')
                ptrs[k]=madd(ptrs[k],vector(4*step),aux,prefix+'address')
            elif r == 13 and modes[14][k] == "jump" and (cfg["fuse_tail_pc"] or k in pc_bit_groups):
                pass  # q2 and b13 are consumed separately by the PC builder.
            elif depth == 3 and modes[r+1][k] == "prefetch" and fold_path4(k):
                pass  # q3, b3 and b4 will directly form the depth-5 address.
            elif depth == 4 and modes[r][k] == "prefetch" and fold_path4(k):
                assert defer
                if next_state == "q":
                    hi = select(bit, vector(3), vector(2), prefix+"address.high")
                    aux = select(bits[k][-2], hi, bit, prefix+"address.aux")
                    ptrs[k] = madd(ptrs[k], vector(4), aux, prefix+"address")
                else:
                    base = bases[5] if cfg["compact_heap"] else bases[5] + 31
                    step = 1 if cfg["compact_heap"] else -1
                    hi = select(bit, vector(base+3*step), vector(base+2*step), prefix+"address.high")
                    lo = select(bit, vector(base+step), vector(base), prefix+"address.low")
                    aux = select(bits[k][-2], hi, lo, prefix+"address.aux")
                    ptrs[k] = madd(ptrs[k], vector(4*step), aux, prefix+"address")
            elif depth == 1 and cfg["path2_flow"] and (r, k) not in set(tuple(x) for x in cfg["path2_valu_groups"]) and state[k] == next_state == "q" and defer:
                hi = select(bit, vector(3), vector(2), prefix+"path.high")
                ptrs[k] = select(ptrs[k], hi, bit, prefix+"path")
            elif state[k] == "q" and next_state == "q" and defer:
                ptrs[k] = madd(ptrs[k], vector(2), bit, prefix + "path")
            else:
                # Convert the current coordinate to the next level directly.
                # All bias and base constants become the two select choices.
                if cfg["compact_heap"]:
                    def coordinate(kind, d):
                        if kind == "q":
                            return -1, (1 << d)-1
                        if 4 <= d <= 7:
                            return -1, bases[d]+(1 << d)-1
                        return 1, bases[d]-(6 if overfetch_member(cfg,d,k) else 0)
                    sign, offset = coordinate(state[k], depth)
                    next_sign, next_offset = coordinate(next_state, depth+1)
                    scale = 2*sign*next_sign
                    const = next_offset-scale*offset
                    zero, one = ((const+next_sign, const) if defer else (const, const+next_sign))
                elif next_state == "q":
                    scale = 2 if state[k] == "q" else -2
                    const = 0 if state[k] == "q" else 2*(bases[depth] + (1 << depth)-1)
                    zero, one = (const, const+1) if defer else (const+1, const)
                else:
                    scale = -2 if state[k] == "q" else 2
                    const = bases[depth+1] + 2*((1 << depth)-1) if state[k] == "q" else bases[depth+1]-2*bases[depth]
                    zero, one = (const+1, const) if defer else (const, const+1)
                aux = bit if (zero, one) == (0, 1) else select(bit, vector(one), vector(zero), prefix + "address.aux")
                ptrs[k] = madd(ptrs[k], vector(scale), aux, prefix + "address")
            state[k] = next_state
    g.tag = (16, 0)
    output_io = io.copy()
    if cfg["output_address_block"]:
        writers = {int(base)+j: i for i, op in enumerate(g.ops) for base, size in op[3] for j in range(size)}
        stride = scalar(8)
        for k in range(32):
            if k % cfg["output_address_block"] == 0:
                continue
            ptr = g.new(1)
            previous = output_io[k-1]
            op = g.emit(f"g{k}.output_ptr", "alu", ("+", ptr, previous, stride), [(previous, 1), (stride, 1)], [(ptr, 1)])
            frontier = {writers[int(values[k])+j] for j in range(8)}
            g.control.extend((before, op, -cfg["output_address_window"]) for before in frontier)
            output_io[k] = ptr
    def emit_outputs():
        for k in range(32):
            op = g.emit(f"g{k}.output", "store", ("vstore", output_io[k], values[k]), [(output_io[k], 1), (values[k], 8)], [])
            g.control.extend((before, op, 0) for before in temp_output_reads.get(k, ()))
    if not cfg["heap_io_backup"]:
        emit_outputs()
    # Restore original tree blocks and temporary index words.
    if cfg["compact_heap"]:
        later_reads = []
        for d in range(7, 3, -1):
            restored, reads = [], []
            for block, off in enumerate(range(0, 1 << d, 8)):
                if d in cfg["heap_keep_levels"]:
                    raw = raw_blocks[d][block]
                elif (d, off) in heap_backups:
                    backup, writer = heap_backups[d, off]
                    raw = vload(backup, f"restore.d{d}.read{off}")
                    read = len(g.ops)-1
                    g.control.append((writer, read, 1))
                    if cfg["heap_restore_window"]:
                        g.control.extend((before, read, -cfg["heap_restore_window"]) for before in last_gathers if gather_levels.get(before) in (d, d+1))
                    if cfg["heap_io_backup"]:
                        temp_output_reads.setdefault(io_backup_groups[d,off],[]).append(read)
                    else:
                        zero = vector(0)
                        clear = g.emit(f"restore.d{d}.clear{off}", "store", ("vstore", backup, zero), [(backup, 1), (zero, 8)], [])
                        g.control.append((read, clear, 0))
                else:
                    if d in cfg["heap_reuse_levels"]:
                        sources = adjusted_nodes[d][off:off+8]
                    else:
                        biased = vload(scalar(bases[d]+(1 << d)-8-off), f"restore.d{d}.read{off}")
                        read = len(g.ops)-1
                        reads.append(read)
                        g.control.extend((before, read, 1) for before in syncs[d])
                        sources = [lane(biased, 7-j) for j in range(8)]
                    raw = g.new()
                    bias = scalar(c[5])
                    for j in range(8):
                        dst, src = lane(raw, j), sources[j]
                        g.emit(f"restore.d{d}.unbias{off}.{j}", "alu", ("^", dst, src, bias), [(src, 1), (bias, 1)], [(dst, 1)])
                restored.append((scalar(6+(1 << d)+off), raw, off))
            later_reads.extend(reads)
            for addr, raw, off in restored:
                op = g.emit(f"restore.d{d}.write{off}", "store", ("vstore", addr, raw), [(addr, 1), (raw, 8)], [])
                g.control.extend((before, op, 1) for before in later_reads)
                levels=(d,d+1)
                if cfg['precise_restore']:
                    begin=6+(1<<d)+off
                    levels=tuple(level for level in levels
                                 if begin<bases[level]+(1<<level) and bases[level]<begin+8)
                g.control.extend((before, op, 1) for level in levels for before in syncs.get(level, ()))
                g.control.extend((before, op, 1) for before in last_gathers if gather_levels.get(before) in levels)
        # The six shifted words also overlap the end of depth 3.
        addr, raw = scalar(14), raw_blocks[3][0]
        op = g.emit("restore.d3.write", "store", ("vstore", addr, raw), [(addr, 1), (raw, 8)], [])
        g.control.extend((before, op, 1) for before in later_reads)
        g.control.extend((before, op, 1) for before in syncs[4])
        g.control.extend((before, op, 1) for before in last_gathers if gather_levels.get(before) == 4)
    if cfg["prexor"]:
        for off in range(240 if cfg["compact_heap"] else 0, 248 if cache_gather3 else 240, 8):
            addr = scalar(2054 + off)
            op = g.emit(f"restore.{off}", "store", ("vstore", addr, vector(0)), [(addr, 1), (vector(0), 8)], [])
            depth = 3 if off >= 240 else (off+16).bit_length()-1
            g.control.extend((before, op, 1) for before in syncs.get(depth, ()))
            g.control.extend((before, op, 1) for before in last_gathers if gather_levels.get(before, 4) == depth)
    if cfg["heap_io_backup"]:
        emit_outputs()
    if row_groups:
        previous={}
        for record in row_records.values():
            buffer=record['buffer']
            assert len(record['stores'])==len(record['loads'])==8
            if buffer in previous:
                g.control.extend((before,after,0) for before,after in
                                 zip(previous[buffer]['loads'],record['stores']))
            previous[buffer]=record
        reads=[op for record in row_records.values() for op in record['loads']]
        for offset in range(0,72*cfg["grand_row_buffers"],8):
            address=scalar(2054+offset)
            zero=vector(0)
            op=g.emit(f"grand_row.clear{offset}","store",("vstore",address,zero),
                      [(address,1),(zero,8)],[])
            g.control.extend((before,op,0) for before in reads)
    if late_pairs:
        for offset in range(0,48*cfg['late_pair_buffers'],8):
            address=scalar(2054+offset)
            zero=vector(0)
            op=g.emit(f'late_pair.clear{offset}','store',('vstore',address,zero),
                      [(address,1),(zero,8)],[])
            g.control.extend((before,op,0) for before in pair_loads)
        if cfg['late_pair_order']:
            regions={r['groups'][0]:r for r in g.regions if r['round']==14 and r['groups'][0] in late_pairs}
            previous={}
            for k in pair_order:
                region=regions[k]
                buffer=region['temp_buffer']
                g.control.extend((op,region['start'],-1-i//2)
                                 for i,op in enumerate(previous.get(buffer,())))
                previous[buffer]=region['temp_loads']
    if temp_rank is not None:
        previous={}
        for region in sorted(g.regions,key=lambda r:temp_rank[r['round'],r['groups'][0]]):
            if not region.get('temp_loads'):
                continue
            buffer=region['temp_buffer']
            g.control.extend((op,region['start'],-1-i//2)
                             for i,op in enumerate(previous.get(buffer,())) if i<region['temp_fields'])
            loads=region['temp_loads']
            previous[buffer]=loads+previous.get(buffer,[])[len(loads):]
    if cfg['memory_vectors']:
        # Replicate a scalar through otherwise unused index memory. Each read
        # follows all eight stores; the next fill may share its read cycle.
        assert cfg['compact_heap'] and cfg['heap_io_backup'] and cfg['initial_zero_vector']
        assert not row_groups and not late_pairs
        assert 1 <= cfg['memory_vector_buffers'] <= 8
        eligible={name:i for i,(name,op) in enumerate(zip(g.names,g.ops))
                  if op[1][0]=='vbroadcast' or name.startswith('derive.')}
        selected=(list(eligible) if cfg['memory_vectors']=='all' else list(cfg['memory_vectors']))
        assert len(selected)==len(set(selected)) and set(selected)<=set(eligible)
        order=cfg['memory_vector_order'] or [name for name in eligible if name in selected]
        assert len(order)==len(set(order)) and set(order)==set(selected)
        previous={}
        addresses={}
        g.memory_vectors=[]
        old_tag=g.tag
        g.tag=(-1,0)
        for ordinal,name in enumerate(order):
            i=eligible[name]
            op=g.ops[i]
            assert op[0]=='valu' and len(op[3])==1 and op[3][0][1]==8
            src=op[1][2] if op[1][0]=='vbroadcast' else scalar(int(name.split('.')[1]))
            dst=op[3][0][0]
            buffer=ordinal%cfg['memory_vector_buffers']
            if buffer not in addresses:
                addresses[buffer]=[scalar(2054+8*buffer+j) for j in range(8)]
            stores=[]
            for j,ptr in enumerate(addresses[buffer]):
                store=g.emit(f'memory_vector.{name}.store{j}','store',('store',ptr,src),
                             [(ptr,1),(src,1)],[])
                g.control.append((store,i,1))
                if buffer in previous:g.control.append((previous[buffer],store,0))
                stores.append(g.names[store])
            ptr=addresses[buffer][0]
            g.ops[i]=['load',('vload',dst,ptr),[(ptr,1)],[(dst,8)],op[4]]
            previous[buffer]=i
            g.memory_vectors.append(dict(name=name,buffer=buffer,address=2054+8*buffer,stores=stores))
        for buffer,read in previous.items():
            ptr=addresses[buffer][0]
            zero=vector(0)
            clear=g.emit(f'memory_vector.clear{buffer}','store',('vstore',ptr,zero),
                         [(ptr,1),(zero,8)],[])
            g.control.append((read,clear,0))
        g.tag=old_tag
    g.config = cfg
    g.total_table_words = total_words
    if cfg["dce"]:
        eliminate_dead(g)
    if cfg["merge_chains"]:
        merge_dispatch_regions(g, max(len(chain) for chain in cfg["merge_chains"]))
    elif cfg["merge_regions"] > 1:
        merge_dispatch_regions(g, cfg["merge_regions"])
    audit_pair_padding(g)
    audit_overfetch_padding(g)
    return g


def audit_pair_padding(g):
    """Prove unused vector selection outputs have no logical consumer."""
    observed={int(v)+j for op in g.ops for v,n in op[2] for j in range(n)}
    observed.update(int(v)+j for v,*_ in g.checks for j in range(8))
    for engine,slot,inputs,outputs,*_ in g.ops:
        if slot[0]=='vselect_even':
            _,dest,cond,yes,no=slot
            assert engine=='flow' and outputs==[(dest,8)]
            assert inputs==[(lane(v,j),1) for v in (cond,yes,no) for j in (0,2,4,6)]
            assert all(v.off+8<=g.sizes[v.vid] for v in (dest,cond,yes,no))
            assert not {int(dest)+j for j in (1,3,5,7)}.intersection(observed)
        elif slot[0]=='lookup_pair_store':
            assert engine=='store' and not outputs
            assert all(v.off+8<=g.sizes[v.vid] for v in slot[3:])


def audit_overfetch_padding(g):
    observed={int(v)+j for op in g.ops for v,n in op[2] for j in range(n)}
    observed.update(int(v)+j for v,*_ in g.checks for j in range(8))
    for value in getattr(g,'overfetch_values',()):
        assert g.sizes[value.vid]==8
        assert not {int(value)+j for j in range(8) if j!=6}.intersection(observed)


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
    if g.config["merge_chains"]:
        region_of = {(r["round"], r["groups"][0]): r for r in g.regions}
        seen = set()
        batches = []
        for chain in g.config["merge_chains"]:
            keys = [tuple(key) for key in chain]
            assert 2 <= len(keys) <= 4 and not seen.intersection(keys)
            assert len(set(keys)) == len(keys)
            seen.update(keys)
            batches.append([region_of[key] for key in keys])
    else:
        batches = []
        for rnd in sorted({r["round"] for r in g.regions}):
            regions = [r for r in g.regions if r["round"] == rnd]
            batches.extend(regions[start:start+number] for start in range(0, len(regions), number))
    for batch in batches:
        if len(batch) == 1:
            continue
        assert sum(r.get("span",1) for r in batch) <= 4, "Merged dispatch exceeds the 33-cycle unit limit"
        combined = []
        first_unit = unit_of[batch[0]["start"]]
        relative_start = 0
        for j, region in enumerate(batch):
            unit = unit_of[region["start"]]
            removed_units.add(unit)
            for i, off in g.units[unit]:
                if j and i == region["start"]:
                    continue
                combined.append((i, off+relative_start))
            if j+1 < len(batch):
                region["chain_exit"] = True
                next_start = batch[j+1]["start"]
                previous_exit = region["parts"][-1][1]
                aliases[next_start] = previous_exit
                g.ops[previous_exit] = [*g.ops[next_start][:4], g.ops[previous_exit][4]]
                for load_index,i in enumerate(region.get("temp_loads", ())):
                    removed_units.add(unit_of[i])
                    combined.append((i, relative_start+8*region.get("span",1)+1+load_index//2))
            relative_start += 8*region.get("span",1)
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
    case_bundles = sum(8*r["cases"]*r.get("span",1) for r in g.regions)
    padding = g.total_table_words-case_bundles+(13 if g.config.get("pc_address_pools") else 0)
    return dict(engines=dict(c), weighted_alu_valu=c["alu"]+8*c["valu"],
                bound=max((c[e]+int(cap)-1)//int(cap) for e, cap in zip(ENGINES, CAPACITY)),
                arithmetic_bound=(c["alu"]+8*c["valu"]+59)//60,
                table_bundles=g.total_table_words, table_case_bundles=case_bundles,
                static_padding_bundles=padding, regions=len(g.regions), values=len(g.sizes))


class Scheduler:
    def __init__(self, g):
        import ctypes
        import os
        import subprocess
        import tempfile
        lib_path = ROOT / ".schedule.so"
        if not lib_path.exists() or lib_path.stat().st_mtime < (ROOT / "schedule.cpp").stat().st_mtime:
            fd, temporary = tempfile.mkstemp(prefix=".schedule.", suffix=".so", dir=ROOT)
            os.close(fd)
            try:
                subprocess.run(["g++", "-O3", "-std=c++17", "-shared", "-fPIC", str(ROOT/"schedule.cpp"), "-o", temporary], check=True)
                os.replace(temporary, lib_path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        self.lib = ctypes.CDLL(str(lib_path))
        ptr = np.ctypeslib.ndpointer(dtype=np.int64, flags="C_CONTIGUOUS")
        fptr = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
        self.lib.schedule_search.argtypes = [ctypes.c_int, ctypes.c_int, ptr, ptr, ptr, ptr, ptr, fptr,
                                            ctypes.c_int, ctypes.c_uint64, ctypes.c_double, ptr, ptr]
        self.lib.schedule_search.restype = ctypes.c_int
        self.g = g
        n = len(g.units)
        units = [None] * len(g.ops)
        self.usage = np.zeros((n, 33, 6), dtype=np.int64)
        self.durations = np.zeros(n, dtype=np.int64)
        self.tags = np.zeros((n, 2), dtype=np.int64)
        for u, ops in enumerate(g.units):
            self.tags[u] = g.ops[ops[0][0]][4]
            for i, off in ops:
                assert 0 <= off <= 32, "Compound unit exceeds 33 cycles"
                units[i] = (u, off)
                self.usage[u, off, ENGINES.index(g.ops[i][0])] += 1
                self.durations[u] = max(self.durations[u], off)
        for region in g.regions:
            u,off = units[region["start"]]
            self.usage[u,off:off+8*region.get("span",1)+1,5] = 1
        assert np.all(self.usage <= RESOURCE_CAPACITY)
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
    if g.config.get("lane_allocation") and policy == 0:
        return allocate_lanes(g, times)
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


def allocate_lanes(g, times):
    """Respect read-before-write timing separately for each scratch word."""
    import ctypes
    import os
    import subprocess
    import tempfile
    path = ROOT / ".lane_allocate.so"
    if not path.exists() or path.stat().st_mtime < (ROOT / "lane_allocate.cpp").stat().st_mtime:
        fd, temporary = tempfile.mkstemp(prefix=".lane_allocate.", suffix=".so", dir=ROOT)
        os.close(fd)
        try:
            subprocess.run(["g++", "-O3", "-std=c++17", "-shared", "-fPIC", str(ROOT/"lane_allocate.cpp"), "-o", temporary], check=True)
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)
    lib = ctypes.CDLL(str(path))
    ptr = np.ctypeslib.ndpointer(dtype=np.int64, flags="C_CONTIGUOUS")
    lib.allocate_lanes_native.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ptr, ptr, ptr, ctypes.c_int, ptr]
    lib.allocate_lanes_native.restype = ctypes.c_int
    n, horizon = len(g.sizes), int(max(times))+3
    words=16 if max(g.sizes,default=8)>8 else 8
    first = np.full((n, words), horizon, dtype=np.int64)
    end = np.full((n, words), -1, dtype=np.int64)
    for v in g.initial_zero:
        first[v.vid, :g.sizes[v.vid]] = 0
        end[v.vid, :g.sizes[v.vid]] = 1
    for i, op in enumerate(g.ops):
        t = int(times[i])
        for base, length in op[3]:
            assert 0 <= base.off and base.off+length <= g.sizes[base.vid]
            row = slice(base.off, base.off+length)
            first[base.vid, row] = np.minimum(first[base.vid, row], t+1)
            end[base.vid, row] = np.maximum(end[base.vid, row], t+2)
        for base, length in op[2]:
            assert 0 <= base.off and base.off+length <= g.sizes[base.vid]
            row = slice(base.off, base.off+length)
            end[base.vid, row] = np.maximum(end[base.vid, row], t+1)
    for i, op in enumerate(g.ops):
        for base, length in op[2]:
            assert np.all(first[base.vid, base.off:base.off+length] <= int(times[i])), g.names[i]
    sizes = np.asarray(g.sizes, dtype=np.int64)
    assert np.all((1 <= sizes) & (sizes <= words))
    live = first < end
    events = np.zeros(horizon+1, dtype=np.int64)
    np.add.at(events, first[live], 1)
    np.add.at(events, end[live], -1)
    peak = int(events.cumsum().max())
    if peak > 1536:
        return None, dict(allocation_policy="lanes", lane_live_peak=peak,
                          reason="Per-lane live words exceed scratch capacity")
    output = np.empty(n, dtype=np.int64)
    failures = []
    for policy in range(g.config.get("lane_allocation_trials", 8)):
        failed = lib.allocate_lanes_native(n, horizon, 1536, words, sizes, first, end, policy, output)
        if failed:
            failures.append(int(failed)-1)
            continue
        bases = {v: int(output[v]) for v in range(n) if output[v] >= 0}
        # Independently audit the native bitset result using sorted intervals.
        cells = [[] for _ in range(1536)]
        for v, base in bases.items():
            for j in range(g.sizes[v]):
                if first[v, j] < end[v, j]:
                    cells[base+j].append((int(first[v, j]), int(end[v, j]), v, j))
        for address, intervals in enumerate(cells):
            intervals.sort()
            for a, b in zip(intervals, intervals[1:]):
                assert a[1] <= b[0], (address, a, b)
        return bases, dict(scratch_words=max(bases[v]+g.sizes[v] for v in bases),
                           allocation_policy=f"lanes-{policy}", lane_live_peak=peak)
    return None, dict(allocation_policy="lanes", lane_live_peak=peak, failed_values=failures)


def lower(g, times, bases):
    """Expand dense cases only after all scratch and schedule checks pass."""
    total = int(max(times)) + 1
    tables_first = g.config.get("pc_address_pools", False)

    def word(v):
        return bases[v.vid] + v.off if isinstance(v, pt.V) else v

    slots = [tuple(word(v) for v in op[1]) for op in g.ops]
    for i,slot in enumerate(slots):
        if slot[0]=='vselect_even':
            slots[i]=('vselect',*slot[1:])
    for i in g.pc_constants:
        code, dst, immediate = slots[i]
        slots[i] = (code, dst, immediate+(14 if tables_first else total))
    logical = [{} for _ in range(total)]
    for i, op in enumerate(g.ops):
        logical[int(times[i])].setdefault(op[0], []).append(slots[i])
    for bundle in logical:
        if not bundle:
            # An explicit self-copy preserves a scheduled empty cycle on the
            # frozen machine, whose empty dictionaries do not advance time.
            bundle["alu"] = [("|",0,0,0)]
    # Falling off main code must skip the out-of-line case bodies.
    # A final jump is packed with a last store if its FLOW slot is free.
    if not tables_first:
        assert not logical[-1].get("flow")
        logical[-1]["flow"] = [("jump", total + g.total_table_words)]
    program = list(logical) + [{} for _ in range(g.total_table_words)]
    origins = np.full(len(program), -1, dtype=np.int64)
    origins[:total] = np.arange(total)
    for region in g.regions:
        entry = int(times[region["start"]])
        n, width, cases = region["n"], region["width"], region["cases"]
        span = region.get("span",1)
        for part, (lookups, jump) in enumerate(region["parts"]):
            assert int(times[jump]) == entry+(part+1)*span
            for phase in range(span):
                cycle = entry+part*span+phase+1
                pos = total + region["table"] + part*cases*span + phase
                replacements = {slots[i]: (i, stream) for i, stream in lookups if int(times[i])==cycle}
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
                                elif slot[0] == "lookup_vstore":
                                    slot = ("vstore",slot[2+(q&1)],slot[4+q])
                                elif slot[0] == 'lookup_pair_store':
                                    slot = ('vstore',slot[1],slot[3+q])
                                else:
                                    assert slot[0] == "lookup_copy"
                                    slot = ("|", slot[1], slot[3+q], slot[3+q])
                            elif engine == "flow" and slot == slots[jump] and part == 7 and not region.get("chain_exit"):
                                slot = ("jump", cycle+1)
                            out.append(slot)
                        bundle[engine] = out
                    program[pos+choice*span] = bundle
                    origins[pos+choice*span] = cycle
                program[cycle] = {}
                origins[cycle] = -1
    if tables_first:
        # Cycle zero jumps over the fixed-address tables; the remaining main
        # bundles follow them. No extra dynamic cycle is introduced.
        assert g.config["compact_main"] and origins[0] == 0
        assert not logical[0].get("flow"), "Bootstrap needs the first FLOW slot"
        main = [p for p in range(total) if origins[p] >= 0]
        start = 14+g.total_table_words
        addresses = {old:start+i for i,old in enumerate(main[1:])}
        addresses[0] = 0
        addresses.update((p,14+p-total) for p in range(total,len(program)))
        size = start+len(main)-1
        addresses[len(program)] = size
        order = [0] + [None]*13 + list(range(total,len(program))) + main[1:]
        assert len(order) == size
        placed, mapping = [], []
        for old in order:
            if old is None:
                placed.append({})
                mapping.append(-1)
                continue
            bundle = {engine:[("jump",addresses[slot[1]]) if slot[0]=="jump" else slot
                              for slot in slots] for engine,slots in program[old].items()}
            placed.append(bundle)
            mapping.append(int(origins[old]))
        placed[0]["flow"] = [("jump",addresses[main[1]])]
        program, origins = placed, np.asarray(mapping,dtype=np.int64)
    elif g.config["compact_main"]:
        # Each out-of-line handler replaces a main-program position that is
        # never executed. Remove those holes and relocate absolute addresses.
        keep = np.flatnonzero(origins >= 0)
        addresses = {int(old): new for new, old in enumerate(keep)}
        addresses[len(program)] = len(keep)
        constants = {(int(times[i]), slots[i]) for i in g.pc_constants}
        compact = []
        for old in keep:
            bundle = {}
            for engine, instructions in program[old].items():
                out = []
                for slot in instructions:
                    if slot[0] == "jump":
                        slot = ("jump", addresses[slot[1]])
                    elif engine == "load" and (int(origins[old]), slot) in constants:
                        slot = ("const", slot[1], addresses[slot[2]])
                    out.append(slot)
                bundle[engine] = out
            compact.append(bundle)
        program, origins = compact, origins[keep]
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
    audit_pair_padding(g)
    audit_overfetch_padding(g)
    class Scratch(dict):
        def __init__(self,initial):
            super().__init__(initial)
            self.pending={}
        def __setitem__(self,key,value):
            assert key not in self.pending, (key,'Multiple SSA writes in one cycle')
            self.pending[key]=value
        def commit(self):
            dict.update(self,self.pending)
            self.pending.clear()

    class Memory(list):
        def __init__(self,initial):
            super().__init__(initial)
            self.pending={}
        def __setitem__(self,key,value):
            self.pending[key]=value
        def commit(self):
            for key,value in self.pending.items():
                list.__setitem__(self,key,value)
            self.pending.clear()

    times=np.arange(len(g.ops),dtype=np.int64)
    if (g.config.get('grand_row_groups') or g.config.get('late_pair_groups') or
            g.config.get('memory_vectors')):
        # Row reuse introduces edges from a later round to an earlier-built
        # group. Interpret a dependency-valid logical timeline, with all reads
        # preceding same-cycle writes, instead of construction order.
        scheduler=Scheduler(g)
        starts=np.zeros(len(g.units),dtype=np.int64)
        for unit in scheduler.order:
            for child,lag in scheduler.children[unit]:
                starts[child]=max(starts[child],starts[unit]+lag)
        times=scheduler.op_times(starts)
    order=np.argsort(times,kind='stable')
    for seed in seeds:
        random.seed(seed)
        tree = frozen.Tree.generate(10)
        inp = frozen.Input.generate(tree, 256, 16)
        mem = frozen.build_mem_image(tree, inp)
        initial = mem.copy()
        trace = {}
        for ref in frozen.reference_kernel2(mem.copy(), trace):
            pass
        mem=Memory(mem)
        values = Scratch({int(v)+j:0 for v in g.initial_zero for j in range(g.sizes[v.vid])})

        def binary(op, a, b):
            if op == "+": return (a+b)&MASK
            if op == "-": return (a-b)&MASK
            if op == "*": return a*b&MASK
            if op == "^": return a^b
            if op == "&": return a&b
            if op == ">>": return a>>b
            if op == "==": return int(a == b)
            raise ValueError(op)

        cycle=None
        for op_id in order:
            if int(times[op_id])!=cycle:
                values.commit()
                mem.commit()
                cycle=int(times[op_id])
            engine,slot,*_=g.ops[op_id]
            op, *args = slot
            if op == "lookup_vstore":
                q,positive,negative,*cache=args
                index=values[int(q)]
                address=values[int(negative if index&1 else positive)]
                for j in range(8):
                    mem[address+j]=values[int(cache[index])+j]
            elif op == 'lookup_pair_store':
                dst,q,*cache=args
                index=values[int(q)]
                if op_id in g.lookup_bits:
                    index=2*index+values[int(g.lookup_bits[op_id])]
                address=values[int(dst)]
                for j in range(8):
                    # Only the adjacent child pair is semantically retained.
                    # Poison the other writes; later ordered stores/clears
                    # must eliminate them before any observed memory read.
                    mem[address+j]=(values[int(cache[index])+j] if j<2 else
                                    (0xa5a50000 ^ (index<<8) ^ j))
            elif op == "lookup_store":
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
            elif op == 'vselect_even':
                dst,cond,yes,no=args
                for j in range(8):
                    values[int(dst)+j]=(values[int(yes if values[int(cond)+j] else no)+j]
                                        if j%2==0 else 0x5a5a0000+j)
            elif op == "vselect":
                dst, bit, yes, no = args
                for j in range(8): values[int(dst)+j] = values[int(yes if values[int(bit)+j] else no)+j]
            elif op == "add_imm":
                dst, src, immediate = args
                values[int(dst)] = (values[int(src)] + immediate) & MASK
            elif engine == "store":
                addr, src = args
                assert op in ('store','vstore')
                for j in range(8 if op=='vstore' else 1):
                    mem[values[int(addr)]+j] = values[int(src)+j]
            else:
                assert op in ("jump", "jump_indirect")
        values.commit()
        mem.commit()
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
