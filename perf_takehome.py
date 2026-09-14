"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


CFG = {
    "SCALAR_MOD": 3,   # 1 in N offloadable vector ops -> scalar alu
    "D4_FLOW": 0,      # of n_vec vectors per depth-4 round, this many use flow trees
    "GROUPS": 1,       # split vectors into this many start-staggered groups
    "DELAY": 0,        # cycles between group starts (fake alu chain)
    "EA": 64,          # priority: eround = h*EA + v*EB
    "EB": 0,
    "SCHED": "serial",  # "list" (paced) or "serial" (chain-by-chain SGS)
    "DELTA": 28,        # pacing: cycles between chain start phases
    "GATE": 0,          # how many cycles early an op may run ahead of T_des
    "KA": 2,            # serial SGS Kahn key: round weight
    "KB": 1,            # serial SGS Kahn key: vector weight
    "KSEC": "h",        # serial SGS Kahn key: tie-break on "h" or "v"
    "C6DEF": True,      # defer the stage-6 xor of hash c6 across consecutive rounds
    "HSW": 6,           # rounds >= HSW use SCALAR_MOD2 instead
    "SCALAR_MOD2": 5,
    "OFF_P": 3, "OFF_Q": 10,     # offload fraction P/Q for h < HSW
    "OFF_P2": 2, "OFF_Q2": 9,    # offload fraction for h >= HSW
    "RUSH": 2,          # fast-track this many vectors through all rounds first
    "RB": 40,           # rush priority bonus in the Kahn key
}


def build_dep_lists(ops):
    """Return (preds_strict, preds_nonstrict) predecessor index lists.

    Semantics: within a bundle all reads see pre-cycle state and writes apply
    at end of cycle, so:
      - RAW/WAW: consumer/writer must be in a strictly later cycle.
      - WAR: a write may share the cycle with the reads it follows.
    """
    n = len(ops)
    last_w = {}
    readers = {}
    preds_s = [[] for _ in range(n)]
    preds_n = [[] for _ in range(n)]
    for i, (eng, slot, ins, outs, tag) in enumerate(ops):
        ds, dn = set(), set()
        for base, ln in ins:
            for w in range(base, base + ln):
                if w in last_w:
                    ds.add(last_w[w])
        for base, ln in outs:
            for w in range(base, base + ln):
                if w in last_w:
                    ds.add(last_w[w])
                for r in readers.get(w, ()):
                    dn.add(r)
        ds.discard(i)
        dn.discard(i)
        dn -= ds
        preds_s[i] = sorted(ds)
        preds_n[i] = sorted(dn)
        for base, ln in ins:
            for w in range(base, base + ln):
                readers.setdefault(w, []).append(i)
        for base, ln in outs:
            for w in range(base, base + ln):
                last_w[w] = i
                readers[w] = []
    return preds_s, preds_n


def schedule_ops_serial(ops):
    """
    Serial SGS (Kolisch): place ops one at a time in a topological order that
    prefers low (vector, round), each op at the earliest cycle that respects
    dependencies and has a free engine slot. Unlike a cycle-by-cycle greedy
    list scheduler, this packs later chains into the engine bubbles left by
    earlier ones, so the flow/load/valu engines all saturate and the chains
    self-stagger (breaking the flow-wall -> load-wall -> flow-wall lockstep).
    The topological order is needed because the FIFO temp pool couples chains.
    """
    import heapq

    preds_s, preds_n = build_dep_lists(ops)
    n = len(ops)
    succ = [[] for _ in range(n)]
    indeg = [0] * n
    for i in range(n):
        for p in preds_s[i] + preds_n[i]:
            succ[p].append(i)
        indeg[i] = len(preds_s[i]) + len(preds_n[i])
    # Kahn topological order with key (h*KA + v*KB - RB*[v<RUSH], secondary, idx)
    KA, KB = CFG["KA"], CFG["KB"]
    RUSH, RB = CFG["RUSH"], CFG["RB"]
    if CFG["KSEC"] == "h":
        sec = lambda i: ops[i][4][0]
    else:
        sec = lambda i: ops[i][4][1]

    def pri(i):
        h, v = ops[i][4]
        return (h * KA + v * KB - (RB if v < RUSH else 0), sec(i), i)

    heap = [pri(i) for i in range(n) if indeg[i] == 0]
    heapq.heapify(heap)
    order = []
    while heap:
        *_, i = heapq.heappop(heap)
        order.append(i)
        for j in succ[i]:
            indeg[j] -= 1
            if indeg[j] == 0:
                heapq.heappush(heap, pri(j))
    place = [0] * n
    usage = {e: {} for e in SLOT_LIMITS if e != "debug"}
    for i in order:
        est = 0
        for p in preds_s[i]:
            if place[p] + 1 > est:
                est = place[p] + 1
        for p in preds_n[i]:
            if place[p] > est:
                est = place[p]
        eng = ops[i][0]
        limit = SLOT_LIMITS[eng]
        u = usage[eng]
        c = est
        while u.get(c, 0) >= limit:
            c += 1
        u[c] = u.get(c, 0) + 1
        place[i] = c
    maxc = max(place)
    bundles = [[] for _ in range(maxc + 1)]
    for i, c in enumerate(place):
        bundles[c].append(i)
    out = []
    for c in range(maxc + 1):
        if not bundles[c]:
            continue  # drop empty cycles (safe: only widens gaps)
        bundle = {}
        for i in bundles[c]:
            bundle.setdefault(ops[i][0], []).append(ops[i][1])
        out.append(bundle)
    return out


def schedule_ops(ops, stagger=0):
    """
    Paced list scheduler.

    Each op is [engine, slot, ins, outs, tag] where ins/outs are (base, len)
    scratch ranges and tag is (round, vector). Semantics: within a bundle all
    reads see pre-cycle state and writes apply at end of cycle, so:
      - RAW/WAW: consumer must be in a strictly later cycle than producer.
      - WAR: a write may share the cycle with earlier reads (reads see old
        value).

    Pacing: each op gets a target cycle T_des = t_rel + v*DELTA (t_rel is the
    intra-chain critical-path time), so the 32 independent element chains are
    spread through the pipeline instead of marching in lockstep. This keeps
    the flow engine (tree rounds of some chains) overlapped with the load
    engine (gather rounds of others). An op becomes schedulable GATE cycles
    before its T_des; among schedulable ops, lowest T_des first.
    """
    import heapq

    preds_s, preds_n = build_dep_lists(ops)
    n = len(ops)
    succ_s = [[] for _ in range(n)]
    succ_n = [[] for _ in range(n)]
    indeg_s = [0] * n
    indeg_n = [0] * n
    for i in range(n):
        indeg_s[i] = len(preds_s[i])
        indeg_n[i] = len(preds_n[i])
        for p in preds_s[i]:
            succ_s[p].append(i)
        for p in preds_n[i]:
            succ_n[p].append(i)

    # intra-chain critical-path time (ignore cross-chain pool coupling: it is
    # only a pacing target, not a correctness constraint)
    t_rel = [0] * n
    for i in range(n):
        v = ops[i][4][1]
        t = 0
        for p in preds_s[i]:
            if ops[p][4][1] == v and t_rel[p] + 1 > t:
                t = t_rel[p] + 1
        for p in preds_n[i]:
            if ops[p][4][1] == v and t_rel[p] > t:
                t = t_rel[p]
        t_rel[i] = t

    DELTA = CFG["DELTA"]
    GATE = CFG["GATE"]
    # round 0 is pure valu work: front-load it (no pacing) so the ramp stays
    # full; pace from round 1 (the fetch phases) onward
    T_des = [t_rel[i] + (ops[i][4][1] * DELTA if ops[i][4][0] >= 1 else 0)
             for i in range(n)]

    engines = [e for e in SLOT_LIMITS if e != "debug"]
    waiting = {e: [] for e in engines}   # deps cleared, not yet due
    ready = {e: [] for e in engines}     # due, keyed by T_des
    for i in range(n):
        if indeg_s[i] == 0 and indeg_n[i] == 0:
            heapq.heappush(waiting[ops[i][0]], (T_des[i], i))

    bundles = []
    placed = 0
    cycle = 0
    while placed < n:
        bundle = {}
        placed_now = []
        for eng in engines:
            w = waiting[eng]
            while w and w[0][0] <= cycle + GATE:
                td, i = heapq.heappop(w)
                heapq.heappush(ready[eng], (td, i))
            limit = SLOT_LIMITS[eng]
            heap = ready[eng]
            slots = []
            while heap and len(slots) < limit:
                _, i = heapq.heappop(heap)
                slots.append(ops[i][1])
                placed_now.append(i)
                for j in succ_n[i]:
                    indeg_n[j] -= 1
                    if indeg_n[j] == 0 and indeg_s[j] == 0:
                        heapq.heappush(waiting[ops[j][0]], (T_des[j], j))
            if slots:
                bundle[eng] = slots
        for i in placed_now:
            for j in succ_s[i]:
                indeg_s[j] -= 1
                if indeg_s[j] == 0 and indeg_n[j] == 0:
                    heapq.heappush(waiting[ops[j][0]], (T_des[j], j))
        placed += len(placed_now)
        if not bundle:
            # jump to the next cycle where anything becomes due
            nxt = min((w[0][0] - GATE for w in waiting.values() if w),
                      default=None)
            if nxt is None:
                raise RuntimeError("scheduler deadlock")
            cycle = max(cycle + 1, nxt)
            continue
        bundles.append(bundle)
        cycle += 1
    return bundles


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        # Simple slot packing that just uses one slot per instruction bundle
        instrs = []
        for engine, slot in slots:
            instrs.append({engine: [slot]})
        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))

        return slots

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Optimized kernel: fully unrolled, vectorized, list-scheduled VLIW.

        Cost model per cycle: 6 valu slots (8 lanes each = 48 lane-ops), 12 alu
        slots (scalar), 2 load, 2 store, 1 flow (vselect = 8 lanes).

        Techniques:
        - hash affine stages folded into single multiply_adds; S3+S4 fused into
          two madds + one xor (exact mod 2^32 arithmetic).
        - shallow tree depths resolve node values with vselect trees over the
          level table on the otherwise idle flow engine instead of one scalar
          load per element; deep levels gather via scalar loads.
        - per-vector dedicated temp block (TT): no false WAW/WAR sharing between
          the 256 independent element chains, so the scheduler can keep all
          engines full; gather addresses come from a small FIFO temp pool.
        - index math: position p with p' = 2p + (val&1) via one madd; inside
          the deep-load stretch the address is kept directly:
          addr' = 2*addr + (1-forest_p) + bit. Wrap is unconditional at
          depth == forest_height (p' = 0, nothing emitted).
        - a fraction of xor/shift/& ops are emitted as scalar alu ops to use
          the alu engine alongside valu.
        - vselect conditionals use (p & 2^j) directly (cond is != 0), so bit
          extraction is a single & per level.
        """
        assert batch_size % VLEN == 0
        n_vec = batch_size // VLEN
        forest_p = 7
        inp_indices_p = forest_p + n_nodes
        inp_values_p = inp_indices_p + batch_size
        MOD32 = 2**32

        # knobs
        SCALAR_MOD = CFG["SCALAR_MOD"]
        D4_FLOW = CFG["D4_FLOW"]
        C6DEF = CFG["C6DEF"]
        STAGGER = 0

        ops = []  # [engine, slot, ins[(base,len)], outs[(base,len)], tag(h,v)]

        def emit(engine, slot, ins, outs, tag=(0, 0)):
            ops.append([engine, slot, ins, outs, tag])

        def alloc(n):
            return self.alloc_scratch(None, n)

        vals = alloc(batch_size)  # current values (8 lanes per vector)
        pp = alloc(batch_size)  # positions in tree level
        TT = alloc(batch_size)  # dedicated temp per vector (node / hash temp / bit)

        def sconst(val):
            s = alloc(1)
            emit("load", ("const", s, val), [], [(s, 1)])
            return s

        def vconst(val):
            s = sconst(val)
            v = alloc(VLEN)
            emit("valu", ("vbroadcast", v, s), [(s, 1)], [(v, VLEN)])
            return v

        c1, c2, c3, c4, c5, c6 = (s[1] for s in HASH_STAGES)
        k1v = vconst(4097)      # S1: a*4097 + c1
        c1v = vconst(c1)
        c2v = vconst(c2)
        c34v = vconst((c3 + c4) % MOD32)
        m169v = vconst(33 * 512)
        c35v = vconst((c3 * 512) % MOD32)
        c5v = vconst(c5)
        c6v = vconst(c6)
        onev = vconst(1)
        # derived constants: one valu op each (replacing const load +
        # vbroadcast, so pure load savings at zero extra valu cost)
        def vderive(op, a, b):
            v = alloc(VLEN)
            emit("valu", (op, v, a, b), [(a, VLEN), (b, VLEN)], [(v, VLEN)])
            return v

        def vmderive(a, b, c_):
            v = alloc(VLEN)
            emit("valu", ("multiply_add", v, a, b, c_),
                 [(a, VLEN), (b, VLEN), (c_, VLEN)], [(v, VLEN)])
            return v

        twov = vderive("+", onev, onev)          # 2
        gv2 = vderive("+", twov, onev)           # 3
        gv3 = vmderive(gv2, twov, onev)          # 7
        gv4 = vmderive(gv3, twov, onev)          # 15
        sh16v = vderive("+", gv4, onev)          # 16
        sh19v = vderive("+", sh16v, gv2)         # 19
        m9v = vderive("+", gv3, twov)            # 9:  S5: a*9 + c5
        m33v = vmderive(sh16v, twov, onev)       # 33: S3+S4 fused
        forest_pv = vconst(forest_p)
        negv = vderive("-", onev, forest_pv)  # addr' = 2*addr + (1-forest_p) + bit
        # gv[m] = 2^m - 1; linear-select cond for step k is cond_{k-1} ^ gv[tz(k)+1]
        # because k ^ (k-1) == 2^(tz(k)+1) - 1
        gv = {1: onev, 2: gv2, 3: gv3, 4: gv4}
        basev = {4: vderive("+", forest_pv, gv4),      # forest_p + 15
                 5: vderive("+", forest_pv, vderive("+", gv4, sh16v))}  # +31

        # ---- node tables for shallow depths (contiguous in mem) ----
        # one shared staging buffer for the vloads (reused table by table)
        # When C6DEF: leaves are stored as node^c6 with bit-complemented
        # indexing (leaf j = table[p ^ mask]). During deferred stretches the
        # carried value is sp = val^c6 and the position is pbar = p^mask, so
        # the node xor sp^(node^c6) = val^node comes out right while the c6
        # xor of hash stage 6 is skipped entirely on those rounds.
        stage = alloc(16)
        maxtab = 4 if D4_FLOW > 0 else 3
        tb = {}
        for d in range(0, maxtab + 1):
            ntab = 1 << d
            base = sconst(forest_p + ntab - 1)
            for off in range(0, ntab, VLEN):
                if off == 0:
                    emit("load", ("vload", stage, base), [(base, 1)], [(stage, VLEN)])
                else:
                    b2 = sconst(forest_p + ntab - 1 + off)
                    emit("load", ("vload", stage + off, b2), [(b2, 1)], [(stage + off, VLEN)])
            tb[d] = []
            c6tab = C6DEF and 0 < d <= 3  # deferred entries only reach d<=3
            for j in range(ntab):
                v = alloc(VLEN)
                src = stage + (j ^ (ntab - 1) if c6tab else j)
                emit("valu", ("vbroadcast", v, src), [(src, 1)], [(v, VLEN)])
                if c6tab:
                    emit("valu", ("^", v, v, c6v), [(v, VLEN), (c6v, VLEN)],
                         [(v, VLEN)])
                tb[d].append(v)
        # root ^ c6 for deferred entries into depth-0 rounds (after wrap)
        rootc6v = alloc(VLEN)
        emit("valu", ("^", rootc6v, tb[0][0], c6v),
             [(tb[0][0], VLEN), (c6v, VLEN)], [(rootc6v, VLEN)])

        # ---- per-vector dedicated cond temp (select condition / gather     ----
        # ---- address). Chain-local like TT: no cross-vector sharing, so   ----
        # ---- the 32 element chains stay independent.                      ----
        CT = alloc(batch_size)

        # ---- op emitters ----
        scalar_ctr = [0]
        cur_h = [0]

        def vop(op, d, a, b, allow_scalar=True):
            if allow_scalar and op in ("^", ">>", "&", "+"):
                # Bresenham offload: fraction P/Q of offloadable ops -> alu
                P, Q = (CFG["OFF_P"], CFG["OFF_Q"]) if cur_h[0] < CFG["HSW"] \
                    else (CFG["OFF_P2"], CFG["OFF_Q2"])
                scalar_ctr[0] += 1
                if (scalar_ctr[0] * P) // Q != ((scalar_ctr[0] - 1) * P) // Q:
                    for j in range(VLEN):
                        emit("alu", (op, d + j, a + j, b + j),
                             [(a + j, 1), (b + j, 1)], [(d + j, 1)])
                    return
            emit("valu", (op, d, a, b), [(a, VLEN), (b, VLEN)], [(d, VLEN)])

        def vmadd(d, a, b, c_):
            emit("valu", ("multiply_add", d, a, b, c_),
                 [(a, VLEN), (b, VLEN), (c_, VLEN)], [(d, VLEN)])

        def vsel(d, cond, a, b):
            emit("flow", ("vselect", d, cond, a, b),
                 [(cond, VLEN), (a, VLEN), (b, VLEN)], [(d, VLEN)])

        def emit_hash(a, t, defer=False):
            # a ^= node already done; 6-stage hash in 11 vector ops, one temp.
            # defer=True skips S6's ^c6: the carried value stays val^c6 and
            # the next round's node xor absorbs c6 via the tables (C6DEF).
            vmadd(a, a, k1v, c1v)      # S1
            vop(">>", t, a, sh19v)     # S2
            vop("^", a, a, c2v)
            vop("^", a, a, t)
            vmadd(t, a, m33v, c34v)    # S3+S4
            vmadd(a, a, m169v, c35v)
            vop("^", a, a, t)
            vmadd(a, a, m9v, c5v)      # S5
            vop(">>", t, a, sh16v)     # S6
            if not defer:
                vop("^", a, a, c6v)
            vop("^", a, a, t)

        def emit_tree(d, p_v, out_v, ct_v):
            """out_v[i] = table_d[p_v[i]] via a linear vselect scan on the
            flow engine: acc = (p==k) ? leaf_k : acc for k = 1..2^d-1.
            cond_k = p ^ k is computed as a running xor chain:
            cond_k = cond_{k-1} ^ (k ^ (k-1)), and k^(k-1) == 2^(tz(k)+1)-1,
            so only the four constants gv[1..4] = 1,3,7,15 are needed.
            Uses only per-vector dedicated temps -> chains stay independent."""
            if d == 1:
                vsel(out_v, p_v, tb[1][1], tb[1][0])
                return
            vop("^", ct_v, p_v, onev)  # cond_1 = p ^ 1
            vsel(out_v, ct_v, tb[d][0], tb[d][1])
            for k in range(2, 1 << d):
                m = (k & -k).bit_length()  # tz(k) + 1
                vop("^", ct_v, ct_v, gv[m])  # cond_k = cond_{k-1} ^ (k^(k-1))
                vsel(out_v, ct_v, out_v, tb[d][k])

        # ---- I/O address constants ----
        # 5 const loads + a stride-32 alu chain instead of 32 const loads
        vaddr = [alloc(1) for _ in range(n_vec)]
        c32 = sconst(4 * VLEN)
        for k in range(min(4, n_vec)):
            emit("load", ("const", vaddr[k], inp_values_p + VLEN * k),
                 [], [(vaddr[k], 1)])
        for k in range(4, n_vec):
            emit("alu", ("+", vaddr[k], vaddr[k - 4], c32),
                 [(vaddr[k - 4], 1), (c32, 1)], [(vaddr[k], 1)], (0, 0))

        # Staggering: desynchronize vector groups by DELAY cycles so that at
        # any time different groups sit in different round types (share-round
        # flow work overlaps gather-round load work). A serial dummy alu chain
        # provides the delay; group g's initial vloads read the chain after
        # g*DELAY increments (WAW tracking gives each group its own delay).
        GROUPS = CFG["GROUPS"]
        DELAY = CFG["DELAY"]
        dummy = sconst(0)  # delay chain base (and zero const)
        zero_s = dummy
        for g in range(GROUPS):
            if g > 0:
                for _ in range(DELAY):
                    emit("alu", ("+", dummy, dummy, zero_s),
                         [(dummy, 1), (zero_s, 1)], [(dummy, 1)], (0, 0))
            for k in range(g * n_vec // GROUPS, (g + 1) * n_vec // GROUPS):
                deps = [(vaddr[k], 1)] + ([(dummy, 1)] if g > 0 else [])
                emit("load", ("vload", vals + VLEN * k, vaddr[k]),
                     deps, [(vals + VLEN * k, VLEN)], (0, 0))

        # ---- main loop ----
        # invariant: at share rounds (d <= 4) pp holds the position p; during
        # the deep-load stretch (d >= 5) pp holds the gather address directly
        # (addr' = 2*addr + (1 - forest_p) + bit), except at the d==4 load
        # vectors where a pool temp holds the addr for that round only.
        d = 0
        for h in range(rounds):
            cur_h[0] = h
            last = h == rounds - 1
            nxt_d = 0 if d == forest_height else d + 1
            # defer the S6 ^c6 iff next round's node is a compile-time
            # constant (a depth<=3 tree leaf, or the root after a wrap)
            defer = C6DEF and (not last) and nxt_d <= 3
            for v in range(n_vec):
                a = vals + VLEN * v
                p_v = pp + VLEN * v
                t = TT + VLEN * v
                _emit = emit
                tag = (h, v)
                def emit(engine, slot, ins, outs, _tag=tag, _emit=_emit):
                    _emit(engine, slot, ins, outs, _tag)
                if d == 0:
                    vop("^", a, a, rootc6v if (C6DEF and h) else tb[0][0])
                elif d <= 3 or (d == 4 and v < D4_FLOW):
                    emit_tree(d, p_v, t, CT + VLEN * v)  # node -> t
                    vop("^", a, a, t)
                else:
                    if d == 4:
                        # p_v still needed for the p update; use the dedicated
                        # cond temp as addr scratch for this round only
                        u = CT + VLEN * v
                        vop("+", u, p_v, basev[4], allow_scalar=False)
                    elif d == 5:
                        vop("+", p_v, p_v, basev[5], allow_scalar=False)
                        u = p_v
                    else:
                        u = p_v  # pp already holds the gather address
                    for j in range(VLEN):
                        emit("load", ("load", t + j, u + j),
                             [(u + j, 1)], [(t + j, 1)])
                    vop("^", a, a, t)
                emit_hash(a, t, defer)
                if not last and d < forest_height:
                    if d == 0:
                        vop("&", p_v, a, onev)  # p' = val & 1 (p == 0)
                    elif d <= 4:
                        if C6DEF and not defer and d <= 3:
                            # leaving the deferred stretch: pbar -> true p
                            vop("^", p_v, p_v, gv[min(d, 4)])
                        vop("&", t, a, onev)  # p' = 2p + bit (position)
                        vmadd(p_v, p_v, twov, t)
                        # d==5 entry adds basev[5] to turn position -> address
                    elif d <= forest_height - 1:
                        vop("&", t, a, onev)  # addr' = 2*addr + (1-forest_p) + bit
                        vmadd(p_v, p_v, twov, negv)
                        vop("+", p_v, p_v, t)
                # d == forest_height: wrap to 0, nothing to emit
            d = nxt_d

        # ---- final stores ----
        for k in range(n_vec):
            emit("store", ("vstore", vaddr[k], vals + VLEN * k),
                 [(vaddr[k], 1), (vals + VLEN * k, VLEN)], [])

        if CFG["SCHED"] == "serial":
            bundles = schedule_ops_serial(ops)
        else:
            bundles = schedule_ops(ops, stagger=STAGGER)
        self.instrs = [{"flow": [("pause",)]}] + bundles

    def build_kernel_baseline(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Like reference_kernel2 but building actual instructions.
        Scalar implementation using only scalar ALU and load/store.
        """
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")
        tmp3 = self.alloc_scratch("tmp3")
        # Scratch space addresses
        init_vars = [
            "rounds",
            "n_nodes",
            "batch_size",
            "forest_height",
            "forest_values_p",
            "inp_indices_p",
            "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)

        # Pause instructions are matched up with yield statements in the reference
        # kernel to let you debug at intermediate steps. The testing harness in this
        # file requires these match up to the reference kernel's yields, but the
        # submission harness ignores them.
        self.add("flow", ("pause",))
        # Any debug engine instruction is ignored by the submission simulator
        self.add("debug", ("comment", "Starting loop"))

        body = []  # array of slots

        # Scalar scratch registers
        tmp_idx = self.alloc_scratch("tmp_idx")
        tmp_val = self.alloc_scratch("tmp_val")
        tmp_node_val = self.alloc_scratch("tmp_node_val")
        tmp_addr = self.alloc_scratch("tmp_addr")

        for round in range(rounds):
            for i in range(batch_size):
                i_const = self.scratch_const(i)
                # idx = mem[inp_indices_p + i]
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], i_const)))
                body.append(("load", ("load", tmp_idx, tmp_addr)))
                body.append(("debug", ("compare", tmp_idx, (round, i, "idx"))))
                # val = mem[inp_values_p + i]
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_values_p"], i_const)))
                body.append(("load", ("load", tmp_val, tmp_addr)))
                body.append(("debug", ("compare", tmp_val, (round, i, "val"))))
                # node_val = mem[forest_values_p + idx]
                body.append(("alu", ("+", tmp_addr, self.scratch["forest_values_p"], tmp_idx)))
                body.append(("load", ("load", tmp_node_val, tmp_addr)))
                body.append(("debug", ("compare", tmp_node_val, (round, i, "node_val"))))
                # val = myhash(val ^ node_val)
                body.append(("alu", ("^", tmp_val, tmp_val, tmp_node_val)))
                body.extend(self.build_hash(tmp_val, tmp1, tmp2, round, i))
                body.append(("debug", ("compare", tmp_val, (round, i, "hashed_val"))))
                # idx = 2*idx + (1 if val % 2 == 0 else 2)
                body.append(("alu", ("%", tmp1, tmp_val, two_const)))
                body.append(("alu", ("==", tmp1, tmp1, zero_const)))
                body.append(("flow", ("select", tmp3, tmp1, one_const, two_const)))
                body.append(("alu", ("*", tmp_idx, tmp_idx, two_const)))
                body.append(("alu", ("+", tmp_idx, tmp_idx, tmp3)))
                body.append(("debug", ("compare", tmp_idx, (round, i, "next_idx"))))
                # idx = 0 if idx >= n_nodes else idx
                body.append(("alu", ("<", tmp1, tmp_idx, self.scratch["n_nodes"])))
                body.append(("flow", ("select", tmp_idx, tmp1, tmp_idx, zero_const)))
                body.append(("debug", ("compare", tmp_idx, (round, i, "wrapped_idx"))))
                # mem[inp_indices_p + i] = idx
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], i_const)))
                body.append(("store", ("store", tmp_addr, tmp_idx)))
                # mem[inp_values_p + i] = val
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_values_p"], i_const)))
                body.append(("store", ("store", tmp_addr, tmp_val)))

        body_instrs = self.build(body)
        self.instrs.extend(body_instrs)
        # Required to match with the yield in reference_kernel2
        self.instrs.append({"flow": [("pause",)]})

BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
