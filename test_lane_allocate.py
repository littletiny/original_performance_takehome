"""Machine-level checks for partial-vector scratch reuse."""
import unittest

import numpy as np

from optimize import Graph, allocate, frozen, lane, pt


def graph():
    g = Graph()
    g.config = {"lane_allocation": True}
    return g


def run(g, times, mem):
    bases, report = allocate(g, np.array(times, dtype=np.int64))
    assert bases is not None, report
    program = [{} for _ in range(max(times)+1)]
    for op, cycle in zip(g.ops, times):
        slot = tuple(bases[x.vid]+x.off if isinstance(x, pt.V) else x for x in op[1])
        program[cycle].setdefault(op[0], []).append(slot)
    machine = frozen.Machine(mem, program, frozen.DebugInfo({}))
    machine.run()
    return bases, machine


class LaneAllocationTests(unittest.TestCase):
    def vector_copy(self, late_read):
        g = graph()
        source_address, mask, source, output_address, dest = [g.new(n) for n in (1, 1, 8, 1, 8)]
        g.emit("source_address", "load", ("const", source_address, 0), [], [(source_address, 1)])
        g.emit("mask", "load", ("const", mask, 165), [], [(mask, 1)])
        g.emit("source", "load", ("vload", source, source_address), [(source_address, 1)], [(source, 8)])
        g.emit("output_address", "load", ("const", output_address, 8), [], [(output_address, 1)])
        times = [0, 0, 1, 1]
        for j in range(8):
            src, dst = lane(source, j), lane(dest, j)
            g.emit(f"xor{j}", "alu", ("^", dst, src, mask), [(src, 1), (mask, 1)], [(dst, 1)])
            times.append(j+2)
        g.emit("output", "store", ("vstore", output_address, dest), [(output_address, 1), (dest, 8)], [])
        times.append(10)
        if late_read:
            observed = g.new(1)
            g.emit("late_read", "alu", ("+", observed, source, mask), [(source, 1), (mask, 1)], [(observed, 1)])
            times.append(10)
        bases, machine = run(g, times, list(range(1, 9))+[0]*8)
        self.assertEqual(machine.mem[8:], [x ^ 165 for x in range(1, 9)])
        self.assertEqual(machine.cycle, 11)
        if late_read:
            self.assertNotEqual(bases[source.vid], bases[dest.vid])
            self.assertEqual(machine.cores[0].scratch[bases[observed.vid]], 166)
        else:
            self.assertEqual(bases[source.vid], bases[dest.vid])

    def test_progressive_in_place_vector(self):
        self.vector_copy(False)

    def test_later_read_prevents_overwrite(self):
        self.vector_copy(True)

    def test_unused_simultaneous_writes_stay_distinct(self):
        g = graph()
        a, b = g.new(1), g.new(1)
        g.emit("a", "load", ("const", a, 123), [], [(a, 1)])
        g.emit("b", "load", ("const", b, 456), [], [(b, 1)])
        bases, machine = run(g, [0, 0], [])
        self.assertNotEqual(bases[a.vid], bases[b.vid])
        self.assertEqual(machine.cores[0].scratch[bases[a.vid]], 123)
        self.assertEqual(machine.cores[0].scratch[bases[b.vid]], 456)

    def test_initial_zero_can_be_reused_after_read(self):
        g = graph()
        zero, result = g.new(1), g.new(1)
        g.initial_zero.append(zero)
        g.emit("result", "flow", ("add_imm", result, zero, 42), [(zero, 1)], [(result, 1)])
        bases, machine = run(g, [0], [])
        self.assertEqual(bases[zero.vid], bases[result.vid])
        self.assertEqual(machine.cores[0].scratch[bases[result.vid]], 42)

    def test_same_cycle_raw_dependency_is_rejected(self):
        g = graph()
        source, dest = g.new(1), g.new(1)
        g.emit("source", "load", ("const", source, 123), [], [(source, 1)])
        g.emit("bad_read", "alu", ("+", dest, source, source), [(source, 1)], [(dest, 1)])
        with self.assertRaises(AssertionError):
            allocate(g, np.array([0, 0], dtype=np.int64))


if __name__ == "__main__":
    unittest.main()
