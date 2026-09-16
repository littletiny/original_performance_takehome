"""Regression checks for valid schedule hints with negative compound lags."""
import unittest

import numpy as np

from optimize import CAPACITY, Graph, Scheduler


def negative_lag_graph():
    g = Graph()
    zero = g.new()
    g.initial_zero.append(zero)
    scalar, vector, result = g.new(1), g.new(), g.new()
    d = g.emit('late_load', 'load', ('const', scalar, 7), [], [(scalar, 1)])
    a = g.emit('producer', 'valu', ('vbroadcast', vector, scalar), [(scalar, 1)], [(vector, 8)])
    groups = []
    for label, cycles in (('consumer', 1), ('independent', 9)):
        rows = []
        for cycle in range(cycles):
            for lane in range(12):
                dst = g.new(1)
                op = g.emit(f'{label}.{cycle}.{lane}', 'alu', ('+', dst, zero, zero),
                            [(zero, 1)], [(dst, 1)])
                rows.append((op, cycle))
        groups.append(rows)
    read = g.emit('late_read', 'valu', ('^', result, vector, vector),
                  [(vector, 8)], [(result, 8)])
    groups[0].append((read, 5))
    g.units = [[(d, 4)], [(a, 0)], *groups]
    return g


class ScheduleTests(unittest.TestCase):
    def assert_valid(self, scheduler, times, cycles):
        self.assertTrue(np.all(times[scheduler.dests] >= times[scheduler.sources]+scheduler.lags))
        use = np.zeros((cycles, 5), dtype=np.int64)
        for u, start in enumerate(times):
            duration = int(scheduler.durations[u])+1
            use[int(start):int(start)+duration] += scheduler.usage[u, :duration]
        self.assertTrue(np.all(use <= CAPACITY))

    def test_valid_hint_is_preserved_if_serial_placement_is_worse(self):
        scheduler = Scheduler(negative_lag_graph())
        score, times, _ = scheduler.search([0, 5, 1, 2], iterations=1, noise=0)
        self.assertLessEqual(score, 11)
        self.assert_valid(scheduler, times, score)

    def test_invalid_hint_is_not_an_incumbent(self):
        scheduler = Scheduler(negative_lag_graph())
        score, times, _ = scheduler.search([0, 0, 0, 0], iterations=2, noise=0)
        self.assertGreater(score, 9)
        self.assert_valid(scheduler, times, score)


if __name__ == '__main__':
    unittest.main()
