"""Regression checks for valid schedule hints with negative compound lags."""
import unittest

import numpy as np

from optimize import RESOURCE_CAPACITY, Graph, Scheduler, allocate, lower, frozen


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
        use = np.zeros((cycles, len(RESOURCE_CAPACITY)), dtype=np.int64)
        for u, start in enumerate(times):
            duration = int(scheduler.durations[u])+1
            use[int(start):int(start)+duration] += scheduler.usage[u, :duration]
        self.assertTrue(np.all(use <= RESOURCE_CAPACITY))

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

    def test_dispatches_do_not_overlap_through_free_flow_slots(self):
        graph=Graph()
        regions=[]
        for index in range(2):
            first=graph.emit(f'entry{index}','flow',('jump',0),[],[])
            last=graph.emit(f'exit{index}','flow',('jump',0),[],[])
            regions.append([(first,0),(last,16)])
            graph.regions.append(dict(start=first,span=2))
        zero,dest=graph.new(),graph.new()
        graph.initial_zero.append(zero)
        background=graph.emit('background','flow',('vselect',dest,zero,zero,zero),
                              [(zero,8)],[(dest,8)])
        graph.units=[*regions,[(background,0)]]
        scheduler=Scheduler(graph)
        score,times,_=scheduler.search([0,1,2],iterations=1,noise=0)
        self.assertGreaterEqual(abs(int(times[0])-int(times[1])),17)
        self.assertTrue(any(int(times[u])<int(times[2])<int(times[u])+16 for u in (0,1)))
        self.assert_valid(scheduler,times,score)

    def test_empty_scheduled_cycle_has_a_real_noop(self):
        graph=Graph()
        graph.config=dict(lane_allocation=True,compact_main=True)
        graph.total_table_words=0
        address,value=graph.new(1),graph.new(1)
        graph.emit('address','load',('const',address,0),[],[(address,1)])
        graph.emit('value','load',('const',value,456),[],[(value,1)])
        graph.emit('output','store',('store',address,value),[(address,1),(value,1)],[])
        times=np.array([0,0,2],dtype=np.int64)
        bases,audit=allocate(graph,times)
        self.assertIsNotNone(bases,audit)
        program,origins,_=lower(graph,times,bases)
        machine=frozen.Machine([0],program,frozen.DebugInfo({}))
        machine.run()
        self.assertEqual(machine.cycle,3)
        self.assertEqual(machine.mem,[456])
        self.assertEqual(list(origins),[0,1,2])


if __name__ == '__main__':
    unittest.main()
