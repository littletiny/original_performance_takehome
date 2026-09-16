"""Regression checks for valid schedule hints with negative compound lags."""
import unittest

import numpy as np

from optimize import RESOURCE_CAPACITY, Graph, Scheduler, allocate, lower, frozen, lane


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

    def test_fixed_table_bootstrap_and_case_returns(self):
        # Exercise both alternatives of all eight handlers. The table's
        # absolute addresses stay fixed while main code moves behind it.
        for parity in (0,1):
            graph=Graph()
            graph.config=dict(lane_allocation=True,compact_main=True,pc_address_pools=True)
            graph.total_table_words=16
            zero,targets,result=graph.new(),graph.new(),graph.new()
            one,two=graph.new(1),graph.new(1)
            graph.initial_zero.append(zero)
            times=[]
            choices=[(j+parity)%2 for j in range(8)]
            for j in range(8):
                op=graph.emit(f'pc{j}','load',('const',lane(targets,j),2*j+choices[j]),
                              [],[(lane(targets,j),1)])
                graph.pc_constants.append(op)
                times.append(j//2)
            for value,immediate in ((one,1),(two,2)):
                graph.emit(f'value{immediate}','load',('const',value,immediate),[],[(value,1)])
                times.append(4)
            start=graph.emit('entry','flow',('jump_indirect',targets),[(targets,8)],[])
            times.append(5)
            parts=[]
            for j in range(8):
                choice=one if choices[j] else zero
                op=graph.emit(f'choose{j}','alu',('lookup_xor',lane(result,j),lane(zero,j),choice,one,two),
                              [(lane(zero,j),1),(choice,1),(one,1),(two,1)],[(lane(result,j),1)])
                times.append(6+j)
                slot=('jump_indirect',lane(targets,j+1)) if j<7 else ('jump',0)
                jump=graph.emit(f'next{j}','flow',slot,[(slot[1],1)] if j<7 else [],[])
                times.append(6+j)
                parts.append(([(op,0)],jump))
            graph.regions=[dict(start=start,parts=parts,n=2,width=1,cases=2,span=1,table=0)]
            graph.emit('output','store',('vstore',zero,result),[(zero,1),(result,8)],[])
            times.append(14)
            bases,audit=allocate(graph,np.asarray(times,dtype=np.int64))
            self.assertIsNotNone(bases,audit)
            program,origins,_=lower(graph,times,bases)
            machine=frozen.Machine([99]*8,program,frozen.DebugInfo({}))
            machine.run()
            self.assertEqual(machine.mem,[1+choice for choice in choices])
            self.assertEqual(machine.cycle,15)
            self.assertEqual(len(program),36)
            self.assertEqual(int(origins[0]),0)


if __name__ == '__main__':
    unittest.main()
