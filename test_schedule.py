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
        for span,parity in ((span,parity) for span in (1,2,3) for parity in (0,1)):
            graph=Graph()
            graph.config=dict(lane_allocation=True,compact_main=True,pc_address_pools=True)
            graph.total_table_words=16*span
            zero,targets,result=graph.new(),graph.new(),graph.new()
            one,two=graph.new(1),graph.new(1)
            graph.initial_zero.append(zero)
            times=[]
            choices=[(j+parity)%2 for j in range(8)]
            for j in range(8):
                op=graph.emit(f'pc{j}','load',('const',lane(targets,j),span*(2*j+choices[j])),
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
                times.append(6+j*span)
                slot=('jump_indirect',lane(targets,j+1)) if j<7 else ('jump',0)
                jump=graph.emit(f'next{j}','flow',slot,[(slot[1],1)] if j<7 else [],[])
                times.append(5+(j+1)*span)
                parts.append(([(op,0)],jump))
            graph.regions=[dict(start=start,parts=parts,n=2,width=1,cases=2,span=span,table=0)]
            graph.emit('output','store',('vstore',zero,result),[(zero,1),(result,8)],[])
            times.append(6+8*span)
            bases,audit=allocate(graph,np.asarray(times,dtype=np.int64))
            self.assertIsNotNone(bases,audit)
            program,origins,_=lower(graph,times,bases)
            machine=frozen.Machine([99]*8,program,frozen.DebugInfo({}))
            machine.run()
            self.assertEqual(machine.mem,[1+choice for choice in choices])
            self.assertEqual(machine.cycle,7+8*span)
            self.assertEqual(len(program),20+16*span)
            self.assertEqual(int(origins[0]),0)

    def test_vector_store_lookup_aligns_both_halves_without_corrupting_rows(self):
        graph=Graph()
        graph.config=dict(lane_allocation=True,compact_main=True,pc_address_pools=True)
        graph.total_table_words=64
        zero,targets,choices,positive,negative=[graph.new() for _ in range(5)]
        sources=[graph.new() for _ in range(4)]
        graph.initial_zero.append(zero)
        times=[]
        def constant(name,dest,value,pc=False):
            op=graph.emit(name,'load',('const',dest,value),[],[(dest,1)])
            if pc: graph.pc_constants.append(op)
            times.append((len(times))//2)
        for block,source in enumerate(sources):
            for j in range(8):
                constant(f'data{block}.{j}',lane(source,j),100+8*block+j)
        for j in range(8):
            constant(f'choice{j}',lane(choices,j),j)
            constant(f'pc{j}',lane(targets,j),8*j+j,pc=True)
            constant(f'positive{j}',lane(positive,j),68+8*j)
            constant(f'negative{j}',lane(negative,j),64+8*j)
        start=graph.emit('entry','flow',('jump_indirect',targets),[(targets,8)],[])
        times.append(32)
        parts=[]
        cache=[sources[q//2] for q in range(8)]
        for j in range(8):
            op=graph.emit(f'row{j}','store',
                          ('lookup_vstore',lane(choices,j),lane(positive,j),lane(negative,j),*cache),
                          [(lane(choices,j),1),(lane(positive,j),1),(lane(negative,j),1),
                           *[(v,8) for v in sources]],[])
            times.append(33+j)
            slot=('jump_indirect',lane(targets,j+1)) if j<7 else ('jump',0)
            jump=graph.emit(f'next{j}','flow',slot,[(slot[1],1)] if j<7 else [],[])
            times.append(33+j)
            parts.append(([(op,0)],jump))
        graph.regions=[dict(start=start,parts=parts,n=8,width=1,cases=8,span=1,table=0)]
        result=graph.new(1)
        graph.emit('finish','alu',('|',result,zero,zero),[(zero,1)],[(result,1)])
        times.append(41)
        bases,audit=allocate(graph,np.asarray(times,dtype=np.int64))
        self.assertIsNotNone(bases,audit)
        program,_,_=lower(graph,times,bases)
        machine=frozen.Machine([0]*136,program,frozen.DebugInfo({}))
        machine.run()
        self.assertEqual(machine.cycle,42)
        self.assertEqual(machine.mem[:64],[0]*64)
        for j in range(8):
            self.assertEqual(machine.mem[68+8*j:72+8*j],list(range(100+4*j,104+4*j)))


if __name__ == '__main__':
    unittest.main()
