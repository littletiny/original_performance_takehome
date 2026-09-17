"""Execute every case of two interleaved tables on the unchanged machine."""
import copy
import unittest

import numpy as np

from optimize import Graph,Scheduler,allocate,bootstrap_cycle,frozen,lane,lower
from search_compact import static_size


def fixture(permuted):
    graph=Graph();graph.config=dict(lane_allocation=True,compact_main=True,pc_address_pools=True)
    graph.total_table_words=128
    zero=graph.new(1);graph.initial_zero.append(zero)
    def scalar(name,value):
        ref=graph.new(1)
        graph.emit(name,'load',('const',ref,value),[],[(ref,1)])
        return ref
    scale_scalar=scalar('scale_scalar',16);scale=graph.new()
    graph.emit('scale','valu',('vbroadcast',scale,scale_scalar),[(scale_scalar,1)],[(scale,8)])
    cache_address=scalar('cache_address',16);cache=graph.new()
    graph.emit('cache','load',('vload',cache,cache_address),[(cache_address,1)],[(cache,8)])
    order=(0,2,4,6,1,3,5,7) if permuted else tuple(range(8))
    for region in range(2):
        choices,offsets,targets,result=[graph.new() for _ in range(4)]
        address=zero if region==0 else scalar('second_input',8)
        graph.emit(f'choices{region}','load',('vload',choices,address),[(address,1)],[(choices,8)])
        for j in range(8):
            graph.emit(f'pc{region}.{j}','load',('const',lane(offsets,j),14+8*region+j),[],[(lane(offsets,j),1)])
        graph.emit(f'targets{region}','valu',('multiply_add',targets,choices,scale,offsets),
                   [(choices,8),(scale,8),(offsets,8)],[(targets,8)])
        output=scalar(f'output_address{region}',24+8*region)
        first_unit=len(graph.units)
        start=graph.emit(f'entry{region}','flow',('jump_indirect',targets),[(targets,1)],[])
        relative=[(start,0)];parts=[]
        for p,j in enumerate(order):
            dest,choice=lane(result,j),lane(choices,j)
            sources=[lane(cache,q) for q in range(8)]
            op=graph.emit(f'copy{region}.{j}','alu',('lookup_copy',dest,choice,*sources),
                          [(choice,1),*[(v,1) for v in sources]],[(dest,1)])
            relative.append((op,p+1))
            slot=('jump_indirect',lane(targets,order[p+1])) if p<7 else ('jump',0)
            jump=graph.emit(f'jump{region}.{p}','flow',slot,[(slot[1],1)] if p<7 else [],[])
            relative.append((jump,p+1));parts.append(([(op,0)],jump))
        graph.units[first_unit:]=[relative]
        graph.regions.append(dict(start=start,parts=parts,n=8,width=1,cases=8,span=1,
                                  table=8*region,lane_stride=1,case_stride=16,table_lanes=list(order)))
        graph.emit(f'output{region}','store',('vstore',output,result),[(output,1),(result,8)],[])
    scheduler=Scheduler(graph)
    _,units,_=scheduler.search(iterations=5,seed=914,noise=.1)
    times=scheduler.op_times(units);bases,audit=allocate(graph,times)
    assert bases is not None,audit
    return graph,times,bases


class InterleavedPCTests(unittest.TestCase):
    def test_every_case_preserves_outputs_and_memory_in_both_lane_orders(self):
        for permuted in (False,True):
            graph,times,bases=fixture(permuted)
            program,_,_=lower(graph,times,bases)
            data=[0x12345678+51*j for j in range(8)]
            for shift in range(8):
                a=[(j+shift)%8 for j in range(8)]
                b=[(3*j+shift)%8 for j in range(8)]
                memory=a+b+data+[0]*16+[0xdeadbeef]
                machine=frozen.Machine(memory,program,frozen.DebugInfo({}));machine.run()
                self.assertEqual(machine.mem,a+b+data+[data[q] for q in a+b]+[0xdeadbeef])
                self.assertEqual(machine.cycle,int(times.max())+1)

    def test_overlapping_table_banks_are_rejected_before_execution(self):
        graph,times,bases=fixture(False)
        bad=copy.deepcopy(graph);bad.regions[1]['table']=0
        with self.assertRaisesRegex(AssertionError,'Overlapping dispatch cases'):
            lower(bad,times,bases)

    def test_prologue_keeps_absolute_pcs_and_pauses_before_writing_memory(self):
        graph,times,_=fixture(True)
        graph.config['pc_prologue']=13
        zero=graph.initial_zero[0];value=graph.new(1)
        graph.emit('first_flow','flow',('add_imm',value,zero,19),[(zero,1)],[(value,1)])
        pause=graph.emit('initial.pause','flow',('pause',),[],[])
        graph.control.extend((pause,i,1) for i,op in enumerate(graph.ops) if op[0]=='store')
        times=np.append(times,[0,1])
        bases,audit=allocate(graph,times)
        self.assertIsNotNone(bases,audit)
        bootstrap=bootstrap_cycle(graph,times)
        self.assertGreater(bootstrap,1)
        program,origins,_=lower(graph,times,bases)
        self.assertEqual(program[0]['flow'][0][0],'add_imm')
        self.assertEqual(program[bootstrap]['flow'][0][0],'jump')
        self.assertEqual(len(program),static_size(graph,int(times.max())+1,times))
        self.assertEqual(list(origins[:bootstrap+1]),list(range(bootstrap+1)))
        a=list(range(8));b=list(reversed(a));data=[0x12567800+17*j for j in range(8)]
        memory=a+b+data+[0]*16+[0xdeadbeef]
        machine=frozen.Machine(memory,program,frozen.DebugInfo({}));machine.run()
        self.assertEqual(machine.mem,memory)
        self.assertEqual(machine.cycle,2)
        machine.run()
        self.assertEqual(machine.mem,a+b+data+[data[q] for q in a+b]+[0xdeadbeef])
        self.assertEqual(machine.cycle,int(times.max())+1)


if __name__=='__main__':unittest.main()
