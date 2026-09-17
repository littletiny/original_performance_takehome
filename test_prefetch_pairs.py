"""Real eight-word writes and ordering for early adjacent-child prefetch."""
import copy
import json
import unittest

from optimize import (
    ROOT, Graph, PackedV, Scheduler, allocate, audit_overfetch_padding,
    audit_pair_padding, build, frozen, lane, lower, verify_semantics,
)


class PrefetchPairTests(unittest.TestCase):
    def test_overlapping_vloads_keep_both_children_with_allocated_scratch(self):
        for phase in (0,1):
            g=Graph(ref_type=PackedV)
            g.config=dict(lane_allocation=True,compact_main=False)
            g.total_table_words=0
            def constant(value):
                dst=g.new(1)
                g.emit(f'constant.{value}','load',('const',dst,value),[],[(dst,1)])
                return dst
            addresses=[constant(4+2*q) for q in (5,0,7,2,6,1,4,3)]
            output=constant(48)
            conds=[g.new(),g.new()]
            for j in range(8):
                for offset,value in ((0,(j+phase)&1),(1,0xdead0000+j)):
                    dst=lane(conds[j//4],2*(j%4)+offset)
                    g.emit(f'bit.{j}.{offset}','load',('const',dst,value),[],[(dst,1)])
            row=g.new(22)
            loads=[]
            for j,addr in enumerate(addresses):
                dst=lane(row,2*j)
                op=g.emit(f'load{j}','load',('vload',dst,addr),[(addr,1)],[(dst,8)])
                if loads:g.control.append((loads[-1],op,1))
                loads.append(op)
            result=g.new()
            for block in range(2):
                no=lane(row,8*block);yes=lane(no,1);cond=conds[block];dest=g.new()
                g.emit(f'choose{block}','flow',('vselect_even',dest,cond,yes,no),
                       [(lane(v,j),1) for v in (cond,yes,no) for j in (0,2,4,6)],[(dest,8)])
                for j in range(4):
                    src=lane(dest,2*j);dst=lane(result,4*block+j)
                    g.emit(f'gather{block}.{j}','alu',('|',dst,src,src),[(src,1)],[(dst,1)])
            g.emit('output','store',('vstore',output,result),[(output,1),(result,8)],[])
            g.prefetch_pair_rows=[dict(value=row,loads=[g.names[i] for i in loads])]
            audit_pair_padding(g);audit_overfetch_padding(g)
            scheduler=Scheduler(g)
            _,units,_=scheduler.search(iterations=5,seed=915,noise=.1)
            times=scheduler.op_times(units)
            allocated,report=allocate(g,times)
            self.assertIsNotNone(allocated,report)
            fixed={};cursor=0
            for v,size in enumerate(g.sizes):
                fixed[v]=cursor;cursor+=size+1
            for bases in (fixed,allocated):
                program,_,_=lower(g,times,bases)
                initial=[(0xf1234567+0xabcdef*j)&0xffffffff for j in range(64)]
                machine=frozen.Machine(initial,program,frozen.DebugInfo({}))
                machine.run()
                expected=initial.copy()
                expected[48:56]=[initial[4+2*q+((j+phase)&1)]
                                 for j,q in enumerate((5,0,7,2,6,1,4,3))]
                self.assertEqual(machine.mem,expected)
                self.assertEqual(machine.cycle,int(times.max())+1)

    def test_missing_load_order_clobbers_an_observed_child(self):
        cfg=json.loads((ROOT/'results/compact_915/config.json').read_text())
        cfg.update(fold_path4_groups=[0],prefetch_pair_groups=[0])
        g=build(cfg)
        named={name:i for i,name in enumerate(g.names)}
        first=named['r3.g0.pair_load0']
        delayed=named['r3.g0.pair_load3']
        following=named['r3.g0.pair_load4']
        choose=named['r5.g0.prefetched_pair.half1']
        edge=(delayed,following,1)
        self.assertIn(edge,g.control)
        g.control.extend(((first,delayed,100),(delayed,choose,1)))
        verify_semantics(g,(0,1))
        bad=copy.deepcopy(g)
        bad.control.remove(edge)
        with self.assertRaises(AssertionError) as error:verify_semantics(bad,(0,))
        self.assertIsInstance(error.exception.args[0],tuple)
        self.assertEqual(error.exception.args[0][1:4],(5,0,0))


if __name__=='__main__':
    unittest.main()
