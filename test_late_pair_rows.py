"""Frozen-machine checks for overlapping child-pair stores and dead lanes."""
import copy
import json
from pathlib import Path
import unittest

import numpy as np

from optimize import (
    Graph, WideV, Scheduler, allocate, audit_pair_padding, build, frozen, lane,
    lower, merge_dispatch_regions, verify_semantics,
)


def fixture(merged):
    g=Graph(ref_type=WideV)
    g.config=dict(lane_allocation=True,compact_main=True,pc_address_pools=True,
                  store_children=True,merge_chains=[],lane_allocation_trials=16)
    g.total_table_words=1024
    scalars={}
    def scalar(value):
        if value not in scalars:
            v=g.new(1)
            g.emit(f'constant.{value}','load',('const',v,value),[],[(v,1)])
            scalars[value]=v
        return scalars[value]
    def constant(name,dest,value,pc=False):
        op=g.emit(name,'load',('const',dest,value),[],[(dest,1)])
        if pc:g.pc_constants.append(op)

    source=[g.new(14),g.new(14)]
    data=[(0xf0123001+0x1234567*j)&0xffffffff for j in range(16)]
    for block in range(2):
        for j in range(14):
            value=data[8*block+j] if j<8 else 0xdead0000+16*block+j
            constant(f'data{block}.{j}',lane(source[block],j),value)
    cache=[lane(source[q//4],2*(q%4)) for q in range(8)]
    pointers=[[scalar(16+24*s+2*j) for j in range(8)] for s in range(2)]
    zero=g.new()
    z=scalar(0)
    g.emit('zero','valu',('vbroadcast',zero,z),[(z,1)],[(zero,8)])
    setups=[]
    expected=[]
    for region in range(2):
        targets=g.new()
        conds=[[g.new(),g.new()] for _ in range(2)]
        for j in range(8):
            constant(f'pc{region}.{j}',lane(targets,j),512*region+64*j+j*8+(7-j),True)
        for s in range(2):
            values=[]
            for j in range(8):
                q=j if s==0 else 7-j
                bit=(j+s+region)&1
                constant(f'bit{region}.{s}.{j}',lane(conds[s][j//4],2*(j%4)),bit)
                constant(f'poison{region}.{s}.{j}',lane(conds[s][j//4],2*(j%4)+1),0xff000000+j)
                values.append(data[2*q+bit])
            expected.extend(values)
        setups.append((targets,conds))

    previous=[]
    load_rows=[]
    selected_outputs=[]
    for region,(targets,conds) in enumerate(setups):
        start_unit=len(g.units)
        start=g.emit(f'r14.g{2*region}.entry','flow',('jump_indirect',targets),[(targets,8)],[])
        g.control.extend((before,start,-1-i//2) for i,before in enumerate(previous))
        relative={start:0}
        stores=[[],[]]
        parts=[]
        for j in range(8):
            rows=[]
            for s in range(2):
                q=scalar(j if s==0 else 7-j)
                addr=pointers[s][j]
                op=g.emit(f'r{region}.row{j}.{s}','store',('lookup_pair_store',addr,q,*cache),
                          [(addr,1),(q,1),*[(v,2) for v in cache]],[])
                stores[s].append(op);rows.append((op,s));relative[op]=j+1
            slot=('jump_indirect',lane(targets,j+1)) if j<7 else ('jump',0)
            jump=g.emit(f'r{region}.jump{j}','flow',slot,[(slot[1],1)] if j<7 else [],[])
            relative[jump]=j+1;parts.append((rows,jump))
        # Some scalar choice constants were first requested while building the
        # cases. They are independent units, not part of the lookup body.
        body=set(relative)
        independent=[unit for unit in g.units[start_unit:] if unit[0][0] not in body]
        g.units[start_unit:]=independent+[list(relative.items())]
        loads=[]
        loaded=[[g.new(9),g.new(9)] for _ in range(2)]
        for block in range(2):
            for s in range(2):
                dst=loaded[s][block];addr=pointers[s][4*block]
                op=g.emit(f'r{region}.load{block}.{s}','load',('vload',dst,addr),[(addr,1)],[(dst,8)])
                g.control.extend((before,op,1) for j,before in enumerate(stores[s])
                                 if 2*j<8*block+8 and 8*block<2*j+8)
                loads.append(op)
        g.regions.append(dict(start=start,parts=parts,n=8,width=2,span=1,cases=64,
                              table=512*region,groups=[2*region,2*region+1],round=14,
                              temp_loads=loads,temp_buffer=0,temp_fields=4))
        previous=loads
        load_rows.extend(loads)
        for s in range(2):
            chosen=[]
            for block in range(2):
                no=loaded[s][block];yes=lane(no,1);cond=conds[s][block];dest=g.new()
                g.emit(f'r{region}.choose{s}.{block}','flow',('vselect_even',dest,cond,yes,no),
                       [(lane(v,j),1) for v in (cond,yes,no) for j in (0,2,4,6)],[(dest,8)])
                selected_outputs.append(dest)
                chosen.extend(lane(dest,j) for j in (0,2,4,6))
            result=g.new()
            for j,src in enumerate(chosen):
                dst=lane(result,j)
                g.emit(f'r{region}.gather{s}.{j}','alu',('|',dst,src,src),[(src,1)],[(dst,1)])
            addr=scalar(80+16*region+8*s)
            g.emit(f'r{region}.output{s}','store',('vstore',addr,result),[(addr,1),(result,8)],[])
    for offset in range(0,48,8):
        addr=scalar(16+offset)
        op=g.emit(f'clear{offset}','store',('vstore',addr,zero),[(addr,1),(zero,8)],[])
        g.control.extend((before,op,0) for before in load_rows)
    if merged:merge_dispatch_regions(g,2)
    audit_pair_padding(g)
    return g,expected,selected_outputs


class LatePairTests(unittest.TestCase):
    def test_pair_rows_are_independent_of_extended_dispatch_buffers(self):
        cfg=json.loads((Path(__file__).parent/'results/compact_915/config.json').read_text())
        cfg.update(late_pair_groups=[0,4],dispatch_spans=[[3,2,3]],
                   heap_keep_levels=[4,5,7])
        g=build(cfg)
        regions={(r['round'],r['groups'][0]):r for r in g.regions}
        extended=regions[3,2]['temp_buffer']
        for k in (0,4):
            self.assertNotEqual(extended,regions[14,k]['temp_buffer'])
        verify_semantics(g,(0,1))

    def test_overlapping_pairs_preserve_both_choices_and_clear_padding(self):
        for merged in (False,True):
            g,expected,_=fixture(merged)
            scheduler=Scheduler(g)
            _,units,_=scheduler.search(iterations=5,seed=917,noise=.1)
            times=scheduler.op_times(units)
            allocated,audit=allocate(g,times)
            self.assertIsNotNone(allocated,audit)
            fixed={};cursor=0
            for v,n in enumerate(g.sizes):
                fixed[v]=cursor;cursor+=n+1
            self.assertLessEqual(cursor,1536)
            for bases in (fixed,allocated):
                program,_,_=lower(g,times,bases)
                memory=[0x55555555]*128
                memory[16:64]=[0]*48
                memory[80:112]=[0]*32
                machine=frozen.Machine(memory,program,frozen.DebugInfo({}))
                machine.run()
                self.assertEqual(machine.cycle,int(times.max())+1)
                self.assertEqual(machine.mem[80:112],expected)
                self.assertEqual(machine.mem[16:64],[0]*48)
                self.assertEqual(machine.mem[:16],memory[:16])
                self.assertEqual(machine.mem[64:80],memory[64:80])
                self.assertEqual(machine.mem[112:],memory[112:])

    def test_odd_selection_output_cannot_be_observed(self):
        g,_,outputs=fixture(False)
        bad=g.new(1);src=lane(outputs[0],1)
        g.emit('observe_odd','alu',('|',bad,src,src),[(src,1)],[(bad,1)])
        with self.assertRaises(AssertionError):audit_pair_padding(g)

    def test_delayed_old_second_block_requires_reuse_dependency(self):
        cfg=json.loads((Path(__file__).parent/'results/compact_917/config.json').read_text())
        g=build(dict(cfg,late_pair_groups=[0,4],late_pair_buffers=1))
        named={name:i for i,name in enumerate(g.names)}
        old_start=named['r14.g0.dispatch.jump0']
        old_load=named['r14.g0.dispatch.pair_load1.0']
        new_start=named['r14.g4.dispatch.jump0']
        edge=(old_load,new_start,-2)
        self.assertIn(edge,g.control)
        g.control.append((old_start,old_load,100))
        verify_semantics(g,(0,))
        bad=copy.deepcopy(g)
        verify_semantics(bad,(0,))
        bad.control.remove(edge)
        with self.assertRaises(AssertionError) as error:verify_semantics(bad,(0,))
        self.assertIsInstance(error.exception.args[0],tuple)
        self.assertEqual(error.exception.args[0][1:5],(15,0,0,4))


if __name__=='__main__':
    unittest.main()
