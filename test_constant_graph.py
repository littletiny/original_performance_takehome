"""Constant DAG rewrites preserve machine values and have no forward cycles."""
import json
import unittest

import numpy as np

from constant_graph import MASK,resynthesize
from optimize import ROOT,Graph,Scheduler,allocate,build,frozen,lower,verify_semantics


class ConstantGraphTests(unittest.TestCase):
    def test_forward_sources_and_wrapping_values_on_real_machine(self):
        g=Graph();g.config=dict(lane_allocation=True,compact_main=True);g.total_table_words=0
        known={}
        for number in (1,4,16,0xfffffffc):
            ref=g.new(1);known[number]=ref
            g.emit(f'constant.{number}','load',('const',ref,number),[],[(ref,1)])
        for number in range(17,41):
            ref=g.new(1);known[number]=ref;a,b=known[number-1],known[1]
            g.emit(f'constant.{number}','alu',('+',ref,a,b),[(a,1),(b,1)],[(ref,1)])
        for number in (0xfffffffd,0xfffffffe,0xffffffff,0):
            ref=g.new(1);known[number]=ref;a,b=known[(number-1)&MASK],known[1]
            g.emit(f'constant.{number}','alu',('+',ref,a,b),[(a,1),(b,1)],[(ref,1)])
        expected=list(known)
        for address,number in enumerate(expected):
            ptr=g.new(1);value=known[number]
            g.emit(f'output_address.{address}','load',('const',ptr,address),[],[(ptr,1)])
            g.emit(f'output.{address}','store',('store',ptr,value),[(ptr,1),(value,1)],[])
        resynthesize(g,known)
        self.assertTrue(g.constant_rewrites)
        self.assertLess(max(row['new_depth'] for row in g.constant_rewrites),10)
        scheduler=Scheduler(g)
        _,units,_=scheduler.search(iterations=10,seed=915,noise=.1)
        times=scheduler.op_times(units)
        bases,report=allocate(g,times)
        self.assertIsNotNone(bases,report)
        program,_,_=lower(g,times,bases)
        machine=frozen.Machine([0]*len(expected)+[0xdeadbeef],program,frozen.DebugInfo({}))
        machine.run()
        self.assertEqual(machine.mem,expected+[0xdeadbeef])
        self.assertEqual(machine.cycle,int(times.max())+1)

    def test_runtime_header_anchors_preserve_full_hash_and_memory(self):
        cfg=json.loads((ROOT/'results/compact_915/config.json').read_text())
        g=build(dict(cfg,resynthesize_constants=True))
        self.assertGreater(len(g.constant_rewrites),50)
        verify_semantics(g,(0,1))


if __name__=='__main__':unittest.main()
