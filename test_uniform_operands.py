"""Scalar hash stages may share constant operands without a vector broadcast."""
import json
import unittest

from balance_resources import balance
from optimize import ROOT, build, frozen, verify_semantics


class UniformOperandTests(unittest.TestCase):
    def test_h2_constant_broadcast_is_eliminated_and_stays_scalar(self):
        source=ROOT/'results/compact_915'
        config=json.loads((source/'config.json').read_text())
        config.update(share_uniform_operands=True,scalar_constant_labels=['h2.a'])
        config=balance(config,source,0)
        graph=build(config)
        self.assertNotIn(f'broadcast.{frozen.HASH_STAGES[1][1]}',graph.names)
        stages=[op for name,op in zip(graph.names,graph.ops) if '.h2.a.lane' in name]
        self.assertEqual(len(stages),512*8)
        self.assertEqual(len({op[1][3] for op in stages}),1)
        self.assertTrue(all(op[0]=='alu' for op in stages))
        verify_semantics(graph,(0,1))


if __name__=='__main__':
    unittest.main()
