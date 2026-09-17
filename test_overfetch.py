"""Keep the seven padding lanes of each overfetch logically unobserved."""
import json
import unittest

from optimize import ROOT, audit_overfetch_padding, build, lane, verify_semantics


class OverfetchTests(unittest.TestCase):
    def test_per_level_coordinates_preserve_mixed_transitions(self):
        config=json.loads((ROOT/'results/compact_914/config.json').read_text())
        groups=list(range(0,32,2))
        for levels in ({'8':groups,'9':'all','10':'all'},
                       {'8':[],'9':groups,'10':'all'}):
            # Some address constants disappear when every group changes
            # representation, so keep only applicable saved setup recipes.
            probe=build(dict(config,overfetch_by_level=levels,memory_vectors=[],
                             memory_vector_order=[],constant_expressions=None))
            available=set(probe.names)
            cfg=dict(config,overfetch_by_level=levels,
                     memory_vectors=[n for n in config['memory_vectors'] if n in available],
                     memory_vector_order=[n for n in config['memory_vector_order'] if n in available],
                     constant_expressions=None)
            graph=build(cfg)
            verify_semantics(graph,(0,1))

    def test_shifted_address_recurrence_and_padding_use(self):
        config=json.loads((ROOT/'results/compact_917/config.json').read_text())
        graph=build(dict(config,overfetch_groups='all'))
        self.assertEqual(len(graph.overfetch_values),3*32*8)
        self.assertFalse(any(name.startswith(('r8.','r9.')) and name.endswith('address.aux')
                             for name in graph.names))
        verify_semantics(graph,(0,1))
        source=lane(graph.overfetch_values[0],5)
        dest=graph.new(1)
        graph.emit('observe_padding','alu',('^',dest,source,source),[(source,1)],[(dest,1)])
        with self.assertRaises(AssertionError):audit_overfetch_padding(graph)


if __name__=='__main__':
    unittest.main()
