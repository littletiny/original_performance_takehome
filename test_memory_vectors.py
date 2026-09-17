"""A delayed broadcast read must precede reuse of its memory row."""
import json
import unittest

from optimize import ROOT, build, verify_semantics


class MemoryVectorTests(unittest.TestCase):
    def test_broadcast_rows_are_disjoint_from_late_child_rows(self):
        config=json.loads((ROOT/'results/compact_915/config.json').read_text())
        graph=build(dict(config,late_pair_groups=[20,24],late_pair_buffers=2))
        self.assertTrue(graph.memory_vectors)
        for row in graph.memory_vectors:
            self.assertGreaterEqual(row['address'],2054+96)
            self.assertLessEqual(row['address']+8,2294)
        verify_semantics(graph,(0,1))

    def test_delayed_runtime_root_read_survives_buffer_reuse(self):
        config=json.loads((ROOT/'results/compact_917/config.json').read_text())
        graph=build(dict(config,memory_vectors=['root.raw','root.bias']))
        named={name:i for i,name in enumerate(graph.names)}
        first=named['memory_vector.root.raw.store0']
        read=named['root.raw']
        overwrites=[named[f'memory_vector.root.bias.store{j}'] for j in range(8)]
        graph.control.append((first,read,100))
        verify_semantics(graph,(0,))
        for overwrite in overwrites:
            graph.control.remove((read,overwrite,0))
        with self.assertRaises(AssertionError) as error:
            verify_semantics(graph,(0,))
        self.assertIsInstance(error.exception.args[0],tuple)
        self.assertEqual(error.exception.args[0][1:4],(0,0,0))


if __name__=='__main__':
    unittest.main()
