"""A delayed broadcast read must precede reuse of its memory row."""
import json
import unittest

from optimize import ROOT, build, verify_semantics


class MemoryVectorTests(unittest.TestCase):
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
