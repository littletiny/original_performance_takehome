"""Reference-checked hazards when a lookup overwrites only one buffer field."""
import json
import unittest

from optimize import ROOT,build,verify_semantics


class MemoryOrderTests(unittest.TestCase):
    def test_unwritten_second_field_survives_partial_buffer_reuse(self):
        config=json.loads((ROOT/'results/compact_918/config.json').read_text())
        graph=build(dict(config,grand_row_groups=[7],grand_row_buffers=1))
        named={name:i for i,name in enumerate(graph.names)}
        first=named['r3.g2.dispatch.jump0']
        delayed=named['r3.g2.dispatch.child_vector1']
        later=named['r3.g12.dispatch.jump0']
        # Group 7 overwrites field zero only. Keep group 2's old field one
        # pending until after group 12 could otherwise overwrite it.
        graph.control.append((first,delayed,100))
        verify_semantics(graph,(0,))
        before=len(graph.control)
        graph.control=[edge for edge in graph.control if edge[:2]!=(delayed,later)]
        self.assertEqual(before-len(graph.control),1)
        with self.assertRaises(AssertionError) as error:
            verify_semantics(graph,(0,))
        self.assertIsInstance(error.exception.args[0],tuple)
        self.assertEqual(error.exception.args[0][1:4],(4,2,0))


if __name__=='__main__':
    unittest.main()
