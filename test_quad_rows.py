"""Packed quartet selection, buffer lifetime and real source-span checks."""
import copy
import json
from pathlib import Path
import unittest

from optimize import audit_pair_padding,build,verify_semantics


def configuration(groups,**changes):
    root=Path(__file__).parent
    base=json.loads((root/'results/compact_914_pc/config.json').read_text())
    members={k for group in groups for k in group}
    cfg=dict(base,quad_dispatch_groups=groups,quad_row_buffers=1,
             pair_even_odd_order=True,all_even_odd_order=True,
             prefetch_madd_groups=[p for p in base['prefetch_madd_groups'] if p[1] not in members],
             constant_expressions=None,memory_vectors=[],memory_vector_order=[])
    cfg.update(changes)
    return cfg


class QuadRowTests(unittest.TestCase):
    def test_folded_and_unfolded_paths_with_independent_row_layouts(self):
        for folded,natural,buffers in ((False,False,1),(True,True,2)):
            graph=build(configuration([[3,7],[10,11,14]],quad_fold_path=folded,
                                      natural_pc_order=natural,all_even_odd_order=not natural,
                                      quad_row_buffers=buffers,quad_row_order=[7,3,10,11,14],
                                      early_pair_groups=[24,28],late_pair_groups=[20,24],
                                      late_pair_order=[24,20]))
            verify_semantics(graph,(0,1))

    def test_delayed_last_quartet_read_requires_reuse_edge(self):
        graph=build(configuration([[2,3]]))
        named={name:i for i,name in enumerate(graph.names)}
        start=named['r3.g2.dispatch.jump0']
        read=named['r3.g2.dispatch.quad_load1.3']
        store=named['r3.g3.dispatch.quad_grand3']
        edge=(read,store,0)
        self.assertIn(edge,graph.control)
        graph.control.append((start,read,100))
        verify_semantics(graph,(0,1))
        bad=copy.deepcopy(graph);bad.control.remove(edge)
        with self.assertRaises(AssertionError) as error:verify_semantics(bad,(0,))
        self.assertIsInstance(error.exception.args[0],tuple)
        self.assertEqual(error.exception.args[0][1:4],(5,2,0))

    def test_chained_producers_read_old_blocks_before_reuse(self):
        cfg=configuration([[2,3]],merge_chains=[[[3,2],[3,3]]])
        graph=build(cfg)
        verify_semantics(graph,(0,1))

    def test_quad_vstore_accounts_for_all_eight_source_words(self):
        graph=build(configuration([[3,7]]))
        op=next(op for op in graph.ops if op[1][0]=='lookup_quad_store')
        source=next(source for source in op[1][3:] if source.off==4)
        self.assertEqual(graph.sizes[source.vid],12)
        graph.sizes[source.vid]=8
        with self.assertRaises(AssertionError):audit_pair_padding(graph)

    def test_rows_cannot_overlap_the_shallow_cache_without_memory_broadcasts(self):
        with self.assertRaises(AssertionError):
            build(configuration([[3,7]],quad_row_buffers=3,late_pair_groups=[20,24],late_pair_buffers=2))

    def test_explicit_regular_buffer_order_ignores_cached_quad_consumers(self):
        cfg=configuration([[3,7]])
        graph=build(cfg)
        cfg['temp_region_order']=[[r['round'],r['groups'][0]] for r in graph.regions if r['round']!=5]
        verify_semantics(build(cfg),(0,1))


if __name__=='__main__':unittest.main()
