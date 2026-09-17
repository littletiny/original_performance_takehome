"""Rebuild fixed-graph bounds and compare them with verified schedules."""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from resource_bounds import analyze


def main():
    rows=[]
    for relative in ('results/compact_915','results/bound899_915/candidate_009'):
        source=ROOT/relative
        bound=analyze(source)
        verified=json.loads((source/'verification.json').read_text())
        assert verified['cycles']==bound['saved_schedule_cycles']
        rows.append(dict(source=relative,cycles=verified['cycles'],
                         static_bundles=verified['static_bundles'],
                         resource_window_bound=bound['bound'],
                         observed_gap=verified['cycles']-bound['bound'],
                         weighted_alu_valu=bound['counts']['weighted_alu_valu'],
                         graph_sha256=bound['graph_sha256'],
                         verified_seeds=verified['seeds']))
    work=rows[0]['weighted_alu_valu']
    report=dict(strict_cycle_cap=899,static_bundle_cap=12000,measured=rows,
                margins=[dict(bound_target=target,max_gap_for_899=899-target,
                              capacity_only_work_ceiling=60*target,
                              necessary_work_cut_from_915=work-60*target)
                         for target in (899,895,890)],
                submitted_source_sha256=hashlib.sha256((ROOT/'perf_takehome.py').read_bytes()).hexdigest(),
                caveats=[
                    'A resource bound is necessary, not an attainable schedule.',
                    'Observed gaps include heuristic search quality; they are not proven unavoidable overhead.',
                    'Work ceilings use only ALU+8*VALU capacity; startup, tails, other ports and scratch still constrain schedules.',
                    'These are fixed-graph bounds, not lower bounds for every implementation of the task.'])
    (Path(__file__).parent/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
