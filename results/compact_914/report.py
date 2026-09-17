"""Rebuild accounting and bind validation artifacts to the promoted source."""
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import re
import runpy
import subprocess
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from optimize import build,counts
from plot_utilization import profile
from resource_bounds import analyze


def main():
    source=Path(__file__).parent
    graph=build(json.loads((source/'config.json').read_text()))
    bound=analyze(source)
    classify=runpy.run_path(str(ROOT/'results/work_inventory_918/run.py'))['classify']
    categories=defaultdict(Counter)
    for name,op in zip(graph.names,graph.ops):
        if op[0] not in ('alu','valu'):continue
        category,detail=classify(name.replace('.pair_address','.address'),op)
        categories[category][detail]+=8 if op[0]=='valu' else 1
    accounting=counts(graph)
    assert sum(sum(row.values()) for row in categories.values())==accounting['weighted_alu_valu']==54536
    accounting.update(cycles=914,bound_scope='Counts for this fixed graph; not a global task lower bound.',
                      work_categories_W={key:dict(total=sum(value.values()),details=dict(value))
                                         for key,value in categories.items()})
    (source/'analysis.json').write_text(json.dumps(accounting,indent=2)+'\n')
    current=(ROOT/'perf_takehome.py').read_text()
    previous=subprocess.check_output(['git','show','1441bfa:perf_takehome.py'],cwd=ROOT,text=True)
    def outside(text):
        before,rest=text.split('# SUB900_CANDIDATE_BEGIN\n',1)
        _,after=rest.split('# SUB900_CANDIDATE_END\n',1)
        return before,after.lstrip('\n')
    assert outside(current)==outside(previous)
    digest=hashlib.sha256(current.encode()).hexdigest()
    primary=json.loads((source/'verification.json').read_text())
    extra=json.loads((source/'verification_extra.json').read_text())
    initial=json.loads((source/'port_utilization.json').read_text())
    assert extra['source_sha256']==initial['source_sha256']==digest
    assert primary['cycles']==extra['cycles']==initial['cycles']==914
    rows0=np.loadtxt(source/'port_utilization.csv',delimiter=',',skiprows=1,dtype=np.int64)
    rows1,second=profile(1)
    assert np.array_equal(rows0[:,2:],rows1[:,2:])
    assert not np.array_equal(rows0[:,1],rows1[:,1])
    (source/'port_utilization_seed1.json').write_text(json.dumps(second,indent=2)+'\n')
    tests={}
    for name,expected in (('submission',9),('native',3),('regression',24)):
        log=(source/f'{name}_tests.log').read_text()
        count=int(re.search(r'Ran (\d+) tests?',log)[1])
        assert count==expected and '\nOK\n' in log
        tests[f'{name}_tests_passed']=count
    report=dict(source_sha256=digest,production_changes_confined_to_generated_block=True,
                cycles=914,static_bundles=primary['static_bundles'],
                static_slot_operations=primary['static_slots'],weighted_arithmetic=54536,
                previous_915_weighted_arithmetic=54577,**tests,
                full_checkpoint_seeds=primary['seeds']+extra['seeds'],checkpoints_per_seed=20480,
                source_bound_port_profile=True,seed0_seed1_identical_per_cycle_ports=True,
                seed0_seed1_different_pc_paths=True,strict_sub900_achieved=False,
                static_bundle_cap=12000,fixed_graph_lower_bound=bound['bound'],
                strict_sub900_counts_only_remaining_reduction=54536-60*899)
    assert report['static_bundles']==10807 and bound['bound']==911
    (source/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
