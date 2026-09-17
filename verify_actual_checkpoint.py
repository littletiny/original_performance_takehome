"""Check the submitted KernelBuilder against a retained, verified schedule."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from optimize import ROOT,build,allocate,lower,verify_frozen,pt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('--seeds',type=int,nargs='+',default=[901,12345])
    args=parser.parse_args()
    cfg=json.loads((args.source/'config.json').read_text())
    saved=json.loads((args.source/'verification.json').read_text())
    graph=build(cfg);times=np.load(args.source/'best.npz')['times']
    bases,audit=allocate(graph,times)
    assert bases is not None,audit
    expected,origins,_=lower(graph,times,bases)
    positions=np.flatnonzero(origins==saved['pause_cycle'])
    assert len(positions)==1
    initial=expected[int(positions[0])]
    assert not initial.get('flow') and not initial.get('store')
    initial['flow']=[('pause',)]
    builder=pt.KernelBuilder();builder.build_kernel(10,2047,256,16)
    actual=builder.instrs
    assert actual==expected,'Submitted builder differs from the full verified program'
    assert len(actual)<=12000
    report=verify_frozen(graph,times,bases,actual,origins,args.seeds)
    report.update(complete_actual_builder_program_equal=True,
                  source_sha256=hashlib.sha256((ROOT/'perf_takehome.py').read_bytes()).hexdigest())
    (args.source/'verification_extra.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
