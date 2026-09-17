"""Test four-operation AND/OR forms absorbing fixed hash boundary XORs."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import random
import time

import z3

from synthesize_affine_outer import target


def solve(name,operator,addend,bits,seconds):
    a,b,d,e,p,q=z3.BitVecs('a b d e p q',bits)
    solver=z3.Solver()
    if addend=='constant':solver.add(p&1==1)
    mask=(1<<bits)-1
    points=set(range(17))
    for bit in range(bits):
        points.update(((1<<bit)+delta)&mask for delta in (-1,0,1,2,3,7))
    rng=random.Random(917)
    points.update(rng.randrange(mask+1) for _ in range(32))

    def inner(x):
        aa=a*x+b;bb=d*x+e
        return aa&bb if operator=='and' else aa|bb

    def add(value):
        expected=int(target(name,value))&mask
        if addend=='constant':
            # p is the inverse outer multiplier, q the transformed addend.
            solver.add(inner(value)==p*expected+q)
        else:
            solver.add(p*inner(value)+(value if addend=='x' else a*value+b)==expected)

    for x in sorted(points):add(x)
    report=dict(target=name,operator=operator,addend=addend,bits=bits,
                template=f'p*((a*x+b) {operator.upper()} (d*x+e))+'+
                         {'constant':'q','x':'x','A':'(a*x+b)'}[addend],
                necessary_inputs=sorted(points),counterexamples=[],
                scope='Named four-operation family; arbitrary coefficients; not global hash optimality.')
    begin=time.monotonic();deadline=begin+seconds
    for iteration in range(100):
        remaining=deadline-time.monotonic()
        if remaining<=0:
            report.update(status='unknown',reason='Projection time budget exhausted')
            break
        solver.set(timeout=max(1,int(remaining*1000)))
        status=solver.check();report.update(status=str(status),iterations=iteration+1)
        if status!=z3.sat:
            if status==z3.unknown:report['reason']=solver.reason_unknown()
            break
        model=solver.model()
        aa,bb,dd,ee,pp,qq=[model.eval(v,model_completion=True).as_long() for v in (a,b,d,e,p,q)]
        if addend=='constant':
            pp=pow(pp,-1,1<<bits);qq=(-pp*qq)&mask
        report['candidate']=dict(a=aa,b=bb,d=dd,e=ee,p=pp,q=qq)
        x=z3.BitVec('verify_x',bits)
        first=aa*x+bb;second=dd*x+ee
        body=first&second if operator=='and' else first|second
        value=pp*body+({'constant':qq,'x':x,'A':first}[addend])
        verifier=z3.Solver()
        remaining=deadline-time.monotonic()
        if remaining<=0:
            report.update(status='unknown',reason='No time left for universal verification')
            break
        verifier.set(timeout=max(1,int(remaining*1000)))
        verifier.add(value!=target(name,x))
        status=verifier.check()
        if status==z3.unsat:
            report.update(status='equivalent',universal_bits=bits)
            break
        if status==z3.unknown:
            report.update(status='unknown',reason=verifier.reason_unknown())
            break
        x=verifier.model().eval(x).as_long()
        assert x not in points
        points.add(x);report['counterexamples'].append(x);add(x)
    else:report['status']='iteration_limit'
    report['seconds']=time.monotonic()-begin
    return report


def run(job):
    name,operator,addend,seconds,output=job
    rows=[]
    for bits in (9,12,16,24,32):
        row=solve(name,operator,addend,bits,seconds)
        rows.append(row)
        (output/f'{name}_{operator}_{addend}_{bits}.json').write_text(json.dumps(row,indent=2)+'\n')
        if row['status']!='equivalent':break
    print(json.dumps({k:rows[-1][k] for k in ('target','operator','addend','bits','status')}),flush=True)
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('results/bitwise_outer_917'))
    parser.add_argument('--seconds',type=float,default=15)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    jobs=[(name,operator,addend,args.seconds,args.output)
          for name in ('absorb_h2','absorb_h6','absorb_both')
          for operator in ('and','or') for addend in ('constant','x','A')]
    with ProcessPoolExecutor(max_workers=3) as pool:groups=list(pool.map(run,jobs))
    (args.output/'summary.json').write_text(json.dumps([r for rows in groups for r in rows],indent=2)+'\n')


if __name__=='__main__':
    main()
