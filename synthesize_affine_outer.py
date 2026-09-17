"""Try absorbing hash XOR constants into a four-operation affine/XOR form.

This keeps an outer MADD, unlike the previous three-operation searches.
All operations in the projected target are prefix-compatible; UNSAT rejects
only the stated full-width family, not every possible hash rewrite.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import random
import time

import z3

from synthesize_xor_pairs import CONSTANTS, target as previous_target


def target(name,x):
    if name in ('absorb_h2','absorb_both'):
        value=9*previous_target('absorb_h2',x)+CONSTANTS[4]
    else:
        value=previous_target('fuse_h4_h5',x)
    if name in ('absorb_h6','absorb_both'):
        value=value^(CONSTANTS[5]^(CONSTANTS[5]>>16))
    return value


def synthesize(name,bits,seconds=90):
    mask=(1<<bits)-1
    a,b,d,e,r,s=z3.BitVecs(' '.join(f'{name}_{bits}_{v}' for v in ('a','b','d','e','r','s')),bits)
    solver=z3.Solver()
    # Odd outer coefficient follows from varying output parity. Term exchange
    # and simultaneous sign-bit flips select equivalent representatives.
    solver.add(r&1==1,a&1==1,d&1==0,z3.ULT(a,1<<(bits-1)),z3.ULT(b,1<<(bits-1)))
    points=set(range(17))
    for bit in range(bits):
        points.update(((1<<bit)+delta)&mask for delta in (-1,0,1,2,3,7))
    rng=random.Random(918917)
    points.update(rng.randrange(mask+1) for _ in range(40))
    def add(value):
        fx=int(target(name,value))&mask
        solver.add(((a*value+b)^(d*value+e))==r*fx+s)
    for value in sorted(points):add(value)
    report=dict(target=name,bits=bits,template='p*((a*x+b) XOR (d*x+e))+q',
                parameterization='inner = r*target+s; r odd, p=inverse(r), q=-p*s',
                necessary_inputs=sorted(points),counterexamples=[],
                scope='Named four-operation template and bit projection; no global optimality claim.')
    begin=time.monotonic();deadline=begin+seconds
    for iteration in range(100):
        remaining=deadline-time.monotonic()
        if remaining<=0:
            report.update(status='unknown',reason='Per-projection time budget exhausted')
            break
        solver.set(timeout=max(1,int(min(30,remaining)*1000)))
        status=solver.check()
        report.update(iterations=iteration+1,status=str(status))
        if status!=z3.sat:
            if status==z3.unknown:report['reason']=solver.reason_unknown()
            break
        model=solver.model()
        aa,bb,dd,ee,rr,ss=[model.eval(v,model_completion=True).as_long() for v in (a,b,d,e,r,s)]
        pp=pow(rr,-1,1<<bits);qq=(-pp*ss)&mask
        report['candidate']=dict(a=aa,b=bb,d=dd,e=ee,p=pp,q=qq)
        x=z3.BitVec(f'verify_{name}_{bits}',bits)
        expression=pp*((aa*x+bb)^(dd*x+ee))+qq
        verifier=z3.Solver()
        remaining=deadline-time.monotonic()
        if remaining<=0:
            report.update(status='unknown',reason='No time left for universal verification')
            break
        verifier.set(timeout=max(1,int(min(30,remaining)*1000)))
        verifier.add(expression!=target(name,x))
        checked=verifier.check()
        if checked==z3.unsat:
            report.update(status='equivalent',universal_bits=bits)
            break
        if checked==z3.unknown:
            report.update(status='unknown',reason=verifier.reason_unknown())
            break
        value=verifier.model().eval(x).as_long()
        assert value not in points
        points.add(value);report['counterexamples'].append(value);add(value)
    else:
        report.update(status='iteration_limit')
    report['seconds']=time.monotonic()-begin
    return report


def run(job):
    name,output,seconds=job
    rows=[]
    for bits in (9,12,16,24,32):
        report=synthesize(name,bits,seconds)
        rows.append(report)
        (output/f'{name}_{bits}.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k not in ('necessary_inputs','counterexamples')}),flush=True)
        if report['status']!='equivalent':break
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('results/affine_outer_917'))
    parser.add_argument('--seconds',type=float,default=90)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(max_workers=3) as pool:
        groups=list(pool.map(run,[(name,args.output,args.seconds) for name in ('absorb_h2','absorb_h6','absorb_both')]))
    (args.output/'summary.json').write_text(json.dumps([r for rows in groups for r in rows],indent=2)+'\n')


if __name__=='__main__':
    main()
