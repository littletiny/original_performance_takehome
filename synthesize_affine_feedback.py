"""Try four-operation hash forms with a variable final MADD addend.

The add-x family uses an inverse odd part of p and covers every possible p
at each checked width. The default add-A-free family also leaves p arbitrary.
The optional add-A screen restricts p to the explicitly listed multipliers.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import random
import time

import z3

from synthesize_affine_outer import target


MULTIPLIERS=(-33,-32,-31,-16,-9,-8,-7,-3,-2,-1,1,2,3,7,8,9,16,31,32,33,
             64,127,128,129,255,256,257,296,297,512,1024,4097)


def solve(name,family,bits,parameter,timeout_ms):
    mask=(1<<bits)-1
    width=bits-parameter if family=='add_x' else bits
    a,b,d,e=z3.BitVecs(' '.join(f'{family}_{name}_{bits}_{parameter}_{v}' for v in ('a','b','d','e')),width)
    r=z3.BitVec(f'inverse_{family}_{name}_{bits}_{parameter}',width)
    solver=z3.Solver()
    if family=='add_x':
        solver.add(r&1==1,z3.ULT(a,1<<(width-1)),z3.ULT(b,1<<(width-1)))
    points=set(range(17))
    for bit in range(bits):
        points.update(((1<<bit)+delta)&mask for delta in (-1,0,1,2,3,7))
    rng=random.Random(917916)
    points.update(rng.randrange(mask+1) for _ in range(24))
    def add(value):
        expected=int(target(name,value))&mask
        if family=='add_x':
            delta=(expected-value)&mask
            assert delta% (1<<parameter)==0
            solver.add(((a*value+b)^(d*value+e))==r*(delta>>parameter))
        else:
            aa=a*value+b
            multiplier=r if family=='add_A_free' else parameter
            solver.add(multiplier*(aa^(d*value+e))+aa==expected)
    for value in sorted(points):add(value)
    report=dict(target=name,family=family,bits=bits,parameter=parameter,
                template=('p*((a*x+b) XOR (d*x+e))+x' if family=='add_x' else
                          'p*((a*x+b) XOR (d*x+e))+(a*x+b)'),
                necessary_inputs=sorted(points),counterexamples=[],
                scope='Named four-operation template and necessary bit projection; not global hash optimality.')
    begin=time.monotonic()
    deadline=begin+20
    for iteration in range(100):
        remaining=deadline-time.monotonic()
        if remaining<=0:
            report.update(status='unknown',reason='Per-projection time budget exhausted')
            break
        solver.set(timeout=max(1,min(timeout_ms,int(remaining*1000))))
        status=solver.check()
        report.update(status=str(status),iterations=iteration+1)
        if status!=z3.sat:
            if status==z3.unknown:report['reason']=solver.reason_unknown()
            break
        model=solver.model()
        aa,bb,dd,ee=[model.eval(v,model_completion=True).as_long() for v in (a,b,d,e)]
        if family=='add_x':pp=pow(model.eval(r).as_long(),-1,1<<width)<<parameter
        elif family=='add_A_free':pp=model.eval(r,model_completion=True).as_long()
        else:pp=parameter
        pp&=0xffffffff
        report['candidate']=dict(a=aa,b=bb,d=dd,e=ee,p=pp)
        x=z3.BitVec(f'verify_{family}_{name}_{bits}_{parameter}',bits)
        expression=pp*((aa*x+bb)^(dd*x+ee))+(x if family=='add_x' else aa*x+bb)
        verifier=z3.Solver()
        remaining=deadline-time.monotonic()
        if remaining<=0:
            report.update(status='unknown',reason='No time left for universal verification')
            break
        verifier.set(timeout=max(1,min(timeout_ms,int(remaining*1000))))
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
    else:report.update(status='iteration_limit')
    report['seconds']=time.monotonic()-begin
    return report


def run(job):
    name,family,parameter,output,timeout=job
    rows=[]
    for bits in (9,12,16,24,32):
        report=solve(name,family,bits,parameter,timeout)
        rows.append(report)
        (output/f'{name}_{family}_{parameter}_{bits}.json').write_text(json.dumps(report,indent=2)+'\n')
        if report['status']!='equivalent':break
    print(json.dumps(dict(target=name,family=family,parameter=parameter,bits=rows[-1]['bits'],status=rows[-1]['status'])),flush=True)
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('results/affine_feedback_917'))
    parser.add_argument('--timeout-ms',type=int,default=4000)
    parser.add_argument('--families',nargs='+',choices=('add_x','add_A','add_A_free'),
                        default=('add_x','add_A_free'))
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    jobs=[]
    for name in ('absorb_h2','absorb_h6','absorb_both'):
        value=int(target(name,0))&0xffffffff
        assert value!=0
        max_shift=(value&-value).bit_length()-1
        assert max_shift<9
        # Since x=0 makes the added x zero, v2(p) cannot exceed v2(target(0)).
        # Check that each retained shift also divides target(x)-x at low width.
        if 'add_x' in args.families:
            for shift in range(max_shift+1):
                assert all((int(target(name,x))-x)%(1<<shift)==0 for x in range(512))
                jobs.append((name,'add_x',shift,args.output,args.timeout_ms))
        if 'add_A' in args.families:
            jobs.extend((name,'add_A',p,args.output,args.timeout_ms) for p in MULTIPLIERS)
        if 'add_A_free' in args.families:
            jobs.append((name,'add_A_free',0,args.output,args.timeout_ms))
    with ProcessPoolExecutor(max_workers=4) as pool:
        groups=list(pool.map(run,jobs))
    (args.output/'summary.json').write_text(json.dumps([r for rows in groups for r in rows],indent=2)+'\n')


if __name__=='__main__':
    main()
