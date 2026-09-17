"""Bounded CEGIS for XORs of two affine 32-bit expressions.

Smaller-width UNSAT results reject the corresponding full-width template.
Only a universally verified 32-bit model is a hash rewrite candidate.
"""
import argparse
import json
from pathlib import Path
import random
import time

import z3

from optimize import frozen

CONSTANTS=[stage[1] for stage in frozen.HASH_STAGES]


def target(name,x):
    c=CONSTANTS
    if name=='absorb_h2':
        x=x^c[1]
    value=(33*x+c[2]+c[3])^((33<<9)*x+(c[2]<<9))
    return value if name=='absorb_h2' else 9*value+c[4]


def synthesize(name,bits,timeout_ms=20000,iterations=100):
    mask=(1<<bits)-1
    a,b,d,e=z3.BitVecs(f'{name}_{bits}_a {name}_{bits}_b {name}_{bits}_d {name}_{bits}_e',bits)
    solver=z3.Solver();solver.set(timeout=timeout_ms)
    # Exactly one multiplier is odd. Swapping terms, complementing both
    # results, and flipping both sign bits preserve the XOR; choose a
    # representative of those symmetries without losing solutions.
    solver.add(a&1==1,d&1==0,z3.ULT(a,1<<(bits-1)),z3.ULT(b,1<<(bits-1)))
    def add(value):
        x=z3.BitVecVal(value,bits)
        solver.add(((a*x+b)^(d*x+e))==target(name,x))
    points=set([0,1,2,3,4,7,8,15,16,mask//2,mask//2+1,mask-1,mask])
    rng=random.Random(20260917)
    points.update(rng.randrange(mask+1) for _ in range(48))
    for value in sorted(points):add(value)
    start=time.monotonic()
    report=dict(template='(a*x+b) XOR (d*x+e)',target=name,bits=bits,
                necessary_inputs=sorted(points),counterexamples=[])
    for iteration in range(iterations):
        status=solver.check()
        report.update(iterations=iteration+1,status=str(status),seconds=time.monotonic()-start)
        if status!=z3.sat:
            if status==z3.unknown:report['reason']=solver.reason_unknown()
            return report
        model=solver.model()
        aa,bb,dd,ee=[model.eval(v).as_long() for v in (a,b,d,e)]
        x=z3.BitVec(f'verify_{name}_{bits}',bits)
        verifier=z3.Solver();verifier.set(timeout=timeout_ms)
        verifier.add(((aa*x+bb)^(dd*x+ee))!=target(name,x))
        checked=verifier.check()
        report['candidate']=[aa,bb,dd,ee]
        if checked==z3.unsat:
            report.update(status='equivalent',universal_bits=bits)
            return report
        if checked==z3.unknown:
            report.update(status='unknown',reason=verifier.reason_unknown())
            return report
        value=verifier.model().eval(x).as_long()
        assert value not in points
        points.add(value);report['counterexamples'].append(value);add(value)
    report.update(status='iteration_limit',seconds=time.monotonic()-start)
    return report


def synthesize_chain(family,bits,timeout_ms=20000,iterations=100):
    """Try two MADDs separated by XOR, using an inverse outer multiplier.

    The target's parity depends on x, so the outer multiplier must be odd.
    Writing its inverse as d makes each necessary equation linear in the
    unknown coefficients before bit-vector XOR, avoiding symbolic products.
    """
    mask=(1<<bits)-1
    a,b,c,d,e=z3.BitVecs(f'chain_{family}_{bits}_a chain_{family}_{bits}_b '
                        f'chain_{family}_{bits}_c chain_{family}_{bits}_d chain_{family}_{bits}_e',bits)
    solver=z3.Solver();solver.set(timeout=timeout_ms)
    solver.add(a&1==(0 if family=='xor_x' else 1),d&1==1)
    def add(value):
        fx=z3.simplify(target('fuse_h4_h5',z3.BitVecVal(value,bits))).as_long()
        selector=value if family=='xor_x' else c
        solver.add(((a*value+b)^selector)==d*fx+e)
    points={0,1,2,3,7,8,15,16,mask//2,mask//2+1,mask-1,mask}
    rng=random.Random(821)
    points.update(rng.randrange(mask+1) for _ in range(40))
    for value in sorted(points):add(value)
    report=dict(template='p*((a*x+b) XOR selector)+q',selector='x' if family=='xor_x' else 'constant c',
                parameterization='(a*x+b) XOR selector = d*target(x)+e; d odd, p=inverse(d), q=-p*e',
                target='fuse_h4_h5',bits=bits,necessary_inputs=sorted(points),counterexamples=[])
    start=time.monotonic()
    for iteration in range(iterations):
        status=solver.check()
        report.update(iterations=iteration+1,status=str(status),seconds=time.monotonic()-start)
        if status!=z3.sat:
            if status==z3.unknown:report['reason']=solver.reason_unknown()
            return report
        model=solver.model()
        aa,bb,cc,dd,ee=[model.eval(v,model_completion=True).as_long() for v in (a,b,c,d,e)]
        pp=pow(dd,-1,1<<bits);qq=(-pp*ee)&mask
        x=z3.BitVec(f'verify_chain_{family}_{bits}',bits)
        expression=pp*((aa*x+bb)^(x if family=='xor_x' else cc))+qq
        verifier=z3.Solver();verifier.set(timeout=timeout_ms)
        verifier.add(expression!=target('fuse_h4_h5',x))
        checked=verifier.check()
        report['candidate']=dict(a=aa,b=bb,c=cc,p=pp,q=qq)
        if checked==z3.unsat:
            report.update(status='equivalent',universal_bits=bits)
            return report
        if checked==z3.unknown:
            report.update(status='unknown',reason=verifier.reason_unknown())
            return report
        value=verifier.model().eval(x).as_long()
        assert value not in points
        points.add(value);report['counterexamples'].append(value);add(value)
    report.update(status='iteration_limit',seconds=time.monotonic()-start)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('results/xor_pair_synthesis'))
    parser.add_argument('--timeout-ms',type=int,default=20000)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    results=[]
    for name in ('absorb_h2','fuse_h4_h5'):
        for bits in (9,12,16,32):
            report=synthesize(name,bits,args.timeout_ms)
            results.append(report)
            (args.output/f'{name}_{bits}.json').write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps(report),flush=True)
            if report['status'] in ('unsat','unknown','iteration_limit'):
                break
    for family in ('xor_x','xor_constant'):
        for bits in (9,12,16,32):
            report=synthesize_chain(family,bits,args.timeout_ms)
            results.append(report)
            (args.output/f'fuse_chain_{family}_{bits}.json').write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps(report),flush=True)
            if report['status'] in ('unsat','unknown','iteration_limit'):
                break
    (args.output/'summary.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':
    main()
