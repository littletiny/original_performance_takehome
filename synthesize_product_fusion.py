"""Check nonlinear three-operation rewrites of fused H3/H4 followed by H5.

All templates use prefix-compatible 32-bit operations. UNSAT in a necessary
low-bit projection rejects that full-width template. A smaller-width model
is never treated as a complete hash rewrite.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import random
import time

import z3

from synthesize_xor_pairs import target


TEMPLATES = {
    'input_xor': ('(x XOR k)*(a*x+b)+c', lambda x,a,b,c,k: (x^k)*(a*x+b)+c),
    'input_or': ('(x OR k)*(a*x+b)+c', lambda x,a,b,c,k: (x|k)*(a*x+b)+c),
    'input_and': ('(x AND k)*(a*x+b)+c', lambda x,a,b,c,k: (x&k)*(a*x+b)+c),
    'factor_xor': ('x*((a*x+b) XOR k)+c', lambda x,a,b,c,k: x*((a*x+b)^k)+c),
    'factor_or': ('x*((a*x+b) OR k)+c', lambda x,a,b,c,k: x*((a*x+b)|k)+c),
    'factor_and': ('x*((a*x+b) AND k)+c', lambda x,a,b,c,k: x*((a*x+b)&k)+c),
    'factor_xor_x': ('x*((a*x+b) XOR x)+c', lambda x,a,b,c,k: x*((a*x+b)^x)+c),
    'xor_addend': ('x*(a*x+b)+(x XOR k)', lambda x,a,b,c,k: x*(a*x+b)+(x^k)),
    'affine_xor_const_plus_x': ('c*((a*x+b) XOR k)+x', lambda x,a,b,c,k: c*((a*x+b)^k)+x),
    'affine_xor_x_plus_x': ('c*((a*x+b) XOR x)+x', lambda x,a,b,c,k: c*((a*x+b)^x)+x),
    # This also contains every product of two affine terms plus a constant.
    'quadratic': ('x*(a*x+b)+c', lambda x,a,b,c,k: x*(a*x+b)+c),
}


def synthesize(family, bits, seconds=60):
    formula, expression = TEMPLATES[family]
    parameters = z3.BitVecs(' '.join(f'{family}_{bits}_{v}' for v in ('a','b','c','k')), bits)
    solver = z3.Solver()
    mask = (1<<bits)-1
    points = set(range(17))
    for bit in range(bits):
        points.update(((1<<bit)+delta)&mask for delta in (-1,0,1,2,3,7))
    rng = random.Random(919)
    points.update(rng.randrange(mask+1) for _ in range(32))
    def add(value):
        x = z3.BitVecVal(value,bits)
        solver.add(expression(x,*parameters) == target('fuse_h4_h5',x))
    for value in sorted(points):
        add(value)
    report = dict(family=family, template=formula, bits=bits, target='fuse_h4_h5',
                  necessary_inputs=sorted(points), counterexamples=[],
                  scope='This template and bit width; only a verified 32-bit model is a complete rewrite.')
    begin = time.monotonic()
    deadline = begin+seconds
    for iteration in range(100):
        remaining = deadline-time.monotonic()
        if remaining <= 0:
            report.update(status='unknown', reason='Per-projection time budget exhausted')
            break
        solver.set(timeout=max(1,int(min(20,remaining)*1000)))
        status = solver.check()
        report.update(iterations=iteration+1, status=str(status))
        if status != z3.sat:
            if status == z3.unknown:
                report['reason'] = solver.reason_unknown()
            break
        model = solver.model()
        values = [model.eval(v,model_completion=True).as_long() for v in parameters]
        report['candidate'] = dict(zip(('a','b','c','k'),values))
        x = z3.BitVec(f'verify_{family}_{bits}',bits)
        verifier = z3.Solver()
        remaining = deadline-time.monotonic()
        if remaining <= 0:
            report.update(status='unknown', reason='No time left for universal verification')
            break
        verifier.set(timeout=max(1,int(min(20,remaining)*1000)))
        verifier.add(expression(x,*values) != target('fuse_h4_h5',x))
        checked = verifier.check()
        if checked == z3.unsat:
            report.update(status='equivalent', universal_bits=bits)
            break
        if checked == z3.unknown:
            report.update(status='unknown', reason=verifier.reason_unknown())
            break
        value = verifier.model().eval(x).as_long()
        assert value not in points
        points.add(value)
        report['counterexamples'].append(value)
        add(value)
    else:
        report.update(status='iteration_limit')
    report['seconds'] = time.monotonic()-begin
    return report


def run(job):
    family, output, seconds = job
    rows = []
    for bits in (9,12,16,32):
        report = synthesize(family,bits,seconds)
        (output/f'{family}_{bits}.json').write_text(json.dumps(report,indent=2)+'\n')
        rows.append(report)
        print(json.dumps({k:v for k,v in report.items() if k not in ('necessary_inputs','counterexamples')}),flush=True)
        if report['status'] != 'equivalent':
            break
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('results/product_fusion_918'))
    parser.add_argument('--seconds',type=float,default=60)
    parser.add_argument('--workers',type=int,default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        groups = list(pool.map(run,[(family,args.output,args.seconds) for family in TEMPLATES]))
    (args.output/'summary.json').write_text(json.dumps([r for rows in groups for r in rows],indent=2)+'\n')


if __name__=='__main__':
    main()
