"""Independently check the projected targets against the original hash stages."""
import hashlib
import json
from pathlib import Path

from optimize import MASK, ROOT, frozen
from synthesize_affine_outer import target


def reference(name,x):
    if name in ('absorb_h2','absorb_both'):
        x^=frozen.HASH_STAGES[1][1]
    for op1,c,op2,shift,amount in frozen.HASH_STAGES[2:5]:
        assert op1=='+' and shift=='<<'
        left=(x+c)&MASK
        right=(x<<amount)&MASK
        x=(left+right if op2=='+' else left^right)&MASK
    if name in ('absorb_h6','absorb_both'):
        c=frozen.HASH_STAGES[5][1]
        bias=c^(c>>16)
        assert bias^(bias>>16)==c
        x^=bias
    return x


def main():
    projections=[]
    for name in ('absorb_h2','absorb_h6','absorb_both'):
        for bits in (9,12):
            mask=(1<<bits)-1
            for x in range(mask+1):
                expected=int(target(name,x))&mask
                for upper in (0,0x5a5a0000,0xfffff000):
                    assert reference(name,(upper&~mask)|x)&mask==expected
            projections.append(dict(target=name,bits=bits,low_inputs=mask+1,high_prefixes=3))
    models=[]
    source=ROOT/'results/affine_outer_917'
    for row in json.loads((source/'summary.json').read_text()):
        if row['status']!='equivalent':continue
        bits=row['bits'];mask=(1<<bits)-1;c=row['candidate']
        for x in range(mask+1):
            value=(c['p']*((c['a']*x+c['b'])^(c['d']*x+c['e']))+c['q'])&mask
            assert value==reference(row['target'],x)&mask
        models.append(dict(target=row['target'],bits=bits,inputs=mask+1))
    assert len(models)==2 and all(row['bits']==9 for row in models)
    paths=['synthesize_affine_outer.py','synthesize_affine_feedback.py',
           'synthesize_xor_pairs.py','tests/frozen_problem.py']
    report=dict(projections=projections,projection_models=models,
                source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},
                scope='Checks target projection and low-width models. No 32-bit hash rewrite was found.')
    (source/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(projections=len(projections),projection_models=len(models),
                         original_hash_evaluations=sum(p['low_inputs']*p['high_prefixes'] for p in projections))))


if __name__=='__main__':
    main()
