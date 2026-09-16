"""Check cross-stage XOR identities and bounded constant-absorption templates.

The rejection proofs concern the stated affine templates only. They are not
lower bounds on all equivalent hash algorithms or on complete kernel cycles.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random

import z3

from optimize import build, frozen

MASK=(1<<32)-1


def check_identity(name,difference):
    solver=z3.Solver()
    solver.set(timeout=30000)
    solver.add(difference)
    result=solver.check()
    assert result==z3.unsat,(name,str(result))
    return dict(status=str(result),claim='Identity holds for all 32-bit inputs')


def reject_affine(name,bits,target):
    # Any 32-bit affine MADD equality must hold after reduction modulo 2**bits.
    # Exhausting this smaller domain avoids treating random samples as proof.
    a,b=z3.BitVecs(name+'_a '+name+'_b',bits)
    solver=z3.Solver()
    solver.set(timeout=30000)
    mask=(1<<bits)-1
    for x in range(1<<bits):
        solver.add(a*x+b==(target(x)&mask))
    result=solver.check()
    assert result==z3.unsat,(name,str(result))
    return dict(status=str(result),projection_bits=bits,exhaustive_inputs=1<<bits,
                rejected_template='a*x+b modulo 2**32, with any constant a and b',
                scope='Only the stated affine replacement; not all possible rewrites')


def reject_biased_affine(name,bits,target):
    # A free input XOR mask models biasing cached tree nodes before their
    # existing mix. UNSAT on necessary sample equations already rules out a
    # universal identity; it does not require exhaustive enumeration.
    a,b,k=z3.BitVecs(name+'_a '+name+'_b '+name+'_k',bits)
    mask=(1<<bits)-1
    points={0,1,2,3,4,7,8,15,16,mask//2,mask//2+1,mask-1,mask}
    rng=random.Random(511)
    points.update(rng.randrange(mask+1) for _ in range(24))
    solver=z3.Solver()
    solver.set(timeout=30000)
    for x in sorted(points):
        solver.add(a*(x^k)+b==(target(x)&mask))
    result=solver.check()
    assert result==z3.unsat,(name,str(result))
    return dict(status=str(result),projection_bits=bits,necessary_inputs=sorted(points),
                rejected_template='a*(x XOR k)+b modulo 2**32, with any constant a, b, and k',
                scope='Necessary equations are inconsistent, including a free input XOR bias; not all possible rewrites')


def audit(source):
    config=json.loads((source/'config.json').read_text())
    graph=build(config)
    c=[stage[1] for stage in frozen.HASH_STAGES]
    lanes=Counter()
    engines=defaultdict(Counter)
    for name,op in zip(graph.names,graph.ops):
        if op[1][0] not in ('^','lookup_xor'):
            continue
        if name.startswith('r') and not name.startswith('restore'):
            label='mix' if '.dispatch.xor' in name else '.'.join(name.split('.')[2:]).split('.lane')[0]
        else:
            label='setup/restoration'
        lanes[label]+=8 if op[0]=='valu' else 1
        engines[label][op[0]]+=1
    materialized=Counter()
    deferred=Counter()
    for value,r,k,stage,bias in graph.checks:
        if stage==5:
            (deferred if bias else materialized)[r]+=1
    x,node=z3.BitVecs('x node',32)
    def transform(value,shift):
        return value ^ z3.LShR(value,shift)
    def affine(value,multiplier,constant):
        return value*multiplier+constant
    identities={
        'h1_madd_preserves_hash':check_identity('h1_madd',(x+c[0]+(x<<12))!=(4097*x+c[0])),
        'h5_madd_preserves_hash':check_identity('h5_madd',(x+c[4]+(x<<3))!=(9*x+c[4])),
        'fused_h3_h4_preserves_hash':check_identity('h3_h4',
            ((33*x+c[2]+c[3])^((33*x+c[2])<<9))!=
            ((33*x+c[2]+c[3])^((33<<9)*x+(c[2]<<9)))),
        'defer_h6_to_next_tree_node':check_identity('defer_h6',
            (transform(x,16)^c[5]^node)!=(transform(x,16)^(node^c[5]))),
        'xor_shift_16_is_an_involution':check_identity('involution16',transform(transform(x,16),16)!=x),
        'xor_shift_19_is_an_involution':check_identity('involution19',transform(transform(x,19),19)!=x),
    }
    for shift,constant in ((16,c[5]),(19,c[1])):
        identities[f'pull_constant_before_shift_{shift}']=check_identity('transport',
            transform(x^transform(z3.BitVecVal(constant,32),shift),shift)!=(transform(x,shift)^constant))
    h1_bias=c[1]^(c[1]>>19)
    h5_bias=c[5]^(c[5]>>16)
    rejections={
        'absorb_h2_xor_in_h1_madd':reject_affine('h1',12,lambda x:((4097*x+c[0])&MASK)^h1_bias),
        'absorb_h6_xor_in_h5_madd':reject_affine('h5',9,lambda x:((9*x+c[4])&MASK)^h5_bias),
        'push_h2_xor_into_fused_h4_madds':reject_affine('h4',9,lambda x:33*(x^c[1])+c[2]+c[3]),
        'absorb_h2_with_free_tree_xor_bias':reject_biased_affine('biased_h1',12,lambda x:((4097*x+c[0])&MASK)^h1_bias),
        'absorb_h6_with_free_input_xor_bias':reject_biased_affine('biased_h5',9,lambda x:((9*x+c[4])&MASK)^h5_bias),
    }
    # An explicit counterexample to carrying a fixed XOR mask unchanged through
    # the first MADD, even allowing a different constant output mask.
    at_zero=((4097*c[5]+c[0])&MASK)^c[0]
    lhs=affine(x^c[5],4097,c[0])
    rhs=affine(x,4097,c[0])^at_zero
    solver=z3.Solver()
    solver.set(timeout=30000)
    solver.add(lhs!=rhs)
    status=solver.check()
    assert status==z3.sat,status
    model=solver.model()
    witness=dict(status=str(status),input=f'0x{model.eval(x).as_long():08x}',
                 left=f'0x{model.eval(lhs).as_long():08x}',right=f'0x{model.eval(rhs).as_long():08x}',
                 output_mask_fixed_by_x_zero=f'0x{at_zero:08x}')
    return dict(source=str(source),config_sha256=hashlib.sha256((source/'config.json').read_bytes()).hexdigest(),
                xor_lane_operations=dict(sorted(lanes.items())),
                xor_engine_slots={k:dict(v) for k,v in sorted(engines.items())},
                h6_deferred_groups=dict(sorted(deferred.items())),
                h6_materialized_groups=dict(sorted(materialized.items())),
                eliminated_h6_vector_equivalents=sum(deferred.values()),
                transformed_constants=dict(h1_bias=hex(h1_bias),h5_bias=hex(h5_bias)),
                proved_identities=identities,rejected_affine_templates=rejections,
                madd_xor_transport_counterexample=witness,
                scope='Current graph and named 32-bit algebraic templates. No global optimality or cycle-bound claim.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('results/compact_918'))
    parser.add_argument('--output',type=Path,default=Path('results/xor_918/audit.json'))
    args=parser.parse_args()
    report=audit(args.source)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
