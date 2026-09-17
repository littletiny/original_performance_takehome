"""Shorten existing scalar-constant DAGs without adding instructions."""
from functools import lru_cache


MASK=(1<<32)-1


def apply_expressions(g,known,replacements):
    """Apply independently checked numeric formulas saved by offline searches."""
    named={name:i for i,name in enumerate(g.names)}
    for target,expression in replacements.items():
        value=int(target);op,a,b=expression
        assert a in known and b in known and value not in (a,b)
        actual={'+':lambda:a+b,'-':lambda:a-b,'^':lambda:a^b}[op]()&MASK
        assert actual==value,(value,expression)
        i=named[f'constant.{value}'];old=g.ops[i]
        assert old[0]=='alu' and old[3]==[(known[value],1)]
        aa,bb=known[a],known[b]
        g.ops[i]=[old[0],(op,old[1][1],aa,bb),[(aa,1),(bb,1)],old[3],old[4]]


@lru_cache(maxsize=32)
def expressions(numbers):
    wanted=set(numbers)
    candidates={value:[] for value in numbers}
    for a in numbers:
        for b in numbers:
            for op,value in (('+',(a+b)&MASK),('-',(a-b)&MASK),('^',a^b)):
                if value in wanted and value not in (a,b):
                    if op!='-' and a>b:continue
                    candidates[value].append((op,a,b))
    return candidates


def resynthesize(g,known):
    """Use only declared known values; runtime tree/input values are excluded.

    Existing ALU constant destinations retain their identities. LOAD constants
    and known words with no mutable-constant ancestors form fixed roots. All
    chosen expressions strictly increase depth, so a forward numeric reference
    cannot introduce a cycle. Unreachable destinations keep their old formula.
    """
    writers={int(base)+j:i for i,op in enumerate(g.ops)
             for base,size in op[3] for j in range(size)}
    targets={}
    for i,(name,op) in enumerate(zip(g.names,g.ops)):
        if name.startswith('constant.') and op[0]=='alu' and len(op[3])==1:
            value=int(name.split('.')[1])
            if known.get(value)==op[3][0][0]:targets[value]=i
    mutable=set(targets.values())
    parents=[set(writers[int(base)+j] for base,size in op[2] for j in range(size)
                 if int(base)+j in writers) for op in g.ops]
    for before,after,_ in g.control:parents[after].add(before)
    @lru_cache(None)
    def ancestry(i):
        depth=1;uses_mutable=i in mutable
        for parent in parents[i]:
            before,changed=ancestry(parent)
            depth=max(depth,before+1);uses_mutable|=changed
        return depth,uses_mutable
    zero={int(v)+j for v in g.initial_zero for j in range(g.sizes[v.vid])}
    infinity=10**9
    depth={value:infinity for value in known}
    for value,ref in known.items():
        if value in targets:continue
        if int(ref) in writers:
            ready,depends=ancestry(writers[int(ref)])
            if not depends:depth[value]=ready
        elif int(ref) in zero:
            assert value==0
            depth[value]=0
    options=expressions(tuple(sorted(known)))
    chosen={}
    changed=True
    while changed:
        changed=False
        for value in targets:
            for op,a,b in options[value]:
                candidate=1+max(depth[a],depth[b])
                if candidate<depth[value]:
                    depth[value]=candidate;chosen[value]=(op,a,b);changed=True
    rewrites=[]
    numeric={int(ref):value for value,ref in known.items()}
    for value,i in targets.items():
        if value not in chosen:continue
        op,a,b=chosen[value]
        assert max(depth[a],depth[b])<depth[value]
        old=g.ops[i]
        old_depth=ancestry(i)[0]
        # An unchanged formula must obey the same final depth ordering. Using
        # its old graph depth here could retain a back-edge to a newly chosen
        # dependency and invalidate the acyclicity argument.
        operands=[numeric.get(int(ref)) for ref in old[1][2:]]
        if (all(operand is not None for operand in operands) and
                1+max(depth[operand] for operand in operands)<=depth[value]):
            continue
        aa,bb=known[a],known[b]
        slot=(op,old[1][1],aa,bb)
        g.ops[i]=[old[0],slot,[(aa,1),(bb,1)],old[3],old[4]]
        rewrites.append(dict(name=g.names[i],old_depth=old_depth,new_depth=depth[value],
                             expression=[op,a,b]))
    g.constant_rewrites=rewrites
