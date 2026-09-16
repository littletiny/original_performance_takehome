"""Export a verified logical schedule and dense cases into the single file."""
import argparse
import base64
import json
from pathlib import Path
import pickle
import textwrap
import zlib

import numpy as np

from optimize import ROOT, build, allocate, lower, verify_frozen, pt

BEGIN = '# SUB900_CANDIDATE_BEGIN\n'
END = '# SUB900_CANDIDATE_END\n'
DECODER = '''
def _build_tuned_standard():
    """Expand offline schedules; tree and input values remain runtime data."""
    import base64, pickle, zlib
    length, main, cases = pickle.loads(zlib.decompress(
        base64.b85decode(_TUNED_STANDARD)))
    program = [{} for _ in range(length)]
    for position, bundle in main:
        program[position] = bundle
    for base, count, stride, template, patches in cases:
        for choice in range(count):
            bundle = {engine: list(slots) for engine, slots in template.items()}
            for engine, position, operation, power, radix in patches:
                index = (choice // power) % radix
                code = operation[0]
                if code == "lookup_xor":
                    slot = ("^", operation[1], operation[2], operation[4+index])
                elif code == "lookup_copy":
                    value = operation[3+index]
                    slot = ("|", operation[1], value, value)
                elif code == "lookup_store":
                    slot = ("store", operation[1], operation[3+index])
                else:
                    assert code == "lookup_load"
                    slot = ("load", operation[1], operation[3+index])
                bundle[engine][position] = slot
            program[base+choice*stride] = bundle
    assert len(program) <= 12_000
    return program

'''


def export(source, write=False):
    graph = build(json.loads((source/'config.json').read_text()))
    times = np.load(source/'best.npz')['times']
    bases, allocation = allocate(graph, times)
    assert bases is not None, allocation
    program, origins, logical = lower(graph, times, bases)
    assert len(program) <= 12_000, f"Static bundle limit exceeded: {len(program)}"
    # Share an early free FLOW slot before any stores. Follow logical cycles
    # across the optional bootstrap jump without shifting absolute PCs.
    for index in sorted(range(len(program)), key=lambda i:int(origins[i])):
        if origins[index] < 0:
            continue
        bundle = program[index]
        assert not bundle.get('store'), 'No safe initial pause before memory writes'
        if bundle.get('flow'):
            if graph.config.get('pc_address_pools') and index == 0:
                assert bundle['flow'][0][0] == 'jump'
                continue
            assert all(slot[0] in ('add_imm', 'vselect', 'select') for slot in bundle['flow'])
            continue
        bundle['flow'] = [('pause',)]
        pause_cycle = int(origins[index])
        break
    else:
        raise AssertionError('No initial pause slot')
    verification = verify_frozen(graph, times, bases, program, origins, range(10))
    cycles = len(logical)
    main_size = len(program) - graph.total_table_words
    cases = []
    case_positions = set()
    for region in graph.regions:
        n, width = region['n'], region['width']
        stride = region.get('span',1)
        count = n ** width
        for part, (lookups, jump) in enumerate(region['parts']):
            for phase in range(stride):
                cycle = int(times[jump])-stride+1+phase
                table_start = 14 if graph.config.get('pc_address_pools') else main_size
                base = table_start + region['table'] + part*count*stride + phase
                patches = []
                for op_id, stream in lookups:
                    if int(times[op_id]) != cycle:
                        continue
                    engine = graph.ops[op_id][0]
                    operation = tuple(bases[x.vid]+x.off if isinstance(x, pt.V) else x for x in graph.ops[op_id][1])
                    position = logical[cycle][engine].index(operation)
                    patches.append((engine, position, operation, n**(width-1-stream), n))
                cases.append((base, count, stride, program[base], patches))
                case_positions.update(base+choice*stride for choice in range(count))
    main = [(i,bundle) for i,bundle in enumerate(program) if bundle and i not in case_positions]
    payload = (len(program), main, cases)
    blob = base64.b85encode(zlib.compress(pickle.dumps(payload, protocol=4), 9)).decode()
    scope = dict(_TUNED_STANDARD=blob)
    exec(DECODER, scope)
    expanded = scope['_build_tuned_standard']()
    assert expanded == program, 'Standalone expansion differs from verified program'
    report = dict(**verification, **allocation, compressed_payload_bytes=len(blob), source=str(source), pause_cycle=pause_cycle)
    (source/'verification.json').write_text(json.dumps(report, indent=2)+'\n')
    if write:
        block = BEGIN + f'# Verified checkpoint: {cycles} dynamic cycles; {len(program)} static bundles.\n'
        block += '_TUNED_STANDARD = (\n' + ''.join(f'    {line!r}\n' for line in textwrap.wrap(blob, 100)) + ')\n'
        block += DECODER + END + '\n\n'
        path = ROOT/'perf_takehome.py'
        text = path.read_text()
        if BEGIN in text:
            start, rest = text.split(BEGIN, 1)
            _, rest = rest.split(END, 1)
            text = start + block + rest.lstrip('\n')
        else:
            text = text.replace('class V(int):\n', block + 'class V(int):\n', 1)
        marker = '        assert batch_size % VLEN == 0\n'
        hook = '''        if ((forest_height, n_nodes, batch_size, rounds) == (10, 2047, 256, 16)
                and CFG.get("USE_EMBEDDED", True)):
            self.instrs = _build_tuned_standard()
            self._merge_pause()
            return
'''
        if 'self.instrs = _build_tuned_standard()' not in text:
            text = text.replace(marker, hook + marker, 1)
        path.write_text(text)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    export(args.source, args.write)
