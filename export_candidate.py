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
    main, table_words, cases = pickle.loads(zlib.decompress(
        base64.b85decode(_TUNED_STANDARD)))
    program = main + [{} for _ in range(table_words)]
    for base, count, template, patches in cases:
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
            program[base+choice] = bundle
    assert len(program) < 500_000
    return program

'''


def export(source, write=False):
    graph = build(json.loads((source/'config.json').read_text()))
    times = np.load(source/'best.npz')['times']
    bases, allocation = allocate(graph, times)
    assert bases is not None, allocation
    program, origins, logical = lower(graph, times, bases)
    # Pause shares the first bundle so the native two-yield harness sees its
    # initial memory checkpoint without moving any absolute dispatch PC.
    assert not program[0].get('flow') and not program[0].get('store')
    program[0]['flow'] = [('pause',)]
    verification = verify_frozen(graph, times, bases, program, origins, range(10))
    cycles = len(logical)
    cases = []
    for region in graph.regions:
        n, width = region['n'], region['width']
        count = n ** width
        for part, (lookups, jump) in enumerate(region['parts']):
            cycle = int(times[jump])
            base = cycles + region['table'] + part*count
            patches = []
            for op_id, stream in lookups:
                engine = graph.ops[op_id][0]
                operation = tuple(bases[x.vid]+x.off if isinstance(x, pt.V) else x for x in graph.ops[op_id][1])
                position = logical[cycle][engine].index(operation)
                patches.append((engine, position, operation, n**(width-1-stream), n))
            cases.append((base, count, program[base], patches))
    payload = (program[:cycles], graph.total_table_words, cases)
    blob = base64.b85encode(zlib.compress(pickle.dumps(payload, protocol=4), 9)).decode()
    scope = dict(_TUNED_STANDARD=blob)
    exec(DECODER, scope)
    expanded = scope['_build_tuned_standard']()
    assert expanded == program, 'Standalone expansion differs from verified program'
    report = dict(**verification, **allocation, compressed_payload_bytes=len(blob), source=str(source))
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
        marker = '        blob = EMBEDDED_SCHEDULES.get(\n'
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
