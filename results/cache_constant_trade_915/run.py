"""Rebuild and verify the retained cache/constant/chain graph configurations."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from optimize import build,verify_semantics
from resource_bounds import analyze_graph


def main():
    rows=[]
    for source in sorted(Path(__file__).parent.glob('*/config.json')):
        graph=build(json.loads(source.read_text()))
        bound=analyze_graph(graph)
        saved=json.loads((source.parent/'bounds.json').read_text())
        assert bound==saved,source.parent.name
        semantics=verify_semantics(graph,(0,1))
        rows.append(dict(configuration=source.parent.name,bound=bound['bound'],
                         weighted_alu_valu=bound['counts']['weighted_alu_valu'],
                         graph_sha256=bound['graph_sha256'],semantics=semantics))
    (Path(__file__).parent/'rebuild.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
