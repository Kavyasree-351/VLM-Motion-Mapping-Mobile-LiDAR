"""Verify checkpoint content hashes and completed notebook outputs."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
expected=json.loads((ROOT/'outputs/sam_trial/model_provenance.json').read_text())
verified={}
for directory in ['sam3','sam31']:
    entry=expected[directory];path=ROOT/'.models'/directory/entry['checkpoint']
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda:handle.read(8*1024*1024),b''):digest.update(chunk)
    assert digest.hexdigest()==entry['checkpoint_sha256'],f'Checksum mismatch: {path}'
    verified[directory]=dict(sha256=digest.hexdigest(),bytes=path.stat().st_size,verified=True)
notebook=json.loads((ROOT/'notebookgyu.sam.executed.ipynb').read_text(encoding='utf-8'))
cells=[c for c in notebook['cells'] if c['cell_type']=='code']
assert len(cells)==5 and all(c['execution_count'] for c in cells)
assert not [o for c in cells for o in c.get('outputs',[]) if o['output_type']=='error']
verified['notebook']=dict(executed_cells=len(cells),errors=0)
(ROOT/'outputs/sam_trial/artifact_validation.json').write_text(json.dumps(verified,indent=2))
print(json.dumps(verified,indent=2))
