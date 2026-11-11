"""Regenerate public controls and independently replay every exact report."""
import json,sys,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from speclab.stopping_demo import demo,decode_input,fixtures
from speclab.stopverify import verify
root=Path(__file__).resolve().parents[1]
saved=demo();manifest=json.loads((root/'examples/stopping-manifest.json').read_text())
assert json.loads((root/'examples/stopping-cases.json').read_text())==fixtures()
assert {f['id']:hashlib.sha256(json.dumps(f['report'],sort_keys=True,separators=(',',':')).encode()).hexdigest()for f in saved}==manifest
for fixture in saved:assert verify(*decode_input(fixture['input']),fixture['report'])['verified']
assert (root/'docs/stopping-data.js').read_text()=='globalThis.STOPPING_FIXTURES='+json.dumps(fixtures(),separators=(',',':'))+';\n'
print(json.dumps({'fixtures':len(saved),'independent_replay':True,'browser_inputs_exact':True}))
