import base64,copy,importlib.util,json,unittest,zlib
from pathlib import Path
from xml.etree import ElementTree as ET
from speclab.policy_demo import demo
from speclab.finite import jsonify
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('render_policy',ROOT/'examples/render_policy.py');r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)

class PolicyFigureTests(unittest.TestCase):
    def test_reproducible_svg_and_complete_metadata(self):
        report=demo();s=r.render(report);self.assertEqual(s,(ROOT/'assets/lookahead-policy.svg').read_text())
        meta=ET.fromstring(s).find('{http://www.w3.org/2000/svg}metadata');self.assertEqual(json.loads(zlib.decompress(base64.b64decode(meta.text))),jsonify(report))

    def test_every_data_cell(self):
        v=r.values(jsonify(demo()));seen={}
        for node in ET.fromstring(r.render(demo())).iter():
            if node.attrib.get('id','').startswith('cell-'):
                key=node.attrib['id'][5:];self.assertNotIn(key,seen);seen[key]=node.text;self.assertEqual(node.text,str(v[key]));self.assertEqual(node.attrib['data-value'],str(v[key]))
        self.assertEqual(set(v),set(seen));self.assertEqual(len(v),115)

    def test_invisible_transition_mass_changes_full_metadata(self):
        report=demo();bad=copy.deepcopy(report);bad['prefix_switch']['bellman_certificates'][0]['choices'][0]['transitions'][0]['probability']+=1
        self.assertEqual(r.values(jsonify(report)),r.values(jsonify(bad)));self.assertNotEqual(r.render(report),r.render(bad))
