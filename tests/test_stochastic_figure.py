import base64,copy,importlib.util,json,unittest,zlib
from pathlib import Path
from xml.etree import ElementTree as ET
from speclab.stochastic_demo import demo
from speclab.finite import jsonify
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('render_stochastic',ROOT/'examples/render_stochastic.py');r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)


class FigureTests(unittest.TestCase):
    def test_exact_svg_and_full_report_metadata(self):
        report=demo();text=r.render(report);self.assertEqual(text,(ROOT/'assets/stochastic-audit.svg').read_text())
        meta=ET.fromstring(text).find('{http://www.w3.org/2000/svg}metadata')
        self.assertEqual(json.loads(zlib.decompress(base64.b64decode(meta.text))),jsonify(report))

    def test_every_probability_sequence_and_count_cell(self):
        report=jsonify(demo());want=r.values(report);seen={}
        for node in ET.fromstring(r.render(report)).iter():
            if node.attrib.get('id','').startswith('cell-'):
                key=node.attrib['id'][5:];self.assertNotIn(key,seen);seen[key]=node.text
                self.assertEqual(node.text,str(want[key]));self.assertEqual(node.attrib['data-value'],str(want[key]))
        self.assertEqual(set(seen),set(want));self.assertEqual(len(seen),96)

    def test_hidden_branch_term_changes_metadata(self):
        report=demo();bad=copy.deepcopy(report)
        bad['prefix_dependent']['corrected']['kernels'][0]['branches'][0]['draft_path_probability']+=1
        self.assertEqual(r.values(jsonify(report)),r.values(jsonify(bad)));self.assertNotEqual(r.render(report),r.render(bad))
