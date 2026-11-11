import copy,json,random,subprocess,sys,tempfile,unittest,hashlib
from fractions import Fraction as F
from itertools import product
from pathlib import Path
from speclab.finite import FiniteModel,jsonify,rational
from speclab.stopping import audit,proposals,decode,round_kernel,MODES,METRICS
from speclab.stopverify import verify
from speclab.stopping_demo import fixtures,decode_input
from speclab.stochastic_demo import model

ROOT=Path(__file__).resolve().parents[1]

class StoppingTests(unittest.TestCase):
    def test_all_public_reports_independent_and_regenerated(self):
        saved=json.loads((ROOT/'examples/stopping-cases.json').read_text());manifest=json.loads((ROOT/'examples/stopping-manifest.json').read_text())
        for f,old in zip(fixtures(),saved):
            self.assertEqual(f['id'],old['id']);args=decode_input(f['input']);r=audit(*args)
            self.assertEqual(hashlib.sha256(json.dumps(jsonify(r),sort_keys=True,separators=(',',':')).encode()).hexdigest(),manifest[f['id']]);self.assertTrue(verify(*args,r)['verified'])
        self.assertEqual(len(saved),len(fixtures()))

    def test_censor_bias_hand_derived(self):
        r=audit(*decode_input(fixtures()[0]['input']))
        a=r['modes']['after_censor'];self.assertEqual(a['total_variation'],F(1,8))
        first={t:F(0)for t in ['A','B']}
        for x in a['sequence_law']:first[x['tokens'][0]]+=x['probability']
        self.assertEqual(first,{'A':F(5,8),'B':F(3,8)})
        for m in MODES:
            if m!='after_censor':self.assertEqual(r['modes'][m]['total_variation'],0)

    def test_equal_models_still_censor_bias(self):
        r=audit(*decode_input(fixtures()[3]['input']))
        self.assertEqual(r['target'],r['draft']);self.assertGreater(r['modes']['after_censor']['total_variation'],0)

    def test_keep_stop_changes_work(self):
        r=audit(*decode_input(fixtures()[1]['input']))
        self.assertLess(r['modes']['after_keep']['expected_counts']['draft_calls'],r['modes']['fixed']['expected_counts']['draft_calls'])
        self.assertEqual(r['modes']['after_keep']['sequence_law'],r['naive_target_law'])

    def test_full_binary_grid_horizon_two(self):
        for a,b,t in product(range(5),repeat=3):
            p=model(['A','B'],2,lambda x:[F(a,4),F(4-a,4)]);q=model(['A','B'],2,lambda x:[F(b,4),F(4-b,4)])
            r=audit(p,q,1,F(t,4));verify(p,q,1,F(t,4),r)

    def test_seeded_complete_prefix_models(self):
        rng=random.Random(20261111)
        for i in range(32):
            v=['a','B','AA']if i%4==0 else['B','A'];h=1+i%4
            def rule(prefix):
                xs=[rng.randrange(5)for t in v]
                if not sum(xs):xs[0]=1
                return [F(x,sum(xs))for x in xs]
            p=model(v,h,rule);q=model(v,h,rule);g=1+i%3;t=F(i%5,4)
            r=audit(p,q,g,t);verify(p,q,g,t,r)

    def test_all_count_conservation_and_mass(self):
        for f in fixtures():
            args=decode_input(f['input']);r=audit(*args);h=args[0].horizon
            for mode in MODES:
                out=r['modes'][mode]
                for name in ['sequence_law','count_law','joint_sequence_count_law']:self.assertEqual(sum(x['probability']for x in out[name]),1)
                for rec in out['count_law']:
                    c=rec['counts'];self.assertEqual(c['target_positions'],c['verification_rounds']+c['retained_proposals']);self.assertEqual(c['sampled_proposals'],c['retained_proposals']+c['censored_proposals']);self.assertEqual(c['accepted_proposals']+c['rejection_rounds']+c['bonus_tokens'],h)
                for k in out['kernels']:
                    self.assertEqual(sum(x['probability']for x in k['proposal_leaves']),1);self.assertEqual(sum(x['probability']for x in k['branches']),1)

    def test_before_check_call_is_charged_and_final_slot_reserved(self):
        p,q,g,t=decode_input(fixtures()[7]['input']);r=audit(p,q,g,t)
        e=r['modes']['before_max']['expected_counts'];self.assertEqual(e['draft_calls'],p.horizon-1);self.assertEqual(e['sampled_proposals'],0);self.assertEqual(e['target_positions'],p.horizon)
        one=audit(*decode_input(fixtures()[8]['input']))
        for m in MODES:self.assertEqual(one['modes'][m]['expected_counts']['draft_calls'],0)

    def test_strict_threshold_boundaries_and_zero_support(self):
        for index in [4,5,6,7,8,11]:
            r=audit(*decode_input(fixtures()[index]['input']));self.assertEqual(r['modes']['after_censor']['total_variation'],0)
        r=audit(*decode_input(fixtures()[5]['input']));self.assertEqual(r['modes']['fixed']['expected_counts'],r['modes']['after_keep']['expected_counts'])

    def test_fixed_baselines_full_unweakened(self):
        p,q,g,t=decode_input(fixtures()[2]['input']);r=audit(p,q,g,t)
        for k in [1,2,3]:self.assertEqual(r['fixed_baselines'][str(k)],decode(p,q,k,t,'fixed'))

    def test_every_report_scalar_tamper_rejected(self):
        args=decode_input(fixtures()[0]['input']);r=jsonify(audit(*args));paths=[]
        def walk(v,p=()):
            if isinstance(v,dict):
                for k,x in v.items():walk(x,p+(k,))
            elif isinstance(v,list):
                for i,x in enumerate(v):walk(x,p+(i,))
            else:paths.append(p)
        walk(r)
        for path in paths:
            modified=copy.deepcopy(r);at=modified
            for key in path[:-1]:at=at[key]
            at[path[-1]]=None if at[path[-1]]is not None else 'tamper'
            with self.assertRaises(ValueError,msg=str(path)):verify(*args,modified)
        self.assertGreater(len(paths),1000)

    def test_structural_and_type_tamper(self):
        args=decode_input(fixtures()[0]['input']);r=jsonify(audit(*args))
        variants=[];x=copy.deepcopy(r);x['max_gamma']=True;variants.append(x);x=copy.deepcopy(r);x['extra']=0;variants.append(x);x=copy.deepcopy(r);del x['draft'];variants.append(x);x=copy.deepcopy(r);x['modes']['fixed']['kernels'].reverse();variants.append(x)
        for x in variants:
            with self.assertRaises(ValueError):verify(*args,x)

    def test_invalid_contract(self):
        p,q,g,t=decode_input(fixtures()[0]['input'])
        for gamma in [0,4,True,'1',1.0]:
            with self.assertRaises(ValueError):audit(p,q,gamma,t)
        for threshold in [-1,'4/3',True,.5,'NaN','1/0','1e2','1'*161]:
            with self.assertRaises(ValueError):audit(p,q,g,threshold)
        with self.assertRaises(ValueError):audit(p,model(['A'],2,lambda x:[1]),g,t)
        with self.assertRaises(ValueError):round_kernel(p,q,('A','A'),g,t,'fixed')
        with self.assertRaises(ValueError):proposals(p,q,(),g,t,'unknown')

    def test_cli_real_report_verify_and_parser(self):
        with tempfile.TemporaryDirectory()as temp:
            p=Path(temp)/'input.json';p.write_text(json.dumps(fixtures()[0]['input']));r=Path(temp)/'report.json'
            cmd=[sys.executable,'-m','speclab.stopping_cli'];run=subprocess.run(cmd+['audit',str(p)],capture_output=True,text=True);self.assertEqual(run.returncode,0,run.stderr);r.write_text(run.stdout)
            run=subprocess.run(cmd+['verify',str(p),str(r)],capture_output=True,text=True);self.assertEqual(run.returncode,0,run.stderr);self.assertTrue(json.loads(run.stdout)['verified'])
            for bad in ['{"x":0,"\\u0078":1}','{"x":0.1}','NaN','['*49+']'*49]:
                p.write_text(bad);run=subprocess.run(cmd+['audit',str(p)],capture_output=True,text=True);self.assertNotEqual(run.returncode,0);self.assertEqual(run.stdout,'')
