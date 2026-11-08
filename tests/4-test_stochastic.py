"""Independent marginal branch oracle and exact enumeration of sampler RNG tickets."""
from collections import defaultdict
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import copy,json,random,subprocess,sys,tempfile,unittest
from unittest.mock import patch
from speclab.finite import FiniteModel,jsonify
from speclab.stochastic import correction,round_kernel,enumerate_decode,audit,sample,naive_law,METRICS
from speclab.stochastic_demo import demo,model


def reference_round(p,q,prefix,gamma):
    """Marginalize unused future proposals, unlike the full-path implementation."""
    g=min(gamma,p.horizon-len(prefix)-1);out=defaultdict(F);v=p.vocabulary
    def walk(accepted,mass):
        at=prefix+accepted;pp,qq=p.at(at),q.at(at)
        if len(accepted)==g:
            for i,t in enumerate(v):out[(accepted+(t,),g,False)]+=mass*pp[i]
            return
        reject=sum(max(F(0),b-a) for a,b in zip(pp,qq))
        if reject:
            for i,t in enumerate(v):
                out[(accepted+(t,),len(accepted),True)]+=mass*max(F(0),pp[i]-qq[i])
        for i,t in enumerate(v):
            if min(pp[i],qq[i]):walk(accepted+(t,),mass*min(pp[i],qq[i]))
    walk((),F(1));return {k:v for k,v in out.items() if v}


def maps(law):return {tuple(x['tokens']):x['probability'] for x in law}


class StochasticTests(unittest.TestCase):
    def test_all_225_ternary_quarter_grid_pairs(self):
        ds=[list(map(lambda x:F(x,4),(i,j,4-i-j))) for i in range(5) for j in range(5-i)]
        count=0
        for p,q in product(ds,repeat=2):
            c=correction(p,q);self.assertEqual(c['acceptance_mass'],1-c['total_variation'])
            self.assertEqual([x['corrected_output_mass'] for x in c['cells']],p)
            self.assertEqual(sum(x['accepted_mass'] for x in c['cells']),c['acceptance_mass'])
            if c['residual']:self.assertEqual(sum(c['residual']),1)
            else:self.assertEqual(p,q)
            for x in c['cells']:
                if not x['draft']:self.assertIsNone(x['accept_probability'])
                else:self.assertEqual(x['accepted_mass'],x['draft']*x['accept_probability'])
            count+=1
        self.assertEqual(count,225)

    def test_known_wrong_fallback_bias(self):
        c=correction(['1/2','1/2'],['3/4','1/4'])
        self.assertEqual(c['residual'],[F(0),F(1)])
        self.assertEqual(c['acceptance_mass'],F(3,4))
        self.assertEqual([x['wrong_target_fallback_mass'] for x in c['cells']],[F(5,8),F(3,8)])

    def check_kernel(self,p,q,prefix,gamma):
        r=round_kernel(p,q,prefix,gamma);actual=defaultdict(F)
        for b in r['branches']:
            key=(tuple(b['emitted']),b['accepted_count'],b['reject_index'] is not None);actual[key]+=b['probability']
            joint=b['draft_path_probability']*b['accepted_prefix_probability']*b['replacement_probability']
            if b['reject_index'] is not None:joint*=b['rejection_probability']
            self.assertEqual(joint,b['probability'])
            self.assertEqual(b['target_positions'],r['gamma_used']+1);self.assertEqual(b['draft_calls'],r['gamma_used'])
            self.assertEqual(len(b['emitted']),b['accepted_count']+1)
        self.assertEqual(dict(actual),reference_round(p,q,tuple(prefix),gamma))
        self.assertEqual(sum(actual.values()),1)

    def test_2187_binary_prefix_table_pairs(self):
        prefixes=[x for n in range(3) for x in product(('A','B'),repeat=n)];count=0
        for xs in product((F(0),F(1,2),F(1)),repeat=7):
            p=FiniteModel(['A','B'],3,{key:[x,1-x] for key,x in zip(prefixes,xs)})
            ys=[(F(1,2) if x==0 else F(1) if x==F(1,2) else F(0)) for x in xs]
            q=FiniteModel(['A','B'],3,{key:[y,1-y] for key,y in zip(prefixes,ys)})
            self.check_kernel(p,q,(),2)
            if count%27==0:
                for gamma in (1,2,3):
                    exact=enumerate_decode(p,q,gamma)
                    self.assertEqual(maps(exact['sequence_law']),maps(naive_law(p)))
            count+=1
        self.assertEqual(count,2187)

    def test_36_ternary_and_unary_prefix_controls(self):
        count=0
        for size in (1,2,3):
            v=list('ABC'[:size])
            for horizon in range(1,5):
                for gamma in (1,2,3):
                    def rule(prefix,shift):
                        w=[(i+len(prefix)+sum(v.index(t) for t in prefix)+shift)%4 for i in range(size)]
                        if not sum(w):w[0]=1
                        return [F(x,sum(w)) for x in w]
                    p=model(v,horizon,lambda x:rule(x,1));q=model(v,horizon,lambda x:rule(x,2))
                    self.check_kernel(p,q,(),gamma)
                    result=enumerate_decode(p,q,gamma)
                    self.assertEqual(maps(result['sequence_law']),maps(naive_law(p)))
                    for k in result['kernels']:self.check_kernel(p,q,tuple(k['prefix']),gamma)
                    count+=1
        self.assertEqual(count,36)

    def test_joint_cost_law_and_identities(self):
        r=demo()
        for name in ('identical','prefix_dependent','disjoint_support'):
            report=r[name]['corrected'];seq=defaultdict(F);cost=defaultdict(F)
            for row in report['joint_sequence_cost_law']:
                seq[tuple(row['tokens'])]+=row['probability'];c=tuple(row['counts'][k] for k in METRICS);cost[c]+=row['probability']
                self.assertEqual(row['counts']['target_positions'],row['counts']['verification_rounds']+row['counts']['draft_calls'])
                self.assertEqual(row['counts']['accepted_draft_tokens']+row['counts']['rejection_rounds']+row['counts']['bonus_tokens'],4)
                self.assertGreaterEqual(row['counts']['target_positions'],4)
            self.assertEqual(dict(seq),maps(report['sequence_law']))
            self.assertEqual(sum(cost.values()),1)
            for i,key in enumerate(METRICS):self.assertEqual(sum(c[i]*p for c,p in cost.items()),report['expected_counts'][key])

    def test_identical_and_disjoint_counts(self):
        r=demo();same=r['identical']['corrected']['expected_counts'];opposite=r['disjoint_support']['corrected']['expected_counts']
        self.assertEqual([same[k] for k in METRICS],[1,4,3,3,0,1])
        self.assertEqual([opposite[k] for k in METRICS],[4,10,6,0,3,1])
        self.assertEqual(r['disjoint_support']['corrected']['sequence_law'],[{'tokens':['A']*4,'probability':F(1)}])
        self.assertGreater(r['prefix_dependent']['wrong_total_variation'],0)

    def test_actual_sampler_exact_rng_tree_all_25_binary_pairs(self):
        class Branch(Exception):
            def __init__(self,n):self.n=n
        class Tickets:
            def __init__(self,path):self.path=iter(path)
            def randrange(self,n):
                try:x=next(self.path)
                except StopIteration:raise Branch(n)
                assert 0<=x<n;return x
        for x,y in product([F(i,4) for i in range(5)],repeat=2):
            p=model(['A','B'],2,lambda prefix:[x,1-x]);q=model(['A','B'],2,lambda prefix:[y,1-y]);observed=defaultdict(F);counts=defaultdict(F)
            def run(path,prob):
                with patch('speclab.stochastic.random.Random',return_value=Tickets(path)):
                    try:r=sample(p,q,1,0)
                    except Branch as b:
                        branch=b.n
                    else:
                        observed[tuple(r['tokens'])]+=prob;counts[tuple(r['counts'][k] for k in METRICS)]+=prob;return
                for t in range(branch):run(path+[t],prob/F(branch))
            run([],F(1));ref=enumerate_decode(p,q,1)
            self.assertEqual(dict(observed),maps(ref['sequence_law']))
            self.assertEqual(dict(counts),{tuple(x['counts'][k] for k in METRICS):x['probability'] for x in ref['cost_law']})

    def test_sampler_trace_counters_and_support(self):
        p=model(['A','B'],4,lambda prefix:[F(3,4),F(1,4)]);q=model(['A','B'],4,lambda prefix:[F(1,4),F(3,4)])
        for seed in range(100):
            r=sample(p,q,3,seed);prefix=[]
            for row in r['rounds']:
                self.assertEqual(row['prefix'],prefix);prefix+=row['emitted'];self.assertGreater(row['replacement_distribution'][p.vocabulary.index(row['replacement_token'])],0)
                self.assertEqual(row['emitted'][:-1],row['draft_path'][:row['accepted_count']])
                if row['reject_index'] is not None:self.assertFalse(row['checks'][-1]['accepted'])
            self.assertEqual(prefix,r['tokens']);self.assertEqual(len(prefix),4)
            self.assertEqual(r['counts']['target_positions'],sum(x['target_positions'] for x in r['rounds']))

    def test_same_seed_is_not_same_output_contract(self):
        p=model(['A','B'],4,lambda prefix:[F(1,2)]*2)
        # Identity draft samples three proposals, three accept tickets, then bonus;
        # naive target draws four token tickets. Exact laws agree, RNG streams don't.
        rng=random.Random(1);from speclab.finite import draw
        naive=[p.vocabulary[draw(p.at(()),rng)] for _ in range(4)]
        self.assertNotEqual(naive,sample(p,p,3,1)["tokens"])

    def test_reproducibility_and_global_rng_unchanged(self):
        p=model(['A','B'],3,lambda prefix:[F(1,3),F(2,3)]);q=model(['A','B'],3,lambda prefix:[F(1,2)]*2)
        state=random.getstate();self.assertEqual(sample(p,q,2,19),sample(p,q,2,19));self.assertEqual(state,random.getstate())

    def test_input_and_model_tables_unchanged(self):
        p=model(['A','B'],3,lambda prefix:[F(1,3),F(2,3)]);q=model(['A','B'],3,lambda prefix:[F(1,2)]*2)
        before=copy.deepcopy((p.to_dict(),q.to_dict()));audit(p,q,2);sample(p,q,2,1)
        self.assertEqual(before,(p.to_dict(),q.to_dict()))

    def test_complete_prefixes_required_even_unreachable(self):
        with self.assertRaises(ValueError):FiniteModel(['A','B'],2,{():[1,0],('A',):[1,0]})
        p=model(['A','B'],2,lambda prefix:[1,0]);self.assertEqual(FiniteModel.from_dict(p.to_dict()).to_dict(),p.to_dict())

    def test_invalid_probabilities(self):
        for bad in ([True,0],[0.5,0.5],[1,1],[-1,2],['1/0',1],['1e-2',1],[1],[],['9'*161,1],[2**513,0]):
            with self.assertRaises(ValueError):correction([1,0],bad)

    def test_invalid_models_and_options(self):
        for vocab,horizon in (([],1),(['A','A'],1),(['A B'],1),(['A','B','C','D'],1),(['A'],True),(['A'],0),(['A'],5)):
            with self.assertRaises(ValueError):FiniteModel(vocab,horizon,{():[1]})
        p=model(['A','B'],2,lambda prefix:[F(1,2)]*2)
        for g in (True,0,4,1.0):
            with self.assertRaises(ValueError):enumerate_decode(p,p,g)
        with self.assertRaises(ValueError):round_kernel(p,p,['A','A'],1)
        with self.assertRaises(ValueError):round_kernel(p,p,[],1,fallback='other')
        for seed in (True,1.5,2**129):
            with self.assertRaises(ValueError):sample(p,p,1,seed)
        q=model(['B','A'],2,lambda prefix:[F(1,2)]*2)
        with self.assertRaises(ValueError):audit(p,q,1)

    def test_invalid_model_json(self):
        p=model(['A','B'],2,lambda prefix:[F(1,2)]*2).to_dict()
        for mutation in ('duplicate','missing','extra','badprefix'):
            d=copy.deepcopy(p)
            if mutation=='duplicate':d['table'].append(d['table'][0])
            elif mutation=='missing':d['table'].pop()
            elif mutation=='extra':d['extra']=1
            else:d['table'][0]['prefix']=[0]
            with self.assertRaises(ValueError):FiniteModel.from_dict(d)

    def test_cli_modes_and_saved_demo(self):
        root=Path(__file__).resolve().parents[1]
        p=subprocess.run([sys.executable,'-m','speclab.stochastic','demo'],capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr);self.assertEqual(p.stdout,(root/'examples/stochastic-demo.json').read_text())
        data=json.loads((root/'examples/stochastic-input.json').read_text());a,b=FiniteModel.from_dict(data['target']),FiniteModel.from_dict(data['draft'])
        for mode in ('audit','sample'):
            p=subprocess.run([sys.executable,'-m','speclab.stochastic',mode,str(root/'examples/stochastic-input.json')],capture_output=True,text=True)
            self.assertEqual(p.returncode,0,p.stderr)
            self.assertEqual(json.loads(p.stdout),jsonify(audit(a,b,data['gamma']) if mode=='audit' else sample(a,b,data['gamma'])))

    def test_cli_strict_errors(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'input.json'
            for text in ('{"a":1,"a":2}','{"a":NaN}','{"a":1}','[]'):
                path.write_text(text)
                for mode in ('audit','sample'):
                    p=subprocess.run([sys.executable,'-m','speclab.stochastic',mode,str(path)],capture_output=True,text=True)
                    self.assertEqual(p.returncode,2);self.assertEqual(p.stdout,'')
            path.write_text(' '*2_000_001)
            p=subprocess.run([sys.executable,'-m','speclab.stochastic','audit',str(path)],capture_output=True,text=True)
            self.assertEqual(p.returncode,2);self.assertEqual(p.stdout,'')
