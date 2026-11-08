from collections import defaultdict
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import copy,json,random,subprocess,sys,tempfile,unittest
from speclab.finite import FiniteModel,jsonify
from speclab.stochastic import METRICS,enumerate_decode,naive_law
from speclab.stochastic_demo import model
from speclab.policy import optimize,evaluate,kernel,prices,action_set,prefixes,policy_from_records
from speclab.policy_demo import fixtures,demo


def oracle_joint(p,q,policy):
    """Independent marginal recursion: never call the full-draft-path kernel."""
    v=p.vocabulary
    def suffix(prefix):
        if len(prefix)==p.horizon:return {(prefix,(0,)*6):F(1)}
        g=policy[prefix];rounds=[]
        def proposal(accepted,mass):
            at=prefix+accepted;pp,qq=p.at(at),q.at(at)
            if len(accepted)==g:
                for i,t in enumerate(v):
                    if pp[i]:rounds.append((accepted+(t,),mass*pp[i],len(accepted),0,1))
                return
            for i,t in enumerate(v):
                excess=max(F(0),pp[i]-qq[i])
                if excess:rounds.append((accepted+(t,),mass*excess,len(accepted),1,0))
            for i,t in enumerate(v):
                m=min(pp[i],qq[i])
                if m:proposal(accepted+(t,),mass*m)
        proposal((),F(1));out=defaultdict(F)
        for emitted,mass,accepted,reject,bonus in rounds:
            c=(1,g+1,g,accepted,reject,bonus)
            for (seq,tail),rest in suffix(prefix+emitted).items():out[(seq,tuple(x+y for x,y in zip(c,tail)))]+=mass*rest
        return dict(out)
    return suffix(())


def expected(joint,p):return sum(mass*sum(F(c[METRICS.index(k)])*F(value) for k,value in p.items()) for (seq,c),mass in joint.items())


def policies(p,maximum):
    xs=prefixes(p)
    for acts in product(*(action_set(p.horizon-len(x),maximum) for x in xs)):yield dict(zip(xs,acts))


class PolicyTests(unittest.TestCase):
    def test_exhaustive_all_12_policies_25_quarter_grid_pairs_three_prices(self):
        count=0
        for a,b in product(range(5),repeat=2):
            p=model(['A','B'],3,lambda x:[F(a,4),1-F(a,4)])
            q=model(['A','B'],3,lambda x:[F(b,4),1-F(b,4)])
            all_policies=list(policies(p,3));self.assertEqual(len(all_policies),12)
            laws=[oracle_joint(p,q,pi) for pi in all_policies]
            for price in ({'verification_rounds':1,'target_positions':0,'draft_calls':'1/10'},
                          {'verification_rounds':0,'target_positions':1,'draft_calls':'1/3'},
                          {'verification_rounds':'1/7','target_positions':'2/5','draft_calls':'1/11'}):
                r=optimize(p,q,price);want=min(expected(law,price) for law in laws)
                self.assertEqual(r['optimal']['expected_assigned_cost'],want)
                pi=policy_from_records(p,r['optimal']['policy']);ref=oracle_joint(p,q,pi)
                got={(tuple(x['tokens']),tuple(x['counts'][k] for k in METRICS)):x['probability'] for x in r['optimal']['joint_sequence_cost_law']}
                self.assertEqual(got,ref);count+=1
        self.assertEqual(count,75)

    def test_exhaustive_prefix_dependent_policies_all_joint_laws(self):
        count=0
        for shift in range(6):
            def rule(x,s):
                a=(sum(ord(t) for t in x)+len(x)*s+s)%5;return [F(a,4),1-F(a,4)]
            p=model(['A','B'],3,lambda x:rule(x,shift));q=model(['A','B'],3,lambda x:rule(x,shift+1))
            price={'verification_rounds':1,'target_positions':'1/5','draft_calls':'1/3'};best=None
            for pi in policies(p,3):
                ref=oracle_joint(p,q,pi);r=evaluate(p,q,pi,price)
                got={(tuple(x['tokens']),tuple(x['counts'][k] for k in METRICS)):x['probability'] for x in r['joint_sequence_cost_law']}
                self.assertEqual(got,ref);value=expected(ref,price);self.assertEqual(r['expected_assigned_cost'],value)
                best=value if best is None else min(best,value);count+=1
            self.assertEqual(optimize(p,q,price)['optimal']['expected_assigned_cost'],best)
        self.assertEqual(count,72)

    def test_adaptation_strictly_beats_every_fixed_full_strength_baseline(self):
        r=demo();a=r['prefix_switch'];self.assertEqual(a['optimal']['expected_assigned_cost'],F(23,10))
        self.assertEqual(a['best_fixed_assigned_cost'],F(47,20));self.assertEqual(a['gain_over_best_fixed'],F(1,20))
        self.assertEqual([x['expected_assigned_cost'] for x in a['fixed_baselines']],[F(4),F(27,10),F(11,4),F(47,20)])
        pi={tuple(x['prefix']):x['action'] for x in a['optimal']['policy']};self.assertEqual(pi[()],3);self.assertEqual(pi[('B','B')],0);self.assertEqual(pi[('A','A')],1)

    def test_bellman_every_action_transition_and_slack(self):
        p,q=fixtures();cost={'verification_rounds':1,'target_positions':0,'draft_calls':'1/10'};r=optimize(p,q,cost)
        cert={tuple(x['prefix']):x for x in r['bellman_certificates']}
        self.assertEqual(set(cert),set(prefixes(p)))
        for prefix,c in cert.items():
            self.assertEqual([a['action'] for a in c['choices']],list(action_set(p.horizon-len(prefix),3)))
            self.assertEqual(c['value'],min(a['q_value'] for a in c['choices']))
            for a in c['choices']:
                self.assertEqual(sum(x['probability'] for x in a['transitions']),1)
                self.assertEqual(a['q_value'],a['immediate_expected_cost']+a['future_expected_cost'])
                self.assertEqual(a['bellman_slack'],a['q_value']-c['value']);self.assertGreaterEqual(a['bellman_slack'],0)
                future=F(0)
                for x in a['transitions']:
                    next_prefix=tuple(x['next_prefix']);tail=cert[next_prefix]['value'] if next_prefix in cert else F(0)
                    self.assertEqual(x['optimal_tail_value'],tail);future+=x['probability']*tail
                self.assertEqual(a['future_expected_cost'],future)

    def test_fixed_policy_matches_existing_exact_stochastic_engine(self):
        p,q=fixtures();price={'verification_rounds':1,'target_positions':0,'draft_calls':'1/10'}
        for gamma in (1,2,3):
            pi={x:min(gamma,p.horizon-len(x)-1) for x in prefixes(p)};a=evaluate(p,q,pi,price);b=enumerate_decode(p,q,gamma)
            for k in ('sequence_law','joint_sequence_cost_law','expected_counts'):self.assertEqual(a[k],b[k])

    def test_36_sizes_horizons_and_gamma_limits(self):
        n=0
        for size,horizon,maximum in product((1,2,3),range(1,5),(1,2,3)):
            v=list('ABC'[:size])
            def rule(x,s):
                w=[(i+len(x)+s)%4+1 for i in range(size)];return [F(a,sum(w)) for a in w]
            p=model(v,horizon,lambda x:rule(x,1));q=model(v,horizon,lambda x:rule(x,2))
            r=optimize(p,q,{'verification_rounds':1,'target_positions':0,'draft_calls':'1/10'},max_gamma=maximum)
            self.assertEqual(r['optimal']['sequence_law'],naive_law(p));self.assertLessEqual(r['optimal']['expected_assigned_cost'],r['best_fixed_assigned_cost'])
            self.assertEqual(len(r['fixed_baselines']),maximum+1)
            for row in r['optimal']['joint_sequence_cost_law']:
                c=row['counts'];self.assertEqual(c['target_positions'],c['verification_rounds']+c['draft_calls']);self.assertEqual(c['accepted_draft_tokens']+c['rejection_rounds']+c['bonus_tokens'],horizon)
            n+=1
        self.assertEqual(n,36)

    def test_score_work_no_draft_and_zero_cost_tie(self):
        p,q=fixtures()
        for price in ({'verification_rounds':0,'target_positions':1,'draft_calls':'1/10'}, {'verification_rounds':0,'target_positions':0,'draft_calls':0}):
            r=optimize(p,q,price);self.assertTrue(all(x['selected_action']==0 for x in r['bellman_certificates']))
            self.assertEqual(r['gain_over_best_fixed'],0)
        same=optimize(p,p,{'verification_rounds':1,'target_positions':0,'draft_calls':'1/10'})
        self.assertEqual(same['optimal']['expected_assigned_cost'],F(13,10));self.assertEqual(same['best_fixed_gamma'],3)

    def test_input_and_rng_unchanged(self):
        p,q=fixtures();price={'verification_rounds':1,'target_positions':0,'draft_calls':'1/10'};before=copy.deepcopy((p.to_dict(),q.to_dict(),price));rng=random.getstate()
        optimize(p,q,price);self.assertEqual((p.to_dict(),q.to_dict(),price),before);self.assertEqual(random.getstate(),rng)

    def test_invalid_prices_limits_and_policy(self):
        p,q=fixtures();price={'verification_rounds':1,'target_positions':0,'draft_calls':'1/10'};pi={x:0 for x in prefixes(p)}
        for bad in ({},dict(price,extra=1),dict(price,draft_calls=-1),dict(price,draft_calls=0.1),dict(price,draft_calls=True),dict(price,draft_calls='1/0')):
            with self.assertRaises(ValueError):optimize(p,q,bad)
        for maximum in (True,0,4,'2',None):
            with self.assertRaises(ValueError):optimize(p,q,price,max_gamma=maximum)
        for bad in ({},dict(pi,extra=1),{**pi,():True},{**pi,():4},{**pi,('A','A','A'):1}):
            with self.assertRaises(ValueError):evaluate(p,q,bad,price)
        with self.assertRaises(ValueError):policy_from_records(p,[{'prefix':[],'action':0},{'prefix':[],'action':0}])
        with self.assertRaises(ValueError):policy_from_records(p,[{'prefix':[[]],'action':0}])
        for a in (True,-1,4,'0'):
            with self.assertRaises(ValueError):kernel(p,q,(),a)
        other=model(['X'],4,lambda x:[1])
        with self.assertRaises(ValueError):optimize(p,other,price)

    def test_cli_demo_optimize_and_strict_errors(self):
        p,q=fixtures();data={'target':p.to_dict(),'draft':q.to_dict(),'prices':{'verification_rounds':1,'target_positions':0,'draft_calls':'1/10'},'max_gamma':3}
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'input.json';path.write_text(json.dumps(jsonify(data)))
            def cli(*args):return subprocess.run([sys.executable,'-m','speclab.policy',*args],text=True,capture_output=True)
            r=cli('optimize',str(path));self.assertEqual(r.returncode,0);self.assertEqual(json.loads(r.stdout),jsonify(optimize(p,q,data['prices'])))
            self.assertEqual(json.loads(cli('demo').stdout),jsonify(demo()))
            for text in ('{}','{"target":1,"target":2}','NaN',json.dumps(jsonify(dict(data,max_gamma=0))),' '*2_000_001):
                path.write_text(text);r=cli('optimize',str(path));self.assertEqual(r.returncode,2);self.assertEqual(r.stdout,'')
            for args in (('optimize',),('demo',str(path))):
                r=cli(*args);self.assertEqual(r.returncode,2);self.assertEqual(r.stdout,'')

if __name__=='__main__':unittest.main()
