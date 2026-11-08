"""Finite-table oracle lookahead planning, never a measured latency predictor."""
from collections import defaultdict
from fractions import Fraction as F
from functools import lru_cache
from itertools import product
import argparse,json,sys
from .finite import FiniteModel,distribution,jsonify,rational
from .stochastic import METRICS,_pair,round_kernel,naive_law

PRICE_KEYS=('verification_rounds','target_positions','draft_calls')

def prices(values):
    if not isinstance(values,dict) or set(values)!=set(PRICE_KEYS):raise ValueError('prices require rounds, target positions and draft calls')
    out={k:rational(values[k]) for k in PRICE_KEYS}
    if any(x<0 for x in out.values()):raise ValueError('prices must be nonnegative')
    return out


def action_set(remaining,max_gamma):
    if isinstance(max_gamma,bool) or not isinstance(max_gamma,int) or not 1<=max_gamma<=3:raise ValueError('max gamma must be 1..3')
    # Canonicalize boundary-equivalent positive lookaheads. Last slot has only 0.
    return tuple(range(min(max_gamma,remaining-1)+1))


def kernel(target,draft,prefix,action):
    _pair(target,draft,1);target.at(prefix)
    if isinstance(action,bool) or not isinstance(action,int) or not 0<=action<=3:raise ValueError('action must be 0..3')
    if action:
        return round_kernel(target,draft,prefix,action)
    branches=[]
    for tok,p in zip(target.vocabulary,target.at(prefix)):
        if p:branches.append({'draft_path':[],'draft_path_probability':F(1),'accepted_count':0,
            'accepted_prefix_probability':F(1),'reject_index':None,'rejection_probability':F(0),
            'replacement_distribution':target.at(prefix),'replacement_token':tok,'replacement_probability':p,
            'emitted':[tok],'probability':p,'target_positions':1,'draft_calls':0,'verification_rounds':1,'bonus_tokens':1})
    return {'prefix':list(prefix),'gamma_requested':0,'gamma_used':0,'branches':branches}


def counts(b):
    return (1,b['target_positions'],b['draft_calls'],b['accepted_count'],int(b['reject_index'] is not None),b['bonus_tokens'])


def charge(c,p):return sum(F(c[METRICS.index(k)])*p[k] for k in PRICE_KEYS)


def prefixes(target):
    return [x for n in range(target.horizon) for x in product(target.vocabulary,repeat=n)]


def validate_policy(target,policy,max_gamma):
    if not isinstance(policy,dict) or set(policy)!=set(prefixes(target)):raise ValueError('policy must cover every nonterminal prefix exactly')
    for prefix,a in policy.items():
        if isinstance(a,bool) or not isinstance(a,int) or a not in action_set(target.horizon-len(prefix),max_gamma):raise ValueError('noncanonical or disallowed action')
    return dict(policy)


def evaluate(target,draft,policy,assigned_prices,*,max_gamma=3):
    """Full exact sequence x count law under a deterministic prefix policy."""
    _pair(target,draft,1);p=prices(assigned_prices);action_set(target.horizon,max_gamma)
    policy=validate_policy(target,policy,max_gamma);kernels={}
    @lru_cache(None)
    def visit(prefix):
        if len(prefix)==target.horizon:return {(prefix,(0,)*len(METRICS)):F(1)}
        k=kernel(target,draft,prefix,policy[prefix]);kernels[prefix]=k;out=defaultdict(F)
        for b in k['branches']:
            c=counts(b)
            for (seq,tail),mass in visit(prefix+tuple(b['emitted'])).items():
                out[(seq,tuple(x+y for x,y in zip(c,tail)))]+=b['probability']*mass
        return dict(out)
    joint=visit(());seqs=defaultdict(F);costs=defaultdict(F)
    for (seq,c),mass in joint.items():seqs[seq]+=mass;costs[c]+=mass
    expected={k:sum(F(c[i])*mass for c,mass in costs.items()) for i,k in enumerate(METRICS)}
    law=[{'tokens':list(seq),'probability':mass} for seq,mass in sorted(seqs.items())]
    assert law==naive_law(target)
    assert sum(seqs.values())==sum(costs.values())==1
    assert expected['target_positions']==expected['verification_rounds']+expected['draft_calls']
    assert expected['accepted_draft_tokens']+expected['rejection_rounds']+expected['bonus_tokens']==target.horizon
    value=sum(charge(c,p)*mass for c,mass in costs.items())
    return {'sequence_law':law,'joint_sequence_cost_law':[{'tokens':list(seq),'counts':dict(zip(METRICS,c)),'probability':mass} for (seq,c),mass in sorted(joint.items())],
        'cost_law':[{'counts':dict(zip(METRICS,c)),'probability':mass,'assigned_cost':charge(c,p)} for c,mass in sorted(costs.items())],
        'expected_counts':expected,'expected_assigned_cost':value,'prices':p,
        'policy':[{'prefix':list(x),'action':policy[x]} for x in prefixes(target)],
        'reachable_kernels':[kernels[x] for x in sorted(kernels,key=lambda x:(len(x),x))]}


def optimize(target,draft,assigned_prices,*,max_gamma=3):
    """Bellman optimum over prefix-conditioned choices, including no draft.

    Knows every target/draft conditional. Planning work is outside path counters.
    Ties choose the smallest effective draft length, including no drafting.
    """
    _pair(target,draft,1);p=prices(assigned_prices);action_set(target.horizon,max_gamma)
    values={};chosen={};certificates=[]
    for prefix in reversed(prefixes(target)):
        choices=[]
        for a in action_set(target.horizon-len(prefix),max_gamma):
            k=kernel(target,draft,prefix,a);immediate=F(0);future=F(0);transitions=defaultdict(F)
            for b in k['branches']:
                next_prefix=prefix+tuple(b['emitted']);mass=b['probability'];now=charge(counts(b),p)
                immediate+=mass*now;future+=mass*values.get(next_prefix,F(0));transitions[next_prefix]+=mass
            choices.append({'action':a,'immediate_expected_cost':immediate,'future_expected_cost':future,'q_value':immediate+future,
                'transitions':[{'next_prefix':list(x),'probability':mass,'optimal_tail_value':values.get(x,F(0))} for x,mass in sorted(transitions.items())]})
        best=min(choices,key=lambda c:(c['q_value'],c['action']));values[prefix]=best['q_value'];chosen[prefix]=best['action']
        for c in choices:c['bellman_slack']=c['q_value']-best['q_value']
        certificates.append({'prefix':list(prefix),'value':values[prefix],'selected_action':chosen[prefix],'choices':choices})
    result=evaluate(target,draft,chosen,p,max_gamma=max_gamma);assert result['expected_assigned_cost']==values[()]
    fixed=[]
    for a in range(max_gamma+1):
        policy={x:min(a,target.horizon-len(x)-1) for x in prefixes(target)}
        r=evaluate(target,draft,policy,p,max_gamma=max_gamma)
        fixed.append({'requested_gamma':a,'expected_assigned_cost':r['expected_assigned_cost'],'expected_counts':r['expected_counts']})
    fixed_best=min(fixed,key=lambda x:(x['expected_assigned_cost'],x['requested_gamma']))
    assert values[()]<=fixed_best['expected_assigned_cost']
    return {'scope':'oracle finite-table prefix policy; assigned costs, not measured latency',
        'max_gamma':max_gamma,'target':target.to_dict(),'draft':draft.to_dict(),'optimal':result,
        'bellman_certificates':sorted(certificates,key=lambda x:(len(x['prefix']),x['prefix'])),
        'fixed_baselines':fixed,'best_fixed_gamma':fixed_best['requested_gamma'],
        'best_fixed_assigned_cost':fixed_best['expected_assigned_cost'],
        'gain_over_best_fixed':fixed_best['expected_assigned_cost']-values[()],
        'corrected_total_variation':F(0)}


def policy_from_records(target,records,*,max_gamma=3):
    if not isinstance(records,list):raise ValueError('policy must be records')
    policy={}
    for rec in records:
        if not isinstance(rec,dict) or set(rec)!={'prefix','action'} or not isinstance(rec['prefix'],list) or any(not isinstance(x,str) for x in rec['prefix']):raise ValueError('wrong policy record')
        key=tuple(rec['prefix'])
        if key in policy:raise ValueError('duplicate policy prefix')
        policy[key]=rec['action']
    return validate_policy(target,policy,max_gamma)


def load(path):
    def pairs(xs):
        d={}
        for k,v in xs:
            if k in d:raise ValueError('duplicate JSON key')
            d[k]=v
        return d
    from pathlib import Path
    raw=Path(path).read_bytes()
    if len(raw)>2_000_000:raise ValueError('input exceeds 2 MB')
    d=json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('nonfinite JSON')))
    if not isinstance(d,dict) or set(d)!={'target','draft','prices','max_gamma'}:raise ValueError('wrong planner keys')
    return FiniteModel.from_dict(d['target']),FiniteModel.from_dict(d['draft']),prices(d['prices']),d['max_gamma']


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('mode',choices=('demo','optimize'));ap.add_argument('input',nargs='?');args=ap.parse_args(argv)
    try:
        if args.mode=='demo':
            if args.input:raise ValueError('demo takes no input')
            from .policy_demo import demo
            out=demo()
        else:
            if not args.input:raise ValueError('optimize requires input JSON')
            t,d,p,g=load(args.input);out=optimize(t,d,p,max_gamma=g)
        print(json.dumps(jsonify(out),indent=2,sort_keys=True))
    except (ValueError,TypeError,OSError,UnicodeError) as e:print('error: '+str(e),file=sys.stderr);return 2
    return 0

if __name__=='__main__':raise SystemExit(main())
