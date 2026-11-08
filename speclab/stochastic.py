"""Exact stochastic speculative decoding with full finite-law/cost certificates."""
from collections import defaultdict
from fractions import Fraction as F
from functools import lru_cache
from itertools import product
import random
from .finite import FiniteModel,distribution,draw,jsonify


def correction(target,draft):
    """Per-token maximal-coupling decomposition; null residual if never rejected."""
    if not isinstance(target,(list,tuple)) or not 1<=len(target)<=3:raise ValueError('distribution size must be 1..3')
    p,q=distribution(target,len(target)),distribution(draft,len(target))
    accepted=[min(a,b) for a,b in zip(p,q)];alpha=sum(accepted)
    residual=[max(F(0),a-b) for a,b in zip(p,q)];reject=1-alpha
    r=[v/reject for v in residual] if reject else None
    cells=[{'token_index':i,'target':a,'draft':b,'accept_probability':min(F(1),a/b) if b else None,
            'accepted_mass':accepted[i],'residual_mass':residual[i],
            'residual_probability':r[i] if r else None,
            'corrected_output_mass':accepted[i]+residual[i],
            'wrong_target_fallback_mass':accepted[i]+reject*a} for i,(a,b) in enumerate(zip(p,q))]
    assert all(c['corrected_output_mass']==c['target'] for c in cells)
    return {'target':p,'draft':q,'acceptance_mass':alpha,'rejection_mass':reject,'residual':r,
            'total_variation':sum(abs(a-b) for a,b in zip(p,q))/2,'cells':cells}


def _pair(target,draft,gamma):
    if not isinstance(target,FiniteModel) or not isinstance(draft,FiniteModel):raise ValueError('FiniteModel pair required')
    if target.vocabulary!=draft.vocabulary or target.horizon!=draft.horizon:raise ValueError('model vocabulary/order/horizon must match')
    if isinstance(gamma,bool) or not isinstance(gamma,int) or not 1<=gamma<=3:raise ValueError('gamma must be 1..3')


def round_kernel(target,draft,prefix,gamma,*,fallback='residual'):
    """All full proposal paths, first-rejection branches and bonus branches.

    Near the finite horizon, draft g=min(gamma,remaining-1), reserving one
    correction/bonus slot. Target scoring includes every drafted position plus
    the bonus prefix, even when an earlier rejection makes scoring redundant.
    """
    _pair(target,draft,gamma)
    if fallback not in ('residual','target'):raise ValueError('unknown fallback')
    prefix=tuple(prefix);target.at(prefix);g=min(gamma,target.horizon-len(prefix)-1);v=target.vocabulary
    branches=[]
    for path in product(v,repeat=g):
        mass=F(1);checks=[]
        for j,tok in enumerate(path):
            at=prefix+path[:j];q=draft.at(at);mass*=q[v.index(tok)]
            checks.append(correction(target.at(at),q))
        if not mass:continue
        survive=F(1)
        for j,(tok,c) in enumerate(zip(path,checks)):
            alpha=c['cells'][v.index(tok)]['accept_probability'];rejected=mass*survive*(1-alpha)
            if rejected:
                residual=c['residual'] if fallback=='residual' else c['target'];assert residual is not None
                for d,p in enumerate(residual):
                    if p:branches.append({'draft_path':list(path),'draft_path_probability':mass,
                        'accepted_count':j,'accepted_prefix_probability':survive,'reject_index':j,
                        'rejection_probability':1-alpha,'replacement_distribution':residual,
                        'replacement_token':v[d],'replacement_probability':p,
                        'emitted':list(path[:j]) + [v[d]],'probability':rejected*p,
                        'target_positions':g+1,'draft_calls':g,'verification_rounds':1,'bonus_tokens':0})
            survive*=alpha
        bonus=target.at(prefix+path)
        for d,p in enumerate(bonus):
            if mass*survive*p:branches.append({'draft_path':list(path),'draft_path_probability':mass,
                'accepted_count':g,'accepted_prefix_probability':survive,'reject_index':None,
                'rejection_probability':F(0),'replacement_distribution':bonus,'replacement_token':v[d],
                'replacement_probability':p,'emitted':list(path)+[v[d]],'probability':mass*survive*p,
                'target_positions':g+1,'draft_calls':g,'verification_rounds':1,'bonus_tokens':1})
    assert sum(b['probability'] for b in branches)==1
    return {'prefix':list(prefix),'gamma_used':g,'fallback':fallback,'branches':branches}


METRICS=('verification_rounds','target_positions','draft_calls','accepted_draft_tokens','rejection_rounds','bonus_tokens')


def enumerate_decode(target,draft,gamma,*,fallback='residual'):
    """Exact terminal sequence x joint cost law; caches one kernel per prefix."""
    _pair(target,draft,gamma);kernels={}
    @lru_cache(None)
    def solve(prefix):
        if len(prefix)==target.horizon:return {(prefix,(0,)*len(METRICS)):F(1)}
        kernel=round_kernel(target,draft,prefix,gamma,fallback=fallback);kernels[prefix]=kernel;out=defaultdict(F)
        for b in kernel['branches']:
            cost=(1,b['target_positions'],b['draft_calls'],b['accepted_count'],int(b['reject_index'] is not None),b['bonus_tokens'])
            for (seq,tail),p in solve(prefix+tuple(b['emitted'])).items():
                out[(seq,tuple(x+y for x,y in zip(cost,tail)))]+=b['probability']*p
        return dict(out)
    joint=solve(());sequences=defaultdict(F);costs=defaultdict(F)
    for (seq,c),p in joint.items():sequences[seq]+=p;costs[c]+=p
    expected={key:sum(F(c[i])*p for c,p in costs.items()) for i,key in enumerate(METRICS)}
    assert sum(sequences.values())==sum(costs.values())==1
    assert expected['accepted_draft_tokens']+expected['rejection_rounds']+expected['bonus_tokens']==target.horizon
    assert expected['target_positions']==expected['verification_rounds']+expected['draft_calls']
    return {'fallback':fallback,'gamma':gamma,'horizon':target.horizon,
            'sequence_law':[{'tokens':list(k),'probability':p} for k,p in sorted(sequences.items())],
            'cost_law':[{'counts':dict(zip(METRICS,k)),'probability':p} for k,p in sorted(costs.items())],
            'joint_sequence_cost_law':[{'tokens':list(seq),'counts':dict(zip(METRICS,c)),'probability':p} for (seq,c),p in sorted(joint.items())],
            'expected_counts':expected,'kernels':[kernels[k] for k in sorted(kernels,key=lambda k:(len(k),k))]}


def naive_law(target):
    out=[]
    for seq in product(target.vocabulary,repeat=target.horizon):
        p=F(1)
        for j,tok in enumerate(seq):p*=target.at(seq[:j])[target.vocabulary.index(tok)]
        if p:out.append({'tokens':list(seq),'probability':p})
    return out


def audit(target,draft,gamma):
    correct=enumerate_decode(target,draft,gamma);wrong=enumerate_decode(target,draft,gamma,fallback='target');naive=naive_law(target)
    def as_map(xs):return {tuple(r['tokens']):r['probability'] for r in xs}
    n,c,w=map(as_map,(naive,correct['sequence_law'],wrong['sequence_law']));allkeys=set(n)|set(c)|set(w)
    tv=lambda x:sum(abs(x.get(k,F(0))-n.get(k,F(0))) for k in allkeys)/2
    assert tv(c)==0
    return {'target':target.to_dict(),'draft':draft.to_dict(),'naive_target_law':naive,'corrected':correct,
            'corrected_total_variation':tv(c),'wrong_target_fallback_control':wrong,'wrong_total_variation':tv(w),
            'naive_counts':{'verification_rounds':target.horizon,'target_positions':target.horizon,'draft_calls':0}}


def sample(target,draft,gamma,seed=0):
    """Independent procedural sampler, not a draw from the enumerated kernel."""
    _pair(target,draft,gamma)
    if isinstance(seed,bool) or not isinstance(seed,int) or abs(seed).bit_length()>128:raise ValueError('seed must be an integer of at most 128 bits')
    rng=random.Random(seed);v=target.vocabulary;prefix=();trace=[]
    while len(prefix)<target.horizon:
        g=min(gamma,target.horizon-len(prefix)-1);path=[];qs=[]
        for _ in range(g):
            q=draft.at(prefix+tuple(path));qs.append(q);path.append(v[draw(q,rng)])
        ps=[target.at(prefix+tuple(path[:j])) for j in range(g+1)]
        accepted=0;decisions=[];replacement=None;reject=None
        for j,tok in enumerate(path):
            c=correction(ps[j],qs[j]);prob=c['cells'][v.index(tok)]['accept_probability']
            ok=rng.randrange(prob.denominator)<prob.numerator
            decisions.append({'position':j,'proposed_token':tok,'accept_probability':prob,'accepted':ok,'certificate':c})
            if not ok:replacement=c['residual'];reject=j;break
            accepted+=1
        if replacement is None:replacement=ps[g]
        chosen=v[draw(replacement,rng)];emitted=path[:accepted]+[chosen]
        trace.append({'prefix':list(prefix),'draft_path':path,'checks':decisions,'accepted_count':accepted,
                      'reject_index':reject,'replacement_distribution':replacement,'replacement_token':chosen,
                      'emitted':emitted,'target_positions':g+1,'draft_calls':g,'verification_rounds':1,'bonus_tokens':int(reject is None)})
        prefix+=tuple(emitted)
    counts={key:0 for key in METRICS}
    for r in trace:
        counts['verification_rounds']+=1;counts['target_positions']+=r['target_positions'];counts['draft_calls']+=r['draft_calls']
        counts['accepted_draft_tokens']+=r['accepted_count'];counts['rejection_rounds']+=int(r['reject_index'] is not None);counts['bonus_tokens']+=r['bonus_tokens']
    return {'tokens':list(prefix),'seed':seed,'counts':counts,'rounds':trace}


def main():
    import argparse,json
    from pathlib import Path
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True);sub.add_parser('demo')
    for mode in ('audit','sample'):
        sp=sub.add_parser(mode);sp.add_argument('file')
        if mode=='sample':sp.add_argument('--seed',type=int,default=0)
    ns=p.parse_args()
    def unique(pairs):
        d={}
        for k,v in pairs:
            if k in d:raise ValueError('duplicate JSON key')
            d[k]=v
        return d
    try:
        if ns.mode=='demo':
            from .stochastic_demo import demo
            result=demo()
        else:
            path=Path(ns.file)
            if path.stat().st_size>2_000_000:raise ValueError('input exceeds 2 MB')
            data=json.loads(path.read_text(),object_pairs_hook=unique,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('nonfinite JSON')))
            if not isinstance(data,dict) or set(data)!={'target','draft','gamma'}:raise ValueError('wrong input keys')
            a,b=FiniteModel.from_dict(data['target']),FiniteModel.from_dict(data['draft'])
            result=audit(a,b,data['gamma']) if ns.mode=='audit' else sample(a,b,data['gamma'],ns.seed)
        print(json.dumps(jsonify(result),sort_keys=True,indent=2));return 0
    except (ValueError,OSError,TypeError,RecursionError) as e:p.error(str(e))


if __name__=='__main__':main()
