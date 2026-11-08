"""Developed finite probability fixtures, not trained language-model benchmarks."""
from fractions import Fraction as F
from itertools import product
from .finite import FiniteModel
from .stochastic import audit,sample,correction


def model(vocabulary,horizon,rule):
    return FiniteModel(vocabulary,horizon,{p:rule(p) for n in range(horizon) for p in product(vocabulary,repeat=n)})


def demo():
    v=['A','B'];h=4
    target=model(v,h,lambda p: [F(3,4),F(1,4)] if not p or p[-1]=='A' else [F(1,4),F(3,4)])
    draft=model(v,h,lambda p: [F(1,4),F(3,4)] if len(p)%2==0 else [F(1,2),F(1,2)])
    opposite=model(v,h,lambda p:[0,1]);only_a=model(v,h,lambda p:[1,0])
    r={'one_step':correction([F(1,2),F(1,2)],[F(3,4),F(1,4)]),
       'identical':audit(target,target,3),'prefix_dependent':audit(target,draft,2),'disjoint_support':audit(only_a,opposite,3),
       'sample_trace':sample(target,draft,2,19)}
    assert r['one_step']['cells'][0]['wrong_target_fallback_mass']==F(5,8)
    assert all(r[k]['corrected_total_variation']==0 for k in ('identical','prefix_dependent','disjoint_support'))
    assert r['prefix_dependent']['wrong_total_variation']>0
    assert r['identical']['corrected']['expected_counts']['verification_rounds']==1
    assert r['disjoint_support']['corrected']['expected_counts']['target_positions']==10
    r['hypothetical_cost_scenarios']=[]
    for key in ('identical','prefix_dependent','disjoint_support'):
        e=r[key]['corrected']['expected_counts'];c=F(1,10)
        latency=e['verification_rounds']+c*e['draft_calls']
        work=e['target_positions']+c*e['draft_calls']
        r['hypothetical_cost_scenarios'].append({'fixture':key,'draft_unit_cost':c,
            'modeled_parallel_cost':latency,'weighted_position_score_cost':work,'modeled_parallel_factor':F(h)/latency})
    return r
