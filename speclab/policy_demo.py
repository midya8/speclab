"""Prefix-oracle policy controls; every cost below is deliberately hypothetical."""
from fractions import Fraction as F
from .stochastic_demo import model
from .policy import optimize


def fixtures():
    v=['A','B']
    def p(x):return [F(1,2),F(1,2)] if not x else ([1,0] if x[0]=='A' else [0,1])
    def q(x):return [F(1,2),F(1,2)] if not x else [1,0]
    return model(v,4,p),model(v,4,q)


def demo():
    p,q=fixtures()
    parallel={'verification_rounds':1,'target_positions':0,'draft_calls':'1/10'}
    work={'verification_rounds':0,'target_positions':1,'draft_calls':'1/10'}
    a=optimize(p,q,parallel);b=optimize(p,q,work)
    assert a['optimal']['expected_assigned_cost']==F(23,10)
    assert a['best_fixed_assigned_cost']==F(47,20)
    assert a['gain_over_best_fixed']==F(1,20)
    assert b['optimal']['expected_assigned_cost']==4
    assert all(x['selected_action']==0 for x in b['bellman_certificates'])
    return {'contract':'full-known-table oracle; no measured runtime or planning-overhead model',
        'prefix_switch':a,'score_work':b,
        'boundary':'g=min(action,remaining-1); canonical action zero at final slot',
        'limits':'no EOS, learned predictor, on-the-fly draft stopping, GPU or transformer execution'}
