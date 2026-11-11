"""Exact finite within-round draft stopping. Original SpecLab engines unchanged."""
from collections import defaultdict
from fractions import Fraction as F
from functools import lru_cache
from .finite import FiniteModel,rational,jsonify

MODES=('fixed','before_max','after_keep','after_censor','no_speculation')
METRICS=('verification_rounds','target_positions','draft_calls','sampled_proposals','retained_proposals','censored_proposals','accepted_proposals','rejection_rounds','bonus_tokens')


def validate(target,draft,max_gamma,threshold,mode):
    if not isinstance(target,FiniteModel)or not isinstance(draft,FiniteModel)or target.vocabulary!=draft.vocabulary or target.horizon!=draft.horizon:raise ValueError('matching finite model pair required')
    if type(max_gamma)is not int or not 1<=max_gamma<=3:raise ValueError('max_gamma must be 1..3')
    threshold=rational(threshold)
    if not 0<=threshold<=1:raise ValueError('threshold must be 0..1')
    if mode not in MODES:raise ValueError('unknown stopping mode')
    return threshold


def proposals(target,draft,prefix,max_gamma,threshold,mode):
    """Every stopped draft leaf, including any sampled-but-censored token."""
    threshold=validate(target,draft,max_gamma,threshold,mode);prefix=tuple(prefix);target.at(prefix)
    limit=min(max_gamma,target.horizon-len(prefix)-1);v=target.vocabulary;leaves=[]
    def walk(kept,sampled,mass,calls,decisions):
        if len(kept)==limit or mode=='no_speculation':
            leaves.append({'retained':list(kept),'sampled':list(sampled),'probability':mass,'draft_calls':calls,'censored':0,'reason':'no-speculation'if mode=='no_speculation'else'horizon-or-limit','decisions':decisions});return
        at=prefix+kept;q=draft.at(at);calls+=1
        if mode=='before_max'and max(q)<threshold:
            leaves.append({'retained':list(kept),'sampled':list(sampled),'probability':mass,'draft_calls':calls,'censored':0,'reason':'before-max-below-threshold','decisions':decisions+[{'prefix':list(at),'criterion':max(q),'sampled_token':None,'stop':True}]});return
        for j,tok in enumerate(v):
            if not q[j]:continue
            low=q[j]<threshold;stop=mode in ('after_keep','after_censor')and low
            ds=decisions+[{'prefix':list(at),'criterion':max(q)if mode=='before_max'else q[j],'sampled_token':tok,'stop':stop}]
            if stop:
                keep=kept if mode=='after_censor'else kept+(tok,)
                leaves.append({'retained':list(keep),'sampled':list(sampled+(tok,)),'probability':mass*q[j],'draft_calls':calls,'censored':int(mode=='after_censor'),'reason':'sample-below-threshold-'+('censored'if mode=='after_censor'else'kept'),'decisions':ds})
            else:walk(kept+(tok,),sampled+(tok,),mass*q[j],calls,ds)
    walk((),(),F(1),0,[])
    assert sum(r['probability']for r in leaves)==1
    return leaves


def round_kernel(target,draft,prefix,max_gamma,threshold,mode):
    leaves=proposals(target,draft,prefix,max_gamma,threshold,mode);prefix=tuple(prefix);v=target.vocabulary;branches=[]
    for leaf_id,leaf in enumerate(leaves):
        path=tuple(leaf['retained']);survive=F(1);mass=leaf['probability'];g=len(path)
        def add(accepted,reject,distribution,base):
            for k,p in enumerate(distribution):
                if not p or not base:continue
                counts=(1,g+1,leaf['draft_calls'],len(leaf['sampled']),g,leaf['censored'],accepted,int(reject is not None),int(reject is None))
                branches.append({'proposal_leaf':leaf_id,'accepted_count':accepted,'reject_index':reject,'replacement_token':v[k],'replacement_distribution':list(distribution),'replacement_probability':p,'emitted':list(path[:accepted]) + [v[k]],'probability':base*p,'counts':dict(zip(METRICS,counts))})
        for j,tok in enumerate(path):
            at=prefix+path[:j];p=target.at(at);q=draft.at(at);k=v.index(tok);alpha=min(F(1),p[k]/q[k]);reject=mass*survive*(1-alpha)
            if reject:
                diff=[max(F(0),a-b)for a,b in zip(p,q)];z=sum(diff);assert z>0;add(j,j,[x/z for x in diff],reject)
            survive*=alpha
        add(g,None,target.at(prefix+path),mass*survive)
    assert sum(r['probability']for r in branches)==1
    return {'prefix':list(prefix),'mode':mode,'max_gamma':max_gamma,'threshold':rational(threshold),'proposal_leaves':leaves,'branches':branches}


def decode(target,draft,max_gamma,threshold,mode):
    threshold=validate(target,draft,max_gamma,threshold,mode);kernels={}
    @lru_cache(None)
    def visit(prefix):
        if len(prefix)==target.horizon:return {(prefix,(0,)*len(METRICS)):F(1)}
        k=round_kernel(target,draft,prefix,max_gamma,threshold,mode);kernels[prefix]=k;out=defaultdict(F)
        for branch in k['branches']:
            now=tuple(branch['counts'][key]for key in METRICS)
            for (seq,counts),p in visit(prefix+tuple(branch['emitted'])).items():out[(seq,tuple(a+b for a,b in zip(now,counts)))]+=branch['probability']*p
        return dict(out)
    joint=visit(());sequence=defaultdict(F);counts=defaultdict(F)
    for (seq,c),p in joint.items():sequence[seq]+=p;counts[c]+=p
    assert sum(sequence.values())==1
    expected={key:sum(F(c[j])*p for c,p in counts.items())for j,key in enumerate(METRICS)}
    assert expected['target_positions']==expected['verification_rounds']+expected['retained_proposals']
    assert expected['sampled_proposals']==expected['retained_proposals']+expected['censored_proposals']
    assert expected['accepted_proposals']+expected['rejection_rounds']+expected['bonus_tokens']==target.horizon
    return {'mode':mode,'sequence_law':[{'tokens':list(seq),'probability':p}for seq,p in sorted(sequence.items())],
            'count_law':[{'counts':dict(zip(METRICS,c)),'probability':p}for c,p in sorted(counts.items())],
            'joint_sequence_count_law':[{'tokens':list(seq),'counts':dict(zip(METRICS,c)),'probability':p}for (seq,c),p in sorted(joint.items())],
            'expected_counts':expected,'kernels':[kernels[k]for k in sorted(kernels,key=lambda k:(len(k),k))]}


def audit(target,draft,max_gamma,threshold):
    validate(target,draft,max_gamma,threshold,'fixed');v=target.vocabulary
    # Target-only law computed directly, independently of the round process.
    naive={}
    def target_walk(seq,mass):
        if len(seq)==target.horizon:naive[seq]=mass;return
        for tok,p in zip(v,target.at(seq)):
            if p:target_walk(seq+(tok,),mass*p)
    target_walk((),F(1));reports={}
    for mode in MODES:
        report=decode(target,draft,max_gamma,threshold,mode);law={tuple(r['tokens']):r['probability']for r in report['sequence_law']};keys=set(naive)|set(law)
        report['total_variation']=sum(abs(law.get(k,F(0))-naive.get(k,F(0)))for k in keys)/2
        if mode!='after_censor':assert report['total_variation']==0
        reports[mode]=report
    return {'schema':'speclab-stopping-v1','target':target.to_dict(),'draft':draft.to_dict(),'max_gamma':max_gamma,'threshold':rational(threshold),
            'naive_target_law':[{'tokens':list(k),'probability':p}for k,p in sorted(naive.items())],'modes':reports,
            'fixed_baselines':{str(g):decode(target,draft,g,threshold,'fixed')for g in (1,2,3)}}
