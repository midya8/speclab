"""Independent full-path and Bernoulli-product reference for stopping evidence.

Never imports the stopping producer. Enumerates even unobserved draft suffixes,
then marginalizes those suffixes rather than using its stopped-tree recursion.
"""
from collections import defaultdict
from fractions import Fraction as R
from itertools import product
from .finite import FiniteModel,rational

FIELDS=('verification_rounds','target_positions','draft_calls','sampled_proposals','retained_proposals','censored_proposals','accepted_proposals','rejection_rounds','bonus_tokens')
RULES=('fixed','before_max','after_keep','after_censor','no_speculation')


def reference(target,draft,maximum,threshold,*,_baselines=True):
    if type(target)is not FiniteModel or type(draft)is not FiniteModel or target.vocabulary!=draft.vocabulary or target.horizon!=draft.horizon:raise ValueError('matching finite models required')
    if type(maximum)is not int or maximum<1 or maximum>3:raise ValueError('maximum 1..3')
    threshold=rational(threshold)
    if threshold<0 or threshold>1:raise ValueError('threshold 0..1')
    alphabet=target.vocabulary;h=target.horizon
    def kernel(prefix,mode):
        limit=min(maximum,h-len(prefix)-1);grouped={}
        for path in product(alphabet,repeat=limit):
            raw=R(1)
            for i,tok in enumerate(path):raw*=draft.at(prefix+path[:i])[alphabet.index(tok)]
            if not raw:continue
            retained=[];sampled=[];decisions=[];calls=0;censored=0;reason='horizon-or-limit'
            if mode=='no_speculation':reason='no-speculation'
            else:
                for i,tok in enumerate(path):
                    at=prefix+tuple(retained);q=draft.at(at);calls+=1
                    if mode=='before_max'and max(q)<threshold:
                        decisions.append({'prefix':list(at),'criterion':max(q),'sampled_token':None,'stop':True});reason='before-max-below-threshold';break
                    tokenp=q[alphabet.index(tok)];stop=mode in ('after_keep','after_censor')and tokenp<threshold
                    decisions.append({'prefix':list(at),'criterion':max(q)if mode=='before_max'else tokenp,'sampled_token':tok,'stop':stop});sampled.append(tok)
                    if stop and mode=='after_censor':censored=1;reason='sample-below-threshold-censored';break
                    retained.append(tok)
                    if stop:reason='sample-below-threshold-kept';break
            key=(tuple(retained),tuple(sampled),calls,censored,reason)
            if key not in grouped:grouped[key]={'retained':retained,'sampled':sampled,'probability':R(0),'draft_calls':calls,'censored':censored,'reason':reason,'decisions':decisions}
            grouped[key]['probability']+=raw
        leaves=list(grouped.values());branches=[]
        for leaf_id,leaf in enumerate(leaves):
            path=tuple(leaf['retained']);g=len(path);accept=[];pvec=[];qvec=[]
            for i,tok in enumerate(path):
                p=target.at(prefix+path[:i]);q=draft.at(prefix+path[:i]);pvec.append(p);qvec.append(q);accept.append(min(R(1),p[alphabet.index(tok)]/q[alphabet.index(tok)]))
            for failed in list(range(g))+[None]:
                accepted=g if failed is None else failed;weight=leaf['probability']
                for a in accept[:accepted]:weight*=a
                if failed is not None:weight*=1-accept[failed]
                if not weight:continue
                if failed is None:replacement=target.at(prefix+path)
                else:
                    raw=[max(R(0),pvec[failed][j]-qvec[failed][j])for j in range(len(alphabet))];z=sum(raw);replacement=[x/z for x in raw]
                for tok,prob in zip(alphabet,replacement):
                    if not prob:continue
                    branches.append({'proposal_leaf':leaf_id,'accepted_count':accepted,'reject_index':failed,'replacement_token':tok,'replacement_distribution':list(replacement),'replacement_probability':prob,'emitted':list(path[:accepted]) + [tok],'probability':weight*prob,
                                     'counts':dict(zip(FIELDS,(1,g+1,leaf['draft_calls'],len(leaf['sampled']),g,leaf['censored'],accepted,int(failed is not None),int(failed is None))))})
        return {'prefix':list(prefix),'mode':mode,'max_gamma':maximum,'threshold':threshold,'proposal_leaves':leaves,'branches':branches}
    naive={}
    for seq in product(alphabet,repeat=h):
        mass=R(1)
        for i,tok in enumerate(seq):mass*=target.at(seq[:i])[alphabet.index(tok)]
        if mass:naive[seq]=mass
    reports={}
    for mode in RULES:
        # Forward state expansion, not recursive suffix-law multiplication.
        pending={((),(0,)*len(FIELDS)):R(1)};joint=defaultdict(R);kernels={}
        while pending:
            nxt=defaultdict(R)
            for (seq,count),mass in pending.items():
                if len(seq)==h:joint[(seq,count)]+=mass;continue
                if seq not in kernels:kernels[seq]=kernel(seq,mode)
                for b in kernels[seq]['branches']:
                    newseq=seq+tuple(b['emitted']);newcounts=tuple(count[i]+b['counts'][key]for i,key in enumerate(FIELDS));nxt[(newseq,newcounts)]+=mass*b['probability']
            pending=nxt
        seqs=defaultdict(R);counts=defaultdict(R)
        for (seq,c),mass in joint.items():seqs[seq]+=mass;counts[c]+=mass
        keys=set(seqs)|set(naive);tv=sum(abs(seqs.get(k,R(0))-naive.get(k,R(0)))for k in keys)/2
        reports[mode]={'mode':mode,'sequence_law':[{'tokens':list(seq),'probability':p}for seq,p in sorted(seqs.items())],
                       'count_law':[{'counts':dict(zip(FIELDS,c)),'probability':p}for c,p in sorted(counts.items())],
                       'joint_sequence_count_law':[{'tokens':list(seq),'counts':dict(zip(FIELDS,c)),'probability':p}for (seq,c),p in sorted(joint.items())],
                       'expected_counts':{key:sum(c[i]*mass for c,mass in counts.items())for i,key in enumerate(FIELDS)},
                       'kernels':[kernels[k]for k in sorted(kernels,key=lambda k:(len(k),k))],'total_variation':tv}
    return {'schema':'speclab-stopping-v1','target':target.to_dict(),'draft':draft.to_dict(),'max_gamma':maximum,'threshold':threshold,
            'naive_target_law':[{'tokens':list(seq),'probability':p}for seq,p in sorted(naive.items())],'modes':reports,
            **({'fixed_baselines':{str(g):{k:v for k,v in reference(target,draft,g,threshold,_baselines=False)['modes']['fixed'].items()if k!='total_variation'}for g in (1,2,3)}}if _baselines else {})}


def verify(target,draft,maximum,threshold,report):
    from .finite import jsonify
    expected=jsonify(reference(target,draft,maximum,threshold));report=jsonify(report)
    def compare(a,b,path=''):
        if type(a)is not type(b):raise ValueError('type mismatch '+path)
        if isinstance(a,dict):
            if a.keys()!=b.keys():raise ValueError('field mismatch '+path)
            for key in a:compare(a[key],b[key],path+'/'+key)
        elif isinstance(a,list):
            if len(a)!=len(b):raise ValueError('length mismatch '+path)
            for i,(x,y)in enumerate(zip(a,b)):compare(x,y,path+'/'+str(i))
        elif a!=b:raise ValueError('value mismatch '+path)
    compare(expected,report)
    return {'verified':True,'modes':len(RULES),'kernels':sum(len(r['kernels'])for r in expected['modes'].values())}
