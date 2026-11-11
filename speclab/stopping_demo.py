"""Public hand-specified finite controls, not pretrained-model benchmarks."""
from fractions import Fraction as F
from .finite import jsonify
from .stochastic_demo import model
from .stopping import audit


def fixtures():
    out=[]
    def add(name,title,h,p,q,g=3,t='1/2',v=('A','B')):
        a=model(v,h,p);b=model(v,h,q);out.append({'id':name,'title':title,'input':jsonify({'target':a.to_dict(),'draft':b.to_dict(),'max_gamma':g,'threshold':t})})
    uniform=lambda p:[F(1,2),F(1,2)]
    biased=lambda p:[F(3,4),F(1,4)]
    add('censor_bias','A discarded proposal changes the law',2,uniform,biased,1)
    add('keep_changes_work','Keeping the stop token changes work, not law',4,uniform,biased,3)
    add('prefix_switch','Confidence changes with the draft prefix',4,lambda p:[F(3,4),F(1,4)]if not p or p[-1]=='A'else[F(1,4),F(3,4)],lambda p:[F(3,4),F(1,4)]if len(p)%2==0 else uniform(p),3,'3/5')
    add('target_equal','Even identical models are not safe to censor',3,biased,biased)
    add('disjoint','No overlap: every proposal is rejected',4,lambda p:[1,0],lambda p:[0,1])
    add('boundary_equal','Equality does not trigger a strict threshold',3,uniform,uniform,3,'1/2')
    add('threshold_zero','Threshold zero never stops early',4,uniform,biased,3,'0')
    add('threshold_one','Before-next pays for the distribution it inspects',4,uniform,biased,3,'1')
    add('one_token','Final token: no room for a proposal',1,uniform,biased)
    add('three_tokens','Three-way support and residuals',3,lambda p:[F(1,3)]*3,lambda p:[F(1,2),F(1,3),F(1,6)],2,'2/5',('A','B','C'))
    add('tiny_mass','Exact tiny mass, below binary64 resolution near one',3,uniform,lambda p:[1-F(1,2**128),F(1,2**128)],2,'1/2')
    add('zero_probability','A zero-probability proposal is never sampled',3,uniform,lambda p:[1,0],2,'3/4')
    return out


def decode_input(data):
    from .finite import FiniteModel,rational
    if not isinstance(data,dict)or set(data)!={'target','draft','max_gamma','threshold'}:raise ValueError('wrong stopping input keys')
    return FiniteModel.from_dict(data['target']),FiniteModel.from_dict(data['draft']),data['max_gamma'],rational(data['threshold'])


def demo():
    return [dict(f,report=jsonify(audit(*decode_input(f['input']))))for f in fixtures()]
