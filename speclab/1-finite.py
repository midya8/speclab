"""Strict exact finite autoregressive tables, independent of the n-gram demo."""
from fractions import Fraction as F
from itertools import product
from math import lcm
import re


def rational(x):
    if isinstance(x,bool) or not isinstance(x,(str,int,F)):raise ValueError('exact integer/string/Fraction required, no floats/bools')
    if isinstance(x,str) and (len(x)>160 or not re.fullmatch(r'-?\d+(?:/\d+|\.\d+)?',x)):raise ValueError('invalid rational syntax')
    try:q=F(x)
    except (ValueError,ZeroDivisionError):raise ValueError('invalid rational') from None
    if max(abs(q.numerator).bit_length(),q.denominator.bit_length())>512:raise ValueError('input exceeds 512-bit components')
    return q


def distribution(xs,size):
    if not isinstance(xs,(list,tuple)) or len(xs)!=size:raise ValueError('probability vector shape mismatch')
    ps=tuple(rational(x) for x in xs)
    if any(x<0 for x in ps) or sum(ps)!=1:raise ValueError('probabilities must be nonnegative and sum exactly to one')
    return ps


class FiniteModel:
    """Complete distributions at all prefixes, including unreachable ones."""
    def __init__(self,vocabulary,horizon,table):
        if not isinstance(vocabulary,(list,tuple)) or not 1<=len(vocabulary)<=3:raise ValueError('vocabulary size must be 1..3')
        if any(not isinstance(t,str) or not 1<=len(t)<=16 or not t.isascii() or not t.isalnum() for t in vocabulary):raise ValueError('tokens must be 1..16 ASCII alphanumeric characters')
        if len(set(vocabulary))!=len(vocabulary):raise ValueError('duplicate vocabulary token')
        if isinstance(horizon,bool) or not isinstance(horizon,int) or not 1<=horizon<=4:raise ValueError('horizon must be 1..4')
        if not isinstance(table,dict):raise ValueError('table must map tuple prefixes to distributions')
        self.vocabulary=tuple(vocabulary);self.horizon=horizon
        wanted={x for n in range(horizon) for x in product(self.vocabulary,repeat=n)}
        if set(table)!=wanted:raise ValueError('table must cover every prefix shorter than horizon exactly')
        self.table={k:distribution(v,len(vocabulary)) for k,v in table.items()}

    def at(self,prefix):
        try:return self.table[tuple(prefix)]
        except (KeyError,TypeError):raise ValueError('invalid or complete prefix') from None

    @classmethod
    def from_dict(cls,data):
        if not isinstance(data,dict) or set(data)!={'vocabulary','horizon','table'}:raise ValueError('wrong model keys')
        if not isinstance(data['table'],list):raise ValueError('table must be records')
        table={}
        for rec in data['table']:
            if not isinstance(rec,dict) or set(rec)!={'prefix','probabilities'} or not isinstance(rec['prefix'],list):raise ValueError('wrong table record')
            if any(not isinstance(t,str) for t in rec['prefix']):raise ValueError('prefix tokens must be strings')
            key=tuple(rec['prefix'])
            if key in table:raise ValueError('duplicate prefix')
            table[key]=rec['probabilities']
        return cls(data['vocabulary'],data['horizon'],table)

    def to_dict(self):
        return {'vocabulary':list(self.vocabulary),'horizon':self.horizon,
                'table':[{'prefix':list(k),'probabilities':list(self.table[k])} for k in sorted(self.table,key=lambda k:(len(k),k))]}


def draw(ps,rng):
    # Exact integer categorical sampling, no floating-point CDF.
    denominator=lcm(*(x.denominator for x in ps));ticket=rng.randrange(denominator);c=0
    for i,p in enumerate(ps):
        c+=p.numerator*(denominator//p.denominator)
        if ticket<c:return i
    raise AssertionError('invalid draw distribution')


def jsonify(x):
    if isinstance(x,F):return str(x)
    if isinstance(x,dict):return {k:jsonify(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [jsonify(v) for v in x]
    return x
