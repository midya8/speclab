"""Reproducible probability/counter figure; numbers come only from exact reports."""
import base64,html,json,sys,zlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from speclab.stochastic_demo import demo
from speclab.finite import jsonify


def values(r):
    d={}
    for i,c in enumerate(r['one_step']['cells']):
        for key in ('target','draft','accepted_mass','residual_probability','corrected_output_mass','wrong_target_fallback_mass'):d[f'one-{i}-{key}']=c[key]
    p=r['prefix_dependent'];tables=[p['naive_target_law'],p['corrected']['sequence_law'],p['wrong_target_fallback_control']['sequence_law']]
    maps=[{''.join(x['tokens']):x['probability'] for x in rows} for rows in tables]
    for i,seq in enumerate(sorted(maps[0])):
        d[f'seq-{i}']=seq
        for j,m in enumerate(maps):d[f'prob-{i}-{j}']=m.get(seq,'0')
    d['correct-tv']=p['corrected_total_variation'];d['wrong-tv']=p['wrong_total_variation']
    for i,key in enumerate(('identical','prefix_dependent','disjoint_support')):
        for metric in ('verification_rounds','target_positions','draft_calls'):d[f'cost-{i}-{metric}']=r[key]['corrected']['expected_counts'][metric]
    for i,c in enumerate(r['hypothetical_cost_scenarios']):
        for metric in ('modeled_parallel_cost','weighted_position_score_cost','modeled_parallel_factor'):d[f'model-{i}-{metric}']=c[metric]
    return d


def render(report):
    r=jsonify(report);d=values(r)
    out=['<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="1570" viewBox="0 0 1120 1570">',
         '<rect width="1120" height="1570" fill="#edf2f5"/><rect x="16" y="16" width="1088" height="1538" rx="18" fill="white"/>',
         '<style>text{font-family:Arial,sans-serif;fill:#24313b}.muted{fill:#5c6c78}</style>']
    def t(x,y,s,size=17,extra=''):out.append(f'<text x="{x}" y="{y}" font-size="{size}" {extra}>{html.escape(str(s))}</text>')
    def cell(key,x,y):t(x,y,d[key],18,f'id="cell-{key}" data-value="{d[key]}"')
    t(42,58,'SpecLab / exact sampling is more than matching greedy tokens',25,'font-weight="700"')
    t(42,90,'Finite supplied autoregressive tables. Rational probability laws and explicit decoder-position counts.',17,'class="muted"')
    t(42,137,'1 Rejection needs a residual, not a fresh target draw',22,'font-weight="700"')
    t(42,170,'Accept x with min(1, p(x)/q(x)); on rejection draw from normalized max(0,p-q).',18)
    cols=[(150,'Target p','target'),(300,'Draft q','draft'),(455,'Accepted mass','accepted_mass'),(635,'Residual law','residual_probability'),(800,'Correct output','corrected_output_mass'),(970,'Wrong output','wrong_target_fallback_mass')]
    for x,label,key in cols:t(x,213,label,15,'font-weight="700"')
    for i,tok in enumerate(('A','B')):
        y=253+i*38;t(65,y,tok,18,'font-weight="700"')
        for x,_,key in cols:cell(f'one-{i}-{key}',x+15,y)
    t(42,334,'Wrong fallback yields (5/8,3/8), not (1/2,1/2). Same target distribution is not same RNG trace.',17)
    t(42,381,'2 Enumerate the whole prefix-dependent four-token law',22,'font-weight="700"')
    for x,label in ((80,'Sequence'),(300,'Target only'),(535,'Corrected sampler'),(815,'Wrong target fallback')):t(x,421,label,17,'font-weight="700"')
    for i in range(16):
        y=455+29*i
        if i%2==0:out.append(f'<rect x="55" y="{y-21}" width="1005" height="28" fill="#eef4f4"/>')
        cell(f'seq-{i}',90,y)
        for j,x in enumerate((315,570,860)):cell(f'prob-{i}-{j}',x,y)
    t(65,966,'Total variation from target:',18);cell('correct-tv',570,966);cell('wrong-tv',860,966)
    t(42,1015,'3 Fewer modeled rounds need not mean less scoring work',22,'font-weight="700"')
    for x,label in ((65,'Four-token fixture'),(400,'Rounds'),(620,'Target positions'),(880,'Draft calls')):t(x,1056,label,17,'font-weight="700"')
    for i,label in enumerate(('Identical draft','Prefix-dependent draft','Disjoint support')):
        y=1093+35*i;t(65,y,label,17)
        for metric,x in (('verification_rounds',420),('target_positions',670),('draft_calls',900)):cell(f'cost-{i}-{metric}',x,y)
    t(42,1212,'Hypothetical costs only: each round costs 1; each draft score costs 1/10.',18)
    t(42,1243,'Separate score-work scenario charges each target position 1, not each batch 1.',17,'class="muted"')
    for x,label in ((65,'Same fixture order'),(400,'Parallel cost'),(620,'Score-work cost'),(880,'Modeled factor')):t(x,1287,label,17,'font-weight="700"')
    for i,label in enumerate(('Identical','Prefix-dependent','Disjoint')):
        y=1325+35*i;t(65,y,label,17)
        for metric,x in (('modeled_parallel_cost',420),('weighted_position_score_cost',670),('modeled_parallel_factor',900)):cell(f'model-{i}-{metric}',x,y)
    t(42,1440,'Naive baseline: four rounds, four target positions, zero draft calls; either assigned cost is 4.',17)
    t(42,1474,'Full proposal paths, first-rejection/bonus branches, sequence x cost laws and sampled traces are retained.',16)
    t(42,1516,'No transformer throughput, GPU batching, KV cache, EOS, numerical-kernel or real-LLM quality claim.',16,'class="muted"')
    blob=base64.b64encode(zlib.compress(json.dumps(r,sort_keys=True,separators=(',',':')).encode())).decode()
    out.append('<metadata id="report-zlib-base64">'+blob+'</metadata></svg>');return '\n'.join(out)+'\n'


if __name__=='__main__':
    path=Path(__file__).resolve().parents[1]/'assets/stochastic-audit.svg';text=render(demo())
    if sys.argv[1:]==['--check']:
        if path.read_text()!=text:raise SystemExit('stochastic SVG differs')
    elif not sys.argv[1:]:path.parent.mkdir(exist_ok=True);path.write_text(text)
    else:raise SystemExit('usage: render_stochastic.py [--check]')
