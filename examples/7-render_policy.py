"""Data-bound finite oracle-policy figure; no handmade metrics."""
import base64,html,json,sys,zlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from speclab.policy_demo import demo
from speclab.finite import jsonify


def values(r):
    a=r['prefix_switch'];out={}
    for i,row in enumerate(a['fixed_baselines']+[{'requested_gamma':'adaptive',**a['optimal']}]):
        out[f'base-{i}-action']=row['requested_gamma'];out[f'base-{i}-cost']=row['expected_assigned_cost']
        for key in ('verification_rounds','target_positions','draft_calls'):out[f'base-{i}-{key}']=row['expected_counts'][key]
    reachable={tuple(x['prefix']) for x in a['optimal']['reachable_kernels']}
    for i,row in enumerate(a['bellman_certificates']):
        out[f'policy-{i}-prefix']=''.join(row['prefix']) or '(empty)';out[f'policy-{i}-action']=row['selected_action'];out[f'policy-{i}-value']=row['value'];out[f'policy-{i}-reachable']='yes' if tuple(row['prefix']) in reachable else 'no'
    for i,row in enumerate(a['bellman_certificates'][0]['choices']):
        for key in ('action','immediate_expected_cost','future_expected_cost','q_value','bellman_slack'):out[f'root-{i}-{key}']=row[key]
    for i,row in enumerate(a['optimal']['cost_law']):
        out[f'law-{i}-cost']=row['assigned_cost'];out[f'law-{i}-prob']=row['probability']
    for i,row in enumerate(r['score_work']['fixed_baselines']):out[f'work-{i}-cost']=row['expected_assigned_cost']
    out['work-optimal']=r['score_work']['optimal']['expected_assigned_cost'];out['gain']=a['gain_over_best_fixed']
    return out


def render(report):
    r=jsonify(report);v=values(r);out=['<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="1540" viewBox="0 0 1120 1540">']
    payload=base64.b64encode(zlib.compress(json.dumps(r,sort_keys=True,separators=(',',':')).encode())).decode();out.append('<metadata>'+payload+'</metadata>')
    out+=['<rect width="1120" height="1540" fill="#edf3f5"/>','<rect x="16" y="16" width="1088" height="1508" rx="18" fill="white"/>']
    def text(x,y,s,size=17,color='#273746',bold=False):out.append(f'<text x="{x}" y="{y}" font-family="Arial,sans-serif" font-size="{size}" fill="{color}"'+(' font-weight="bold"' if bold else '')+'>'+html.escape(str(s))+'</text>')
    def cell(x,y,key):out.append(f'<text id="cell-{key}" data-value="{html.escape(str(v[key]),quote=True)}" x="{x}" y="{y}" font-family="Arial,sans-serif" font-size="17" fill="#273746">'+html.escape(str(v[key]))+'</text>')
    text(40,58,'SpecLab / choose lookahead, including no speculation',26,bold=True)
    text(40,87,'Oracle finite tables. Exact Bellman values and path laws, not a measured latency predictor.',16,'#627789')
    text(40,132,'1 A full-strength fixed baseline, not a weak fixed gamma',21,bold=True)
    text(40,163,'Target: root A/B equally likely, then repeat that first token. Draft: root A/B, then always A.')
    text(40,191,'Four output tokens; assigned price = verification rounds + draft calls / 10.')
    text(40,219,'Exact target law: AAAA with 1/2, BBBB with 1/2. Every policy preserves this law.')
    xs=[55,270,470,670,870]
    for x,label in zip(xs,('Requested gamma','Expected rounds','Target positions','Draft calls','Assigned cost')):text(x,260,label,16,bold=True)
    for i in range(5):
        y=293+i*35
        if i%2==0:out.append(f'<rect x="40" y="{y-23}" width="1030" height="33" fill="#edf3f5"/>')
        for x,key in zip(xs,('action','verification_rounds','target_positions','draft_calls','cost')):cell(x,y,f'base-{i}-{key}')
    text(40,490,'Adaptive gain over best fixed (gamma 3):');cell(415,490,'gain')
    text(40,535,'2 Bellman certificate at every prefix',21,bold=True)
    text(40,563,'Unreachable prefixes are solved too, but never charged to the executed policy.',16,'#627789')
    left=[50,205,315,425]
    for x,label in zip(left,('Prefix','Action','Tail value','Reached?')):text(x,603,label,16,bold=True)
    for i in range(15):
        y=634+27*i
        if i%2==0:out.append(f'<rect x="40" y="{y-20}" width="485" height="26" fill="#edf3f5"/>')
        for x,key in zip(left,('prefix','action','value','reachable')):cell(x,y,f'policy-{i}-{key}')
    text(565,603,'Root action values with optimal future',18,bold=True)
    right=[570,655,755,855,970]
    for x,label in zip(right,('Action','Now','Future','Total Q','Slack')):text(x,640,label,15,bold=True)
    for i in range(4):
        for x,key in zip(right,('action','immediate_expected_cost','future_expected_cost','q_value','bellman_slack')):cell(x,675+i*36,f'root-{i}-{key}')
    text(565,845,'Actual adaptive path-cost distribution',18,bold=True)
    text(575,880,'Assigned cost',16,bold=True);text(820,880,'Probability',16,bold=True)
    for i in range(2):cell(585,916+35*i,f'law-{i}-cost');cell(840,916+35*i,f'law-{i}-prob')
    text(565,1003,'Reached prefixes: empty, BB, BBB.',16)
    text(565,1031,'Action 3 at root; action 0 after BB.',16)
    text(40,1090,'3 Score-work prices reverse the decision',21,bold=True)
    text(40,1120,'Now charge each target position 1 and each draft call 1/10; verification rounds are free.')
    text(40,1150,'The optimal action is 0 at every prefix, with assigned cost');cell(690,1150,'work-optimal')
    for i,x in enumerate((80,330,580,830)):
        text(x,1194,'Fixed gamma '+str(i),16,bold=True);cell(x+35,1228,f'work-{i}-cost')
    text(40,1285,'No speculation is a genuine action: one target token, one round, zero draft calls.')
    text(40,1315,'Boundary contract: g = min(action, remaining - 1); final slot has canonical action 0.')
    text(40,1345,'Ties choose the smallest action. All choices and nonnegative Bellman slacks are retained.')
    text(40,1388,'Known target/draft tables are an oracle input. Planning and policy lookup overhead are not modeled.',16,'#627789')
    text(40,1416,'This chooses a length before drafting, not a learned confidence/entropy threshold during drafting.',16,'#627789')
    text(40,1454,'No EOS, predictor training, transformer/GPU execution, throughput or novel algorithm claim.',16,'#627789')
    text(40,1482,'Complete Bellman transitions, model inputs, policy and joint sequence/count laws remain in metadata.',16,'#627789')
    out.append('</svg>');return '\n'.join(out)+'\n'

if __name__=='__main__':
    path=Path(__file__).resolve().parents[1]/'assets/lookahead-policy.svg';text=render(demo())
    if '--check' in sys.argv:
        if path.read_text()!=text:raise SystemExit('policy figure differs')
    else:path.write_text(text)
