'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),{spawnSync}=require('node:child_process');const {audit,Q}=require('../docs/stopping-engine.js'),fixtures=(()=>{const r=spawnSync(process.env.PYTHON||'python3',['-m','speclab.stopping_cli','demo'],{cwd:path.resolve(__dirname,'..'),encoding:'utf8',maxBuffer:20_000_000});if(r.status!==0)throw Error(r.stderr);return JSON.parse(r.stdout);})();
test('all native Python evidence matches independent BigInt arithmetic, including every witness and law',()=>{for(const f of fixtures)assert.deepEqual(audit(f.input),f.report,f.id);});
test('125 binary target/draft/threshold combinations and mixed-case three-token prefix tables',()=>{const code=`import json,random
from fractions import Fraction as F
from itertools import product
from speclab.stochastic_demo import model
from speclab.stopping import audit
from speclab.finite import jsonify
rows=[]
for a,b,t in product(range(5),repeat=3):
 p=model(['A','B'],2,lambda x:[F(a,4),F(4-a,4)]);q=model(['A','B'],2,lambda x:[F(b,4),F(4-b,4)])
 data={'target':p.to_dict(),'draft':q.to_dict(),'max_gamma':2,'threshold':F(t,4)};rows.append({'input':data,'report':audit(p,q,2,F(t,4))})
rng=random.Random(17)
for i in range(8):
 def row(x):
  xs=[rng.randrange(1,6)for _ in range(3)];return[F(v,sum(xs))for v in xs]
 p=model(['a','B','AA'],3,row);q=model(['a','B','AA'],3,row);data={'target':p.to_dict(),'draft':q.to_dict(),'max_gamma':3,'threshold':F(i,8)};rows.append({'input':data,'report':audit(p,q,3,F(i,8))})
print(json.dumps(jsonify(rows)))`;
const run=spawnSync(process.env.PYTHON||'python3',['-c',code],{cwd:path.resolve(__dirname,'..'),encoding:'utf8',maxBuffer:100_000_000});assert.equal(run.status,0,run.stderr);for(const f of JSON.parse(run.stdout))assert.deepEqual(audit(f.input),f.report);});
test('strict inputs and exact rational parsing',()=>{for(const x of [true,.5,'NaN','1/0','1e2','1'.repeat(161)])assert.throws(()=>Q.from(x));assert.equal(Q.from('-0.25').toString(),'-1/4');for(const g of [0,4,true,'1',1.5]){const f=structuredClone(fixtures[0].input);f.max_gamma=g;assert.throws(()=>audit(f));}for(const t of ['-1','2',true]){const f=structuredClone(fixtures[0].input);f.threshold=t;assert.throws(()=>audit(f));}for(const mutate of [x=>x.extra=1,x=>x.target.table.push(x.target.table[0]),x=>x.target.table.pop(),x=>x.draft.vocabulary.reverse(),x=>x.target.table[0].probabilities=['1','1']]){const f=structuredClone(fixtures[0].input);mutate(f);assert.throws(()=>audit(f));}});
