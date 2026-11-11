"""Audit and independently verify finite adaptive draft-stopping reports."""
import argparse,json
from pathlib import Path
from .finite import jsonify
from .stopping_demo import decode_input,demo
from .stopping import audit
from .stopverify import verify


def load(path):
    raw=Path(path).read_bytes()
    if len(raw)>8_000_000:raise ValueError('file exceeds 8 MB')
    def unique(pairs):
        result={}
        for key,value in pairs:
            if key in result:raise ValueError('duplicate JSON key')
            result[key]=value
        return result
    def reject(value):raise ValueError('JSON numbers must be integers, exact probabilities use strings')
    # Bound depth before parsing. Braces in strings and escapes do not count.
    depth=0;string=False;escaped=False
    for c in raw.decode('utf-8'):
        if string:
            if escaped:escaped=False
            elif c=='\\':escaped=True
            elif c=='"':string=False
        elif c=='"':string=True
        elif c in '[{':
            depth+=1
            if depth>48:raise ValueError('JSON nesting exceeds 48')
        elif c in ']}':depth-=1
    return json.loads(raw,object_pairs_hook=unique,parse_float=reject,parse_constant=reject)


def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='mode',required=True);s.add_parser('demo')
    for name in ('audit','verify'):
        child=s.add_parser(name);child.add_argument('input')
        if name=='verify':child.add_argument('report')
    ns=p.parse_args()
    try:
        if ns.mode=='demo':r=demo()
        else:
            args=decode_input(load(ns.input));r=audit(*args)if ns.mode=='audit'else verify(*args,load(ns.report))
        print(json.dumps(jsonify(r),sort_keys=True,indent=2));return 0
    except (ValueError,TypeError,OSError,RecursionError,UnicodeError)as e:p.error(str(e))


if __name__=='__main__':main()
