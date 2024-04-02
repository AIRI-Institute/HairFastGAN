import json
import pprint

import addict


def save_json(path, obj, *args, **kw_args):
    # wrap json.dumps
    with open(path, 'w') as f:
        f.write(json.dumps(obj, *args, **kw_args))


def load_json(path, *args, **kw_args):
    # wrap json.load
    with open(path) as f:
        return addict.Dict(json.load(f, *args, **kw_args))


pp = pprint.pprint
