#!/usr/bin/env python3

import json
import sys

with open(sys.argv[1], "r") as inp:
    data = json.load(inp)

    print("digraph {")
    for d in data:
        if 'from' in d:
            print(f'  n{d["from"]} -> n{d["to"]};')
        else:
            print(f'  n{d["id"]} [URL="https://www.mathgenealogy.org/id.php?id={d["id"]}", label="{d["person"]}\\n{d["year"]}"];')
    print("}")
