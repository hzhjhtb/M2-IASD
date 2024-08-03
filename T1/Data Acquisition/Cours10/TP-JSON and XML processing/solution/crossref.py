#!/usr/bin/env python3

import jmespath
import requests
import sys
import urllib.parse
import json

API_URL = 'https://api.crossref.org/journals'
params = { 'query': sys.argv[1], 'rows': 1000 }

r = requests.get(API_URL + '?' + urllib.parse.urlencode(params))
data = r.json()

results = []
for journal in jmespath.search('message.items[*]', data):
    c = jmespath.search('breakdowns."dois-by-issued-year"[*][1]', journal)
    pub_per_year = round(sum(c)/len(c)) if c else 0
    results.append({"title": journal["title"],
                    "pub_per_year": pub_per_year})

print(json.dumps(sorted(results, key=lambda x:x["pub_per_year"])))
