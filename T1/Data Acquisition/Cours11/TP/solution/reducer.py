#!/usr/bin/env python3

import math
import sys

nb_documents = None
with open("nb_documents.txt") as f:
    nb_documents = int(f.readline())

values = []
current_key = None


def reduce(key, values):
    print(key + "\t", end='')
    idf = math.log(nb_documents*1./len(values))
    result = []
    for value in values:
        document, count = value.split(":")
        count = float(count)
        count *= idf
        result.append([document, count])

    result.sort(key=lambda x: x[1], reverse=True)
    for r in result:
        print("%s:%s:" % (r[0], r[1]), end='')
    print()


for line in sys.stdin:
    key, value = line.split("\t")

    if key != current_key:
        if current_key is not None:
            reduce(current_key, values)
        current_key = key
        values = []

    values.append(value)

reduce(current_key, values)
