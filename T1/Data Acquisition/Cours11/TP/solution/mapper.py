#!/usr/bin/env python3

import re
import sys
import Stemmer

stemmer = Stemmer.Stemmer('english')

stop_words=set()
with open("stop_words.txt") as f:
    for line in f:
        stop_words.add(line.rstrip('\r\n'))

for line in sys.stdin:
    key, value = line.split("\t")
    it = re.finditer(r"\w+",value,re.UNICODE)
    words=dict()
    length=0
    for match in it:
        token=match.group().lower()
        if not(token in stop_words):
            length=length+1
            token=stemmer.stemWord(token)
            if token in words:
                words[token]+=1
            else:
                words[token]=1
    for word, count in words.items():
        print("%s\t%s:%f"%(word,key,count*1./length))
