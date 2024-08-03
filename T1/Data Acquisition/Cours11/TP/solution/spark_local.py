#!/usr/bin/env spark-submit

from operator import add
from pyspark import SparkContext
import re
import math
import Stemmer

stop_words=set()
with open("stop_words.txt") as f:
  for line in f:
    stop_words.add(line.rstrip('\r\n'))

sc=SparkContext()
simplewiki=sc.textFile('simple.tsv')

def mapper(key,value):
  it = re.finditer(r"\w+",value,re.UNICODE)
  words=dict()
  stemmer = Stemmer.Stemmer('english')

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
    yield word, (key,count*1./length)

def reducer(key,values,nb_documents):
    result=[]
    idf=math.log(nb_documents*1./len(values))
    for (document, count) in values:
      count*=idf
      result.append((document,count))
    return key, result

nb_documents=simplewiki.count()
simplewiki\
  .map(lambda x: x.split("\t")) \
  .flatMap(lambda l: list(mapper(l[0],l[1]))) \
  .groupByKey() \
  .map(lambda l: reducer(l[0],list(l[1]),nb_documents)) \
  .saveAsTextFile('spark_output')
