#!/bin/bash
pv simplewiki-*.xml.bz2|  ./extract_wiki.py  > simple.tsv
hdfs dfs -put simple.tsv
cat simple.tsv | wc -l > nb_documents.txt
pv simple.tsv | ./mapper.py | LC_ALL=C sort -t'     '  | ./reducer.py > reduced
hadoop-streaming -files mapper.py,reducer.py,stop_words.txt,nb_documents.txt -mapper mapper.py -reducer reducer.py -input simple.tsv -output output
./spark.py
