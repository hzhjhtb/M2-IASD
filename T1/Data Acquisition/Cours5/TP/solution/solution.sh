#!/bin/bash

time cut -d, -f4 <(tail -n +2 owid-covid-data.csv)|awk '{c+=$1} END{print c}'
# 0,033 total

time join -t, -1 1 -2 3 <(tail -n +2 owid-covid-data.csv) <(tail -n +2 countries.csv) | cut -f4,9 -d, | awk -F, '{a[$2]+=$1} END {for (i in a) print a[i]","i}' | sort -rn |head -10
# 0,088 total

time join -t, -1 1 -2 3 <(tail -n +2 owid-covid-data.csv) <(tail -n +2 countries.csv) | cut -f4,1 -d, | awk -F, '{a[$1]+=$2} END {for (i in a) print a[i]","i}' | sort -k2 -t,| join -t, -1 2 -2 1 - <(tail -n +2 population.csv) | awk -F, '$3>0 { print $2/$3*1000 "," $1}'|join -t, -1 2 -2 3 - <(tail -n +2 countries.csv)|cut -d, -f2,3|sort -rn|head
# 0,065 total

time join -t, -1 1 -2 3 <(tail -n +2 owid-covid-data.csv) <(tail -n +2 countries.csv) | cut -f4,1,13 -d, | awk -F, '$3=="Europe" {a[$1]+=$2} END {for (i in a) print a[i]","i}' | sort -k2 -t,| join -t, -1 2 -2 1 - <(tail -n +2 population.csv) | awk -F, '$3>0 { print $2/$3*1000 "," $1}'|join -t, -1 2 -2 3 - <(tail -n +2 countries.csv)|cut -d, -f2,3|sort -rn|head
# 0,077 total
