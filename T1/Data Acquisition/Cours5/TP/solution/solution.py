#!/usr/bin/env python3

import pandas as pd
import time

countries = pd.read_csv('countries.csv', index_col=2)
covid = pd.read_csv('owid-covid-data.csv')
population = pd.read_csv('population.csv', index_col=0)

print(countries)
print(covid)
print(population)

def initialize_time():
    global t
    t = time.time()

def display_time():
    global t
    u = time.time()
    print("%.1f ms"%(1000*(u-t)))
    t = u

initialize_time()
print(covid["new_deaths"].sum())
display_time()
# 0.5 ms

print(pd.concat([countries,covid.groupby('iso_code')['new_deaths'].sum()],axis=1,join='inner').sort_values('new_deaths',ascending=False).head(10))
display_time()
# 8.1 ms

r=pd.concat([countries,population,covid.groupby('iso_code')['new_deaths'].sum()],axis=1,join='inner').sort_values('population',ascending=False)
r['res']=round(1000*r['new_deaths']/r['population'],1)
print(r.sort_values('res',ascending=False)[['name','res']].head(10))
display_time()
# 11.0 ms

europe=countries[countries['region']=='Europe']
r=pd.concat([europe,population,covid.groupby('iso_code')['new_deaths'].sum()],axis=1,join='inner').sort_values('population',ascending=False)
r['res']=round(1000*r['new_deaths']/r['population'],1)
print(r.sort_values('res',ascending=False)[['name','res']].head(10))
display_time()
# 10.6 ms
