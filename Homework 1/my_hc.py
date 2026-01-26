#!/usr/bin/env python3 -B
# ./rand.py $RANDOM ~/gits/moot/optimize/misc/auto93.csv

import random, sys, xai
xai.the.data=sys.argv[2]
random.seed(int(sys.argv[1]))

data = xai.Data(xai.csv(xai.the.data))

def Y(r): return round(xai.disty(data,r),2)

def report(what,rows):
    a=sorted(rows[:],key=Y)
    print(f":n {len(a):4} :lo {Y(a[0]):5.2f} :mid {Y(a[len(a)//2]):5.2f}", what)

report("baseline",data.rows)
report("sample", xai.shuffle(data.rows)[:30])
