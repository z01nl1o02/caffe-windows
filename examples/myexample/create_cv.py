import os,sys,pdb
import numpy as np
from collections import defaultdict
import random

inputfile='all.txt'
N = 2

class2file = defaultdict(list)
with open(inputfile,'rb') as f:
    for line in f:
        line = line.strip()
        if line == "":
            continue
        relpath = '\\'.join( line.split('\\')[-2:])
        classid = line.split('\\')[-2]
        class2file[classid].append(relpath)

cv = defaultdict(list)
for cid in class2file.keys():
    random.shuffle( class2file[cid] )
    for k, relpath in enumerate(class2file[cid]):
        cvn = k % N
        cv[cvn].append( '%s %s'%(relpath,cid) )

for n in range(N):
    testlines = cv[n]
    trainlines = []
    for k in range(N):
        if k == n:
            continue
        trainlines.extend( cv[k] )
    with open('train_%d.txt'%n, 'wb') as f:
        f.write( '\r\n'.join(trainlines))
    with open('test_%d.txt'%n,'wb') as f:
        f.write( '\r\n'.join(testlines))

