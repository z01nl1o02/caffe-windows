import os,sys,pdb
import numpy as np
from collections import defaultdict
import random

inputfile='all.txt'
ratio = 0.8

class2file = defaultdict(list)
with open(inputfile,'rb') as f:
    for line in f:
        line = line.strip()
        if line == "":
            continue
        relpath = '\\'.join( line.split('\\')[-2:])
        classid = line.split('\\')[-2]
        class2file[classid].append(relpath)

trainlist = []
testlist = []
for cid in class2file.keys():
    random.shuffle( class2file[cid] )
    trainNum = np.int64(len(class2file[cid]) * ratio )
    trainlist.extend( class2file[cid][0:trainNum])
    testlist.extend ( class2file[cid][trainNum:] )

random.shuffle(trainlist)
random.shuffle(testlist)

with open('train.txt', 'wb') as f:
    f.write( '\r\n'.join(trainlist))
with open('test.txt','wb') as f:
    f.write( '\r\n'.join(testlist))

