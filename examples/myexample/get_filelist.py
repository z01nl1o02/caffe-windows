import os,sys,pdb
import numpy as np

root="c:/data/"

def create_from_subfolder(indir, outfile):
    lines = []
    for jpg in os.listdir( os.path.join(root, indir) ):
        sname,ext = os.path.splitext(jpg)
        cid = sname[0]
        lines.append( '%s %s'%(os.path.join(indir, jpg), cid))
    with open(outfile,'wb') as f:
        f.write('\r\n'.join(lines))
    return 

create_from_subfolder('train','train.txt')
create_from_subfolder('test','test.txt')

