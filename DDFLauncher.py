import glob
import os
import sys
import subprocess
import time

for i in [1,2,3,5,6,7,8,10,12,13]:
    ss="DDF.py 3c295.sb211.test.parset --MSName MSlist.sb%02i_.txt --ImageName groth.sb%02i_.largeHR --Mode=Dirty --ColName=DATA --NFacets=20 --NCPU=30 --Npix=40000"%(i,i)
    print ss
    pop=[]
    pop.append(subprocess.Popen(ss,shell=True))
    for p in pop:
        p.wait()

    
