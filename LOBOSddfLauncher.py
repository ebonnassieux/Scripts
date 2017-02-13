import glob
import os
import sys
import subprocess
import time
import numpy as np

def AngValtoHHMMSS(AngVal):
   HH=int(AngVal)
   MM=int((AngVal-HH)*60)
   SS=((AngVal-HH-MM/60.)*3600)
   hhmmss="%02i:%02i:%05.2f"%(HH,MM,SS)
   return hhmmss



#MSName="MSlist.sb00_.sb01_.txt"
MSName=np.array([\
                 "MSlist.sb00_.ndppp.txt",\
                 "MSlist.sb01_.ndppp.txt",\
                 "MSlist.sb02_.ndppp.txt",\
                 "MSlist.sb03_.ndppp.txt",\
                 "MSlist.sb04_.ndppp.txt",\
                 "MSlist.sb05_.ndppp.txt",\
                 "MSlist.sb06_.ndppp.txt",\
                 "MSlist.sb07_.ndppp.txt",\
                 "MSlist.sb08_.ndppp.txt",\
                 "MSlist.sb09_.ndppp.txt",\
                 "MSlist.sb10_.ndppp.txt",\
                 "MSlist.sb11_.ndppp.txt",\
                 "MSlist.sb12_.ndppp.txt",\
                 "MSlist.sb13_.ndppp.txt",\
                 "MSlist.sb14_.ndppp.txt",\
                 "MSlist.sb15_.ndppp.txt",\
                 "MSlist.sb16_.ndppp.txt",\
                 "MSlist.sb17_.ndppp.txt",\
                 "MSlist.sb18_.ndppp.txt",\
                 "MSlist.sb19_.ndppp.txt",\
                 "MSlist.sb20_.ndppp.txt",\
                 "MSlist.sb21_.ndppp.txt",\
                 "MSlist.sb22_.ndppp.txt",\
                 "MSlist.sb23_.ndppp.txt",\
                 "MSlist.sb24_.ndppp.txt",\
                 "MSlist.sb25_.ndppp.txt",\
                 "MSlist.sb26_.ndppp.txt",\
                 "MSlist.sb27_.ndppp.txt",\
                 "MSlist.sb28_.ndppp.txt",\
                 "MSlist.sb29_.ndppp.txt",\
                 "MSlist.sb30_.ndppp.txt",\
                 "MSlist.sb31_.ndppp.txt",\
                 "MSlist.sb32_.ndppp.txt",\
                 "MSlist.sb33_.ndppp.txt",\
             ])


parset="LBcal.DDF.parset"

# values taken from http://vo.astron.nl/lobos/lobos/cone/form
RADec=np.array([\
[217.5780125,52.29161],\
[214.9351625,54.3846069444],\
[215.333545833,53.0627769444],\
#[215.2892,51.3756830556],\
[212.959658333,52.817405],\
[212.834283333,52.2011938889],\
[212.02915,52.9198211111],\
[212.040654167,52.6795988889]
])
# convert RA into hours
print RADec.shape
RADec[:,0]=RADec[:,0]/15.

lbcsnum=np.array([\
#                  1,\
                  2,\
                  3,\
#                  4,\
                  5,\
                  6,\
                  7,\
                  8\
])

#RADec=np.array([\
#["14:30:18.72","52:17:29.80"],\
#["14:19:44.44","54:23:04.58"],\
#["14:21:20.05","53:03:46.00"],\
#["14:21:09.41","51:22:32.46"],\
#["14:11:50.32","52:49:02.66"],\
#["14:11:20.23","52:12:04.30"],\
#["14:08:07.00","52:55:11.36"],\
#["14:08:09.76","52:40:46.56"],])

for i in range(RADec.shape[0]):
    for j in range(MSName.shape[0]):
        imname="LOBOS%i.%s.uniform"%(lbcsnum[i],MSName[j])
        logname=imname+".log"
        ss="DDF.py %s --MSName %s --ImageName %s --Mode=Dirty --ChunkHours 10 --PhaseCenterRADEC=[%s,%s] >%s 2>&1"\
            %(parset,MSName[j],imname,AngValtoHHMMSS(RADec[i,0]),AngValtoHHMMSS(RADec[i,1]),logname)
        print ss
        pop=[]
        pop.append(subprocess.Popen(ss,shell=True))
        for p in pop:
            p.wait()

