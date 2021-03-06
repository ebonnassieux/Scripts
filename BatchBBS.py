import glob
import os
import sys
import subprocess

#def go(Patern=".MS",Parset="bbs.parset",SkyModel="sky.model"):
def go(Parset,SkyModel,njobs=16,PaternMS=None,lMS=None): 

   if PaternMS!=None:
      ll=glob.glob(PaternMS)
   elif lMS!=None:
      ll=lMS
 
   nMS = len(ll)
   jobbounds=range(0,nMS,njobs)
   # if-condition to ensure that stragglers at the end of ll are also accounted for
   if jobbounds[-1]!=(nMS): jobbounds.append(nMS)
     

   for i in range(len(jobbounds)-1): # i.e. every njobs
      # create array on which to append subprocesses for the wait() command
      pop=[]
      # define bounds within which we'll be working this iteration
      jbegin=jobbounds[i]
      jend=jobbounds[i+1]
      print "we're in batch %i - %i of %i"%(jbegin,jend,len(ll))

      for j in range(jbegin,jend):
#         ss="sleep 1s; echo done %s in batch %i"%(ll[j],i)
         ss="calibrate-stand-alone -f %s %s %s  > /dev/null 2>&1"%(ll[j],Parset,SkyModel)
         pop.append(subprocess.Popen(ss,shell=True))

         # impose wait condition
      for p in pop:
         p.wait()


if __name__=="__main__":
    parset, skymodel=sys.argv[1:3]
    # give measurement sets as *.MS last when calling BatchBBS.py
    lMS = sorted(sys.argv[3::])
    go (parset, skymodel,lMS=lMS)
