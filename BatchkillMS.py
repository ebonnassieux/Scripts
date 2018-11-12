import glob
import os
import sys
import subprocess
import getpass

def go(Parset,njobs=5,PaternMS=None,lMS=None,CleanMem=True,AddCols=True): 

   if CleanMem==True:
      pwd=getpass.getpass("Enter sudo password for freemem operation: ")

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
      print "we're in batch %i - %i of %i"%(jbegin+1,jend,len(ll))
      for j in range(jbegin,jend):
#         if AddCols==True:
#            os.system("MSTools.py --ms=%s > %s.mstools.log 2>&1"%(ll[j],ll[j]))
         #ss="sleep 1s; echo done %s in batch %i"%(ll[j],i)
         ss="kMS.py --MSName %s --InCol DATA --OutCol CORRECTED_DATA --SkyModel %s --Decorrelation FT --UVMinMax 0.1,10000 --ApplyCal 0 --SolverType CohJones --dt 0.5 --NChanSols 2 --NIterLM 8 --PolMode IFull --NCPU 6 --BeamModel LOFAR --LOFARBeamMode A --OutSolsName selfcal.pass1 --Decorrelation FT > %s.kMS.log 2>&1"%(ll[j],Parset,ll[j])
         #ss="killMS.py %s --MSName %s  > %s.killms.log 2>&1"%(Parset,ll[j],ll[j])
         print "executing command: %s"%ss
         pop.append(subprocess.Popen(ss,shell=True))
         # impose wait condition
      for p in pop:
         p.wait()
      for j in range(jbegin,jend):
         print "Batch endstate:"
         endstate="tail -n 3 %s.kMS.log"%ll[j]
         os.system(endstate)
#         os.system("/home/cyril.tasse/PipelineCEP/PipelineCEP/clipcal.py --InCol=CORRECTED_DATA --ms=%s > %s.clipcal.log 2>&1"%(ll[j],ll[j]))
      if CleanMem==True:
         print "Cleaning memory"
         os.system("echo %s | sudo -S /cep/lofar/bin/freemem 1; CleanSHM.py"%pwd)


if __name__=="__main__":
    #parset=sys.argv[1]
    skymodel=sys.argv[1]
    # give measurement sets as *.MS last when calling BatchBBS.py
    lMS = sorted(sys.argv[2::])
    print lMS
    go (skymodel,lMS=lMS)
    #ddfcommand="DDF.py  --Data-MS MSlist.6bands.txt --Output-Name 3c295.6bands.uvcut.pass1.QualWeights --Image-Cell 0.1 --Image-NPix 5000 --Image-PhaseCenterRADEC [\"14:11:20.63\",\"+52:12:09\"] --Output-Mode Clean --RIME-DecorrMode FT --Data-ColName=CORRECTED_DATA --Output-RestoringBeam 0.3 --Parallel-NCPU=30 --Selection-UVRangeKm 100,200000 --Weight-ColName=IMAGING_WEIGHT-Deconv-Mode SSD --Deconv-MaxMajorIter 3 --Output-Also OnNeds --Cache-Reset 1 --Mask-External 3c295.sb093_.uvcut.pass2.app.restored.fits.mask.fits --Freq-NBand 6 --Output-Cubes all"
#    print "\nLaunching imaging command:\n\n%s"%ddfcommand
    #os.system(ddfcommand)
