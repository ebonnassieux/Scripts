import numpy as np
import argparse
import getpass
import glob
import os
import sys
import subprocess
from pyrap.tables import table
import bdsf
import multiprocessing

def readArguments():
    # function for user-friendly scripting
    parser=argparse.ArgumentParser("Automate one round of self-cal")
    parser.add_argument("--filename",type=str,help="Name of MS files",required=True,nargs="+")
    parser.add_argument("--basename",help="Base name of selfcal operation, e.g. 3C295.selfcal1.uvcut",required=True)
    parser.add_argument("--skymodel",help="killMS sky model (must be .npy or BBS format)",required=True)
    parser.add_argument("--applytype",type=str,help="How to apply gains. A for amplitude-only, P for phase-only, AP for ampphase. Default is P",required=False,default="p")
    parser.add_argument("--uvcutkm",type=float,help="uvcut params for selfcal. Default is an inner uvcut of 100m, no outer uvcut.",required=False,nargs=2,default=[0.1,200000])
    parser.add_argument("--itermin",type=int,help="Initial loop, useful if you finished a round of selfcal and want to perform one further. Default is 0",required=False,default=0)
    parser.add_argument("--itermax",type=int,help="Maximum number of selfcal loops until we consider to have converged. Default is 10",required=False,default=10)
    parser.add_argument("--ncals",help="Number of simultaneous calibration operations going at once. Fine tuning parameter. Default is 6",required=False,default=10)
    parser.add_argument("--mask",help="Clean mask to use, requires fits file. Optional, as default is to use automasking. Default is None",required=False,default=None)
    parser.add_argument("--NCPU",type=int,help="Max number of CPUs this program can use. Default is 0, assumes direct control & uses every core available.",required=False,default=0)
    args=parser.parse_args()
    return vars(args)

def calib(Parset,njobs=6,PaternMS=None,lMS=None,CleanMem=True,AddCols=True,itername="",applytype="P",ncpu=10):
    # this does calibration & associated things
    if itername=="": itername=Parset
    print "Begin calibration"
    if PaternMS!=None:
       ll=glob.glob(PaternMS)
    elif lMS!=None:
       ll=lMS
    nMS = len(ll)
    jobbounds=range(0,nMS,njobs)
    ncpuperjob=int(ncpu/min(njobs,nMS))
    if jobbounds[-1]!=(nMS): jobbounds.append(nMS)
    for i in range(len(jobbounds)-1):
       pop=[]
       jbegin=jobbounds[i]
       jend=jobbounds[i+1]
       for j in range(jbegin,jend):
           print "we're in batch %i - %i of %i"%(jbegin+1,jend,len(ll))
           ss="kMS.py --MSName %s --SkyModel %s "%(ll[j],Parset)+\
               " --InCol DATA --OutCol CORRECTED_DATA --NCPU %i --OutSolsName %s --ApplyToDir 0"%(ncpuperjob,itername)+\
               " --SolverType CohJones --PolMode IDiag --dt .25 --NChanSols 1 --NIterLM 8 --ApplyMode %s > %s.kMS.log 2>&1"%(applytype,ll[j])
           print "executing command: %s"%ss
           pop.append(subprocess.Popen(ss,shell=True))
       for p in pop:
           p.wait()
       for j in range(jbegin,jend):
           print "Batch endstate:"
           endstate="tail -n 3 %s.kMS.log"%ll[j]
           os.system(endstate)
       if CleanMem==True:
           print "Cleaning memory"
           os.system("echo %s | sudo -S /cep/lofar/bin/freemem 1; CleanSHM.py"%pwd)

def imagingbit(filenames,uvcut,itername,maskname,Clip=True,ncpu=10):
    # this does the imaging & associated things
    mslist="MSlist."+itername+".txt"
    print "Write MS list into %s"%mslist
    f=open(mslist,"w")
    for line in filenames:
        f.write(line+"\n")
    f.close()
    DDFstring="DDF.py --Data-MS %s --Output-Name IMAGES/%s --Image-Cell 150 --Image-NPix 256 --Mask-SigTh 8 "%(mslist,itername)+\
        " --Output-Mode Clean --Data-ColName CORRECTED_DATA  --Parallel-NCPU=%i --Deconv-Mode SSD --Deconv-MaxMajorIter 4 --Output-Also all"%ncpu+\
        " --GAClean-ScalesInitHMP [0,1,2,4] --Cache-Reset 1  --Misc-ConserveMemory 1 --Facets-NFacets 1 --Mask-Auto 1 --Weight-ColName None"+\
        " > %s.ddf.log 2>&1 "%("IMAGES/"+mslist+itername)
    print DDFstring
    os.system(DDFstring)
    os.system("tail %s.ddf.log"%("IMAGES/"+mslist+itername))
    print "Cleaning memory"
    os.system("echo %s | sudo -S /cep/lofar/bin/freemem 1; CleanSHM.py"%pwd)

def selfcalLoop(mslist,uvcut,skymodel,itername,maskname,njobs=6,applytype="P",ncpu=0):
    calib(skymodel,njobs=njobs,lMS=mslist,itername=itername,applytype=applytype,ncpu=ncpu)
    imagingbit(mslist,uvcut,itername,maskname,Clip=True,ncpu=ncpu)
    
if __name__=="__main__":
    # bit executed when function called from terminal
    args      = readArguments()
    filenames = args["filename"]
    name      = args["basename"]
    skymodel  = args["skymodel"]
    maskname  = args["mask"]
    uvcut     = args["uvcutkm"]
    itermin   = args["itermin"]
    itermax   = args["itermax"]
    ncals     = args["ncals"]
    applytype = args["applytype"]
    ncpu      = args["NCPU"]
    # get password for memory clean operations
    global pwd
    pwd=getpass.getpass("Please input your password to perform memroy cleaning operations: ")
    print "cheers m8"
    # determine number of cpus on node if ncpu=0
    if ncpu==0:
        ncpu=multiprocessing.cpu_count()
        # alternate bash command: cat /proc/cpuinfo | grep processor | wc -l    
    for i in range(itermin,itermax):
        print "in iteration: ",i+1
        loopname=name+".pass%i"%(i+1)
        if i!=0:
            makemodelstring="MakeModel.py --BaseImageName %s.pass%i --NCluster 1 > IMAGES/%s.pass%i.makemodel.log 2>&1"%("IMAGES/"+name,i,name,i)
            print makemodelstring
            os.system(makemodelstring)
            skymodel="IMAGES/"+name+".pass%i"%(i)+".npy"
        selfcalLoop(filenames,uvcut,skymodel,loopname,maskname,njobs=ncals,applytype=applytype,ncpu=ncpu)

