import os
import sys
import argparse
import numpy as np
from pyrap.tables import table

def readArguments():
    # create parser
    parser=argparse.ArgumentParser(description="Make an irregular MS data structure regular.")
    parser.add_argument("--filename",type=str,help="Name of the Measurement Set you want to regularise",required=True)
    parser.add_argument("--outname",type=str,help="Name of the output Measurement Set, optional",required=False)
    parser.add_argument("-v","--verbose",help="Be verbose, say everything program does. Default is True",required=False,action="store_true")
    # parse
    args=parser.parse_args()
    return vars(args)

def blockPrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__

def RegulariseMS(MSname,outname=None,verbose=False):
    ms=table(MSname,readonly=False,ack=verbose)
    freq=table(MSname+"/DATA_DESCRIPTION")
    print freq.colnames()

    test=ms.getcol("STATE_ID")

    A0=ms.getcol("ANTENNA1")
    A1=ms.getcol("ANTENNA2")
    lastant1=max(A0)
    nbl=lastant1*(lastant1+1)/2
    print nbl
    datavec=[]
    timevec=[]
    flagvec=[]
    d=ms.getcol("DATA")
    t=ms.getcol("TIME")
    f=ms.getcol("FLAG")
    # find total set of timesteps in MS
    timesteps=np.array(list(set(t)))
    timeset=set()
#    blockPrint()
    for ant1 in set(A0):
        for ant2 in range(ant1+1,lastant1+1):
            indA0A1=np.where(((A0==ant1)&(A1==ant2))|((A0==ant2)&(A1==ant1)))[0]
            timesteps=t[indA0A1]
#            for i in range(16):
#                print timesteps[i],test[indA0A1][i]
#            stop
            timeset=timeset|set(timesteps)
#    enablePrint()
    times=np.array(list(timeset))
    timesteps=times.shape
    regulartimes=np.tile(times,nbl*8)
    print regulartimes.shape
    print t.shape
    print len(list(set(t)))*nbl*8

    ms.close()




if __name__=="__main__":
    # parse arguments
    args=readArguments()
    # assign variables
    msname=args["filename"]
    outname=args["outname"]
    verbos=args["verbose"]
    # launch script
    RegulariseMS(MSname=msname,outname=outname,verbose=verbos)
