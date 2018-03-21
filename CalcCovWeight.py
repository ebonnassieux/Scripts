import os
from pyrap.tables import table
import numpy as np
import pylab
from numpy import ma
import sys
import warnings
import time
import math
import argparse

class CovWeights:
    def __init__(self,MSName,ntsol=1,SaveDataProducts=0,modelms=""):
        if MSName[-1]=="/":
            self.MSName=MSName[0:-1]
        else:
            self.MSName=MSName
        self.MaxCorrTime=0
        self.SaveDataProducts=SaveDataProducts
        self.ntSol=ntsol
        self.modelms=modelms

    def FindWeights(self,tcorr=0):
        ms=table(self.MSName)
        # open antennas
        ants=table(ms.getkeyword("ANTENNA"))
        # open antenna tables
        antnames=ants.getcol("NAME")
        nAnt=len(antnames)
        # load ant indices
        A0=ms.getcol("ANTENNA1")
        A1=ms.getcol("ANTENNA2")
        Times=ms.getcol("TIME")
        nbl=np.where(Times==Times[0])[0].size
        warnings.filterwarnings("ignore")
        warnings.filterwarnings("default")
        if modelms!="":
            mdms=table(self.modelms)
            modeldata=mdms.getcol("MODEL_DATA")
            corrdata=ms.getcol("CORRECTED_DATA")
            residualdata=modeldata-corrdata
            mdms.close()
        else:
            try:
                residualdata=ms.getcol("RESIDUAL_DATA")
            except RuntimeError:
                if verb: print "RESIDUAL_DATA not in ms; try building it"
                try: 
                    modeldata=ms.getcol("MODEL_DATA")
                except RuntimeError:
                    if verb: print "MODEL_DATA not in ms; reading CORRECTED_DATA while assuming it's residuals..."
                    residualdata=ms.getcol("CORRECTED_DATA")
        ### if you do want residual data saved, uncomment below ###
        #if "RESIDUAL_DATA" not in ms.colnames():
        #    desc=ms.getcoldesc("CORRECTED_DATA")
        #    desc["name"]="RESIDUAL_DATA"
        #    desc['comment']=desc['comment'].replace(" ","_")
        #    ms.addcols(desc)
        #    ms.putcol("RESIDUAL_DATA",residualdata)
        ### stop uncommenting now ###
        flags=ms.getcol("FLAG")
        residualdata=ms.getcol("RESIDUAL_DATA")        
        flags=ms.getcol("FLAG")
        # apply flags to data
        residualdata[flags==1]=0
        # exit files gracefully
        ants.close()
        # initialise
        nChan=residualdata.shape[1]
        nPola=residualdata.shape[2]
        nt=residualdata.shape[0]/nbl
        # reshape antennas and data columns
        residualdata=residualdata.reshape((nt,nbl,nChan,nPola))
        # average residual data within calibration cells
        if self.ntSol>1:
            tspill=nt%self.ntSol
            nt1=nt+self.ntSol-tspill
            for i in range(nt1/self.ntSol):
                residualdata[i*self.ntSol:(i+1)*self.ntSol,:,:,:]=np.mean(residualdata[i*self.ntSol:(i+1)*self.ntSol,:,:,:],axis=0)
        A0=A0.reshape((nt,nbl))
        A1=A1.reshape((nt,nbl))
        ant1=np.arange(nAnt)
        # make rms array
        darray=ms.getcol("CORRECTED_DATA").reshape((nt,nbl,nChan,nPola))
        ms.close()
        rmsarray=np.zeros((nt,nbl,nChan,2),dtype=np.complex64)
        residuals=np.zeros_like(rmsarray,dtype=np.complex64)
        rmsarray[:,:,:,0]=darray[:,:,:,1]
        rmsarray[:,:,:,1]=darray[:,:,:,2]
        # make proper residual array
        residuals[:,:,:,0]=darray[:,:,:,0]
        residuals[:,:,:,1]=darray[:,:,:,3]
        # antenna coefficient array
        CoeffArray=np.zeros((nt,nAnt))
        # start calculating the weights
        print "Begin calculating antenna-based coefficients"
        warnings.filterwarnings("ignore")
        print "Find variance-only weights"
        for t_i in range(nt):
            # build weights for each antenna at time t_i
            for ant in ant1:
                # set of vis for baselines ant-ant_i
                set1=np.where(A0[t_i]==ant1)[0]
                # set of vis for baselines ant_i-ant
                set2=np.where(A1[t_i]==ant)[0]
                CoeffArray[t_i,ant] = np.sqrt(np.mean(np.append(residuals[t_i,set1,:,:],residuals[t_i,set2,:,:])*np.append(residuals[t_i,set1,:,:],residuals[t_i,set2,:,:]).conj())\
                                              - np.std( (np.append(rmsarray[t_i,set1,:,:], rmsarray[t_i,set2,:,:]))) )
            PrintProgress(t_i,nt)
        warnings.filterwarnings("default")
        for i in range(nAnt):
            thres=0.25*np.median(CoeffArray[:,i])
            CoeffArray[CoeffArray[:,i]<thres,i]=thres
        coeffFilename=self.MSName+"/CoeffArray.npy"
        print "Save coefficient array as %s"%coeffFilename
        np.save(coeffFilename,CoeffArray)
        return CoeffArray
                        
    def SaveWeights(self,CoeffArray,colname="VAR_WEIGHT",AverageOverChannels=True,tcorr=0):
        print "Begin saving the data"
        ms=table(self.MSName,readonly=False)
        # open antennas
        ants=table(ms.getkeyword("ANTENNA"))
        # open antenna tables
        antnames=ants.getcol("NAME")
        nAnt=len(antnames)
        tarray=ms.getcol("TIME")
        darray=ms.getcol("DATA")
        tvalues=np.array(sorted(list(set(tarray))))
        nt=tvalues.shape[0]
        nbl=tarray.shape[0]/nt
        nchan=darray.shape[1]
        A0=np.array(ms.getcol("ANTENNA1").reshape((nt,nbl)))
        A1=np.array(ms.getcol("ANTENNA2").reshape((nt,nbl)))
        if colname in ms.colnames():
            print "%s column already present; will overwrite"%colname
        else:
            W=np.ones((nt*nbl,nchan))
            desc=ms.getcoldesc("IMAGING_WEIGHT")
            desc["name"]=colname
            desc['comment']=desc['comment'].replace(" ","_")
            ms.addcols(desc)
            ms.putcol(colname,W)
        # create weight array
        w=np.zeros((nt,nbl,nchan))
        ant1=np.arange(nAnt)
        print "Fill weights array"
        A0ind=A0[0,:]
        A1ind=A1[0,:]
        warnings.filterwarnings("ignore")
        for i in range(nbl):
            for j in range(nchan):
                w[:,i,j]=1./(CoeffArray[:,A0ind[i]]*CoeffArray[:,A1ind[i]] + 0.01)
            PrintProgress(i,nbl)
        warnings.filterwarnings("default")
        w=w.reshape(nt*nbl,nchan)
        w[np.isnan(w)]=0
        w[np.isinf(w)]=0
        # normalise
        w=w/np.mean(w)
        # save in weights column
        ms.putcol(colname,w)
        ants.close()
        ms.close()

### auxiliary functions ###
def PrintProgress(currentIter,maxIter,msg=""):
    sys.stdout.flush()
    if msg=="":
        msg="Progress:"
    sys.stdout.write("\r%s %5.1f %% "%(msg,100*(currentIter+1.)/maxIter))
    if currentIter==(maxIter-1):
        sys.stdout.write("\n")
def invSVD(A):
    u,s,v=np.linalg.svd(A)
    s[s<1.e-6*s.max()]=1.e-6*s.max()
    ssq=np.abs((1./s))
    # rebuild matrix
    Asq=np.dot(v,np.dot(np.diag(ssq),np.conj(u)))
    v0=v.T*ssq.reshape(1,ssq.size)
    return Asq
def readArguments():
    parser=argparse.ArgumentParser("Calculate visibility imagin weights based on calibration quality")
    parser.add_argument("-v","--verbose",help="Be verbose, say everything program does. Default is False",required=False,action="store_true")
    parser.add_argument("--filename",type=str,help="Name of the measurement set for which weights want to be calculated",required=True,nargs="+")
    parser.add_argument("--ntsol",type=int,help="Solution interval, in timesteps, for your calibration",required=True)
    parser.add_argument("--ModelDataMS",type=str,help="Solution interval, in channels, for your calibration",required=False,default="")
    args=parser.parse_args()
    return vars(args)



### if program is called as main ###
if __name__=="__main__":
    start_time=time.time()
    args        = readArguments()
    msname      = args["filename"]
    ntsol       = args["ntsol"]
    modelms     = args["ModelDataMS"]
    for ms in msname:
        print "Finding time-covariance weights for: %s"%ms
        covweights=CovWeights(MSName=ms,ntsol=ntsol,modelms=modelms)
        coefficients=covweights.FindWeights(tcorr=0)
    print "Total runtime: %f min"%((time.time()-start_time)/60.)
