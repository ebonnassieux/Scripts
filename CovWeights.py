import os
from pyrap.tables import table
import numpy as np
import pylab
from numpy import ma
import sys
import warnings
import time


class CovWeights:
    def __init__(self,MSName,ntsol=1,MaxCorrTime=0,SaveDataProducts=0):
        if MSName[-1]=="/":
            self.MSName=MSName[0:-1]
        else:
            self.MSName=MSName
        self.MaxCorrTime=MaxCorrTime
        self.SaveDataProducts=SaveDataProducts
        self.ntsol=ntsol

    def FindWeights(self):
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
        # TODO: find better name
        norm=1/ms.getcol("PREDICTED_VIS")
        warnings.filterwarnings("default")
        # get rid of NaN
        norm[np.isnan(norm)]=0
        if "RESIDUAL_DATA" not in ms.colnames():
            desc=ms.getcoldesc("CORRECTED_DATA")
            desc["name"]="RESIDUAL_DATA"
            desc['comment']=desc['comment'].replace(" ","_")
            ms.addcols(desc)
            ms.putcol("RESIDUAL_DATA",ms.getcol("CORRECTED_DATA"))
        residuals=ms.getcol("RESIDUAL_DATA")
        # bootes test; true flags are in different dir
        flags=ms.getcol("FLAG")#np.load(self.MSName+"/Flagging.npy")
        print "Please ensure that RESIDUAL_DATA or CORRECTED_DATA contains residual visibilities from complete skymodel subtraction, and PREDICTED_VIS contains the uncalibrated flux."
        residuals=(ms.getcol("CORRECTED_DATA")*norm)
        flags=ms.getcol("FLAG")
        # apply flags to data
        residuals[flags==1]=0
        # exit files gracefully
        ms.close()
        ants.close()
        # initialise
        nChan=residuals.shape[1]
        nPola=residuals.shape[2]
        nt=residuals.shape[0]/nbl
        # reshape antennas and data columns
        residuals=residuals.reshape((nt,nbl,nChan,nPola))
        A0=A0.reshape((nt,nbl))
        A1=A1.reshape((nt,nbl))
        ant1=np.arange(nAnt)
        # build antenna coefficient array
        # TODO - MAKE COEFFARRAY INTO COMPLEX-VALUED WEIGHTS
        CoeffArray=np.zeros((nt,nAnt,nChan))#,dtype=np.complex64)

        # start calculating the weights
        print "Begin calculating antenna-based coefficients"
        for t_i in range(nt):
            # build weights for each antenna at time t_i
            for ant in ant1:
                # set of vis for baselines ant-ant_i
                set1=np.where(np.array(A0[t_i])==ant)
                # set of vis for baselines ant_i-ant
                set2=np.where(A1[t_i]==ant)
                for k in range(nChan):
                    CoeffArray[t_i,ant,k]=np.sqrt(np.mean( np.abs(np.append(residuals[t_i,set1,k,:],residuals[t_i,set2,k,:])) ))
            PrintProgress(t_i,nt)
        return CoeffArray
                        
    def SaveWeights(self,CoeffArray,colname="COVWEIGHT"):
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
        A0=ms.getcol("ANTENNA1").reshape((nt,nbl))
        A1=ms.getcol("ANTENNA2").reshape((nt,nbl))
        if colname in ms.colnames():
            print "%s column already present; will overwrite"%colname
        else:
            W=np.ones((nt*nbl,nchan))
            desc=ms.getcoldesc("WEIGHT")
            desc["name"]=colname
            desc['comment']=desc['comment'].replace(" ","_")
            ms.addcols(desc)
            ms.putcol(colname,W)
        # create weight array
        w=np.zeros((nt,nbl,nchan))
        ant1=np.arange(nAnt)
        print "Fill weights array"
        for i in A0.shape[0]:
            w[:,i,:]=CoeffArray[:,A0[i],]
                w[:,indA0A1,:]=1./(CoeffArray[:,i,:]*CoeffArray[:,j,:])
                print w[:,indA0A1,:].shape
                stop
        ants.close()
        ms.close()

### auxiliary functions ###
def PrintProgress(currentIter,maxIter):
    sys.stdout.flush()
    sys.stdout.write("\rProgress: %5.1f %% "%(100*(currentIter+1.)/maxIter))
    if currentIter==(maxIter-1):
        sys.stdout.write("\n")

### if program is called as main ###
if __name__=="__main__":
    start_time=time.time()
    ntsol=1
    msname=sys.argv[1]
    print "Finding time-covariance weights for: %s"%msname
    covweights=CovWeights(MSName=msname)
#    covweights.AddWeightsCol()
    coefficients=covweights.FindWeights()
    covweights.SaveWeights(coefficients)
    print "Total runtime: %f min"%((time.time()-start_time)/60.)
