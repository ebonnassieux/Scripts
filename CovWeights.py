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
        CoeffArray=np.zeros((nt,nAnt))#,dtype=np.complex64)
        # start calculating the weights
        print "Begin calculating antenna-based coefficients"
        if tcorr>1:
            print "find covariance between %i nearest times"%tcorr
            for t_i in range(nt):
                t_lo=max(0,t_i-tcorr)
                t_hi=min(nt,t_i+tcorr)
                # build weights for each antenna at time t_i
                for ant in ant1:
                    # set of vis for baselines ant-ant_i
                    ThisBLresiduals=residuals[:,(A0[t_i]==ant)+(A0[t_i]==ant),:,:]
                    temparray=np.zeros_like(ThisBLresiduals[0])
                    for iter in range(t_lo,t_hi):
                        temparray=temparray+ThisBLresiduals[t_i]*ThisBLresiduals[iter]
                    CoeffArray[t_i,ant]=np.sqrt(np.mean(np.abs(temparray)))
                PrintProgress(t_i,nt)
        else:
            print "Find variance-only weights"
            for t_i in range(nt):
                # build weights for each antenna at time t_i
                for ant in ant1:
                    # set of vis for baselines ant-ant_i
                    set1=np.where(A0[t_i]==ant)[0]
                    # set of vis for baselines ant_i-ant
                    set2=np.where(A1[t_i]==ant)[0]
                    CoeffArray[t_i,ant]=np.sqrt( np.std( np.abs(np.append(residuals[t_i,set1,:,:],residuals[t_i,set2,:,:]))))
                PrintProgress(t_i,nt)


        return CoeffArray
                        
    def SaveWeights(self,CoeffArray,colname="COV_WEIGHT",AverageOverChannels=True,timefrac=0.005):
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

        # clean up small values in coeffarray
        #tpercent=int(nt*timefrac)
        #for i in range(nAnt):
        #    for j in range(nchan):
        #        antvals=CoeffArray[:,i,j]
        #        lowvals=np.sort(antvals[antvals!=0].flatten())[:tpercent]
        #        thres=np.mean(lowvals)
         #       CoeffArray[:,i,j]
         #   PrintProgress(i,nAnt)


        warnings.filterwarnings("ignore")
        #if AverageOverChannels==1:
        # average over channels
        #CoeffArray=np.mean(CoeffArray,axis=2)
        for i in range(nbl):
            for j in range(nchan):
                w[:,i,j]=1./(CoeffArray[:,A0ind[i]]*CoeffArray[:,A1ind[i]])
            PrintProgress(i,nbl)
        #else:
        #    for i in range(nbl):
        #        w[:,i,:]=1./(CoeffArray[:,A0ind[i],:]*CoeffArray[:,A1ind[i],:])
        #    PrintProgress(i,nbl)
        warnings.filterwarnings("default")
        w=w.reshape(nt*nbl,nchan)
        w[np.isnan(w)]=0
        w[np.isinf(w)]=0
        w=w/np.mean(w)
#        lo=np.sort(w[w!=0].flatten())[:100]
        # flag away the most ridiculous near-zero weights
#        w[w>np.std(w)*5]=np.std(w)*5

        # save in weights column
        ms.putcol(colname,w)
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
    coefficients=covweights.FindWeights(tcorr=0)
    covweights.SaveWeights(coefficients,colname="VAR_WEIGHT")
    print "Total runtime: %f min"%((time.time()-start_time)/60.)
