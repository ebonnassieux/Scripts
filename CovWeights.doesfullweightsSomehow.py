import argparse
import os
from pyrap.tables import table
import numpy as np
import pylab
from numpy import ma
import sys
import warnings
import time
import math

def readArguments():
    parser=argparse.ArgumentParser("Calculate visibility imagin weights based on calibration quality")
    parser.add_argument("-v","--verbose",help="Be verbose, say everything program does. Default is False",required=False,action="store_true")
    parser.add_argument("--filename",type=str,help="Name of the measurement set for which weights want to be calculated",required=True,nargs="+")
    parser.add_argument("--ntsol",type=int,help="Solution interval, in timesteps, for your calibration",required=True)
    parser.add_argument("--lambda",type=int,help="Determines how many neighbours covariance is calculated for: default is 0",default=0,required=False)
    parser.add_argument("--noisemap",help="Save dudv data + covariances in measurement set",required=False,action="store_true")
    args=parser.parse_args()
    return vars(args)


class CovWeights:
    def __init__(self,MSName,ntsol=1,MaxCorrTime=0,SaveDataProducts=True):
        if MSName[-1]=="/":
            self.MSName=MSName[0:-1]
        else:
            self.MSName=MSName
        self.MaxCorrTime=MaxCorrTime
        self.SaveDataProducts=SaveDataProducts
        self.ntSol=ntsol

    def FindWeights(self,tcorr=0):
        #if tcorr!=0:
        #    tcorr=tcorr*self.ntSol
        ms=table(self.MSName,readonly=False,ack=verb)
        # open antennas
        ants=table(ms.getkeyword("ANTENNA"),ack=verb)
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
        #norm=1/np.ones_like(ms.getcol("MODEL_DATA"))
        warnings.filterwarnings("default")
        # get rid of NaN
        #norm[np.isnan(norm)]=0
        if "RESIDUAL_DATA" not in ms.colnames():
            if verb: print "RESIDUAL_DATA not in MS: adding and filling with CORRECTED_DATA"
            desc=ms.getcoldesc("CORRECTED_DATA")
            desc["name"]="RESIDUAL_DATA"
            desc['comment']=desc['comment'].replace(" ","_")
            ms.addcols(desc)
            ms.putcol("RESIDUAL_DATA",ms.getcol("CORRECTED_DATA"))
        # bootes test; true flags are in different dir
        flags=ms.getcol("FLAG")
        #flags=np.load(self.MSName+"/Flagging.npy")
        residualdata=(ms.getcol("RESIDUAL_DATA"))#*norm)
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
        darray=(ms.getcol("CORRECTED_DATA")-ms.getcol("MODEL_DATA")).reshape((nt,nbl,nChan,nPola))
        ms.close()
        rmsarray=np.zeros((nt,nbl,nChan,2),dtype=np.complex64)
        residuals=np.zeros_like(rmsarray,dtype=np.complex64)
        rmsarray[:,:,:,0]=darray[:,:,:,1]
        rmsarray[:,:,:,1]=darray[:,:,:,2]
        # make proper residual array
        residuals[:,:,:,0]=darray[:,:,:,0]#residualdata[:,:,:,0]
        residuals[:,:,:,1]=darray[:,:,:,3]#residualdata[:,:,:,3]
        # antenna coefficient array
        residuals=residuals-np.mean(residuals)
        # ensure things are properly flagged
        CoeffArray=np.zeros((nt,nAnt))        # start calculating the weights
        if verb: print "Begin calculating antenna-based coefficients"
        if tcorr>1:
            ### SAVE DATA ###
            colname="COV_WEIGHT"
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
                if verb: print "%s column already present; will overwrite"%colname
            else:
                W=np.ones((nt*nbl,nchan))
                desc=ms.getcoldesc("WEIGHT_SPECTRUM")
                desc["name"]=colname
                desc['comment']=desc['comment'].replace(" ","_")
                ms.addcols(desc)
                ms.putcol(colname,W)
            # create weight array
            w=np.zeros((nt,nbl,nchan))
            ant1=np.arange(nAnt)
            if verb: print "Fill weights array"
            A0ind=A0[0,:]
            A1ind=A1[0,:]

            ### WEIGHT COL CREATED ###

            warnings.filterwarnings("ignore")
            if verb: print "find covariance between %i nearest times"%tcorr
            for ant1 in A0:
                print ant1,time.time()
                for ant2 in A1:
                    line=0
                    #print "Antenna %i of %i"%(ant,nAnt)
                    for t_i in range(nt):
                        t_lo=max(0,t_i-tcorr-1)
                        t_hi=min(nt,t_i+tcorr)
                        cmat=np.zeros((t_hi-t_lo,t_hi-t_lo))
                        # set of vis for baselines ant-ant_i
                        #ThisBLresiduals=residuals[:,(A0[t_i]==ant)+(A0[t_i]==ant),:,:]
                        set1=np.where(A0[t_i]==ant1)[0]
                        set2=np.where(A1[t_i]==ant2)[0]
                        ThisBLresiduals=np.append(residuals[0,set1,:,:],residuals[0,set2,:,:])
                        # make covmat
                        for iterator1 in range(t_hi-t_lo):
                            for iterator2 in range(t_hi-t_lo):
                                cmat[iterator1,iterator2]= np.mean(residuals[iterator1*self.ntSol,set1,:,:]*residuals[iterator2*self.ntSol,set2,:,:].conj())#*\
#                                                            np.append(residuals[iterator2*self.ntSol,set1,:,:],residuals[iterator2*self.ntSol,set2,:,:]).conj())
                                if iterator1==iterator2:
                                    cmat[iterator1,iterator2]=cmat[iterator1,iterator2]-np.std( (np.append(rmsarray[iterator1,set1,:,:], rmsarray[iterator2,set2,:,:])))
                        # invert covmat
                        invcovmat=np.linalg.inv(cmat)
                        for j in range(nchan):
                            w[t_i,A0ind[i]*A1ind[i],j]=1/np.abs(np.mean(invcovmat[line]))
                        #w[t_i,ant]=np.sqrt(np.abs(np.mean(invcovmat[line])))
                        #print "saving line %i ; "%line,t_hi,t_lo
                        if t_i-tcorr<1:
                            line=line+1
                        elif t_hi==0:
                            line=line-1
                    if verb: PrintProgress(ant2,nAnt)
            warnings.filterwarnings("default")
        else:
            warnings.filterwarnings("ignore")
            if verb: print "Find variance-only weights"
            for t_i in range(nt):
                # build weights for each antenna at time t_i
                for ant in ant1:
                    # set of vis for baselines ant-ant_i
                    if ant==ant1[0]:
                        set1=np.where(A0[t_i]==ant)[0]
                    else:
                        set1=np.where(A0[t_i]==ant1)[0]
                    #set1=np.append(np.where(A0[t_i]==ant),np.where(A1[t_i]==ant1))
                    # set of vis for baselines ant_i-ant
                    if ant==ant1[0]:
                        set2=np.where(A1[t_i]==ant1)[0]
                    else:
                        set2=np.where(A1[t_i]==ant)[0]
                    #set2=np.append(np.where(A1[t_i]==ant),np.where(A0[t_i]==ant1))
#                    CoeffArray[t_i,ant] = np.sqrt(np.mean(np.append(residuals[t_i,set1,:,:],residuals[t_i,set2,:,:])*np.append(residuals[t_i,set1,:,:],residuals[t_i,set2,:,:]).conj())\
#                    CoeffArray[t_i,ant] = np.sqrt( np.std( (np.append(residuals[t_i,set1,:,:],residuals[t_i,set2,:,:])))**2  ) 
                    CoeffArray[t_i,ant] = (np.mean(np.append(residuals[t_i,set1,:,:],residuals[t_i,set2,:,:])*np.append(residuals[t_i,set1,:,:],residuals[t_i,set2,:,:]).conj())-
                                           - np.std( (np.append(rmsarray[t_i,set1,:,:], rmsarray[t_i,set2,:,:]))) )
#                    CoeffArray[t_i,ant] = (np.mean(np.sqrt(np.append(residuals[t_i,set1,:,:],residuals[t_i,set2,:,:])*np.append(residuals[t_i,set1,:,:],residuals[t_i,set2,:,:]).conj())\
#                                           - np.std( (np.append(rmsarray[t_i,set1,:,:], rmsarray[t_i,set2,:,:])))) )
                if verb: PrintProgress(t_i,nt)
        warnings.filterwarnings("default")
#        CoeffArray=CoeffArray-np.mean(CoeffArray)/3.
        for i in range(nAnt):
            thres=0.25*np.median(CoeffArray[:,i])
            CoeffArray[CoeffArray[:,i]<thres,i]=thres
        coeffFilename="%s/CoeffArray.%i.no1stant.npy"%(self.MSName,tcorr)
        if verb: print "Save coefficient array as %s."%coeffFilename
        np.save(coeffFilename,CoeffArray)
        return CoeffArray
                        
    def SaveWeights(self,CoeffArray,colname="VAR_WEIGHT",AverageOverChannels=True,tcorr=0,PreserveResolution=True):
        if verb: print "Begin saving the data"
        ms=table(self.MSName,readonly=False,ack=verb)
        # open antennas
        ants=table(ms.getkeyword("ANTENNA"),ack=verb)
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
            if verb: print "%s column already present; will overwrite"%colname
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
        if verb: print "Fill weights array"
        A0ind=A0[0,:]
        A1ind=A1[0,:]
        
        #if tcorr<2:
        warnings.filterwarnings("ignore")
        for i in range(nbl):
            for j in range(nchan):
                w[:,i,j]=1./(CoeffArray[:,A0ind[i]] + CoeffArray[:,A1ind[i]] + 0.01 + np.sqrt(CoeffArray[:,A0ind[i]]*CoeffArray[:,A1ind[i]]))
                #w[:,i,j]=1./(CoeffArray[:,A0ind[i]]**2 + CoeffArray[:,A1ind[i]]**2 + 0.01)
            if verb: PrintProgress(i,nbl)
        warnings.filterwarnings("default")
        w[np.isnan(w)]=0
        w[np.isinf(w)]=0
        if PreserveResolution==True:
            for i in range(nbl):
                w[:,i,:]=w[:,i,:]/np.mean(w[:,i,:])
        w=w/np.mean(w)
        w=w.reshape(nt*nbl,nchan)
        #else:
        #    #concatenate coeffarray
        #    if self.ntSol>1:
        #        tspill=nt%self.ntSol
        #        nt1=np.int(math.ceil(1.*nt/self.ntSol))
        #        CoeffArray1=np.zeros((nt1,CoeffArray.shape[1]))
        #        for i in range(nt1):
        #            uplim=min((i+1)*self.ntSol,nt)
        #            CoeffArray1[i,:]=np.mean(CoeffArray[i*self.ntSol:uplim,:],axis=0)
        #        # start nbl loop
        #        for i in range(nbl):
        #            mat1=CoeffArray1[:,A0ind[i]].reshape((nt1,1))
        #            mat2=CoeffArray1[:,A1ind[i]].reshape((1,nt1))
        #            cmat=np.dot(mat1,mat2)
        #            stop
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
    args=readArguments()
    msname      = args["filename"]
    corrtime    = args["lambda"]
    ntsol       = args["ntsol"]
    global verb
    verb        = args["verbose"]
    for ms in msname:
        if verb: print "Finding time-covariance weights for: %s"%ms
        covweights=CovWeights(MSName=ms,ntsol=ntsol)
        coefficients=covweights.FindWeights(tcorr=corrtime)
        covweights.SaveWeights(coefficients,colname="VAR_WEIGHT",tcorr=corrtime)
    if verb: print "Total runtime: %f min"%((time.time()-start_time)/60.)
