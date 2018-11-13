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
    def __init__(self,MSName,ntsol=1,SaveDataProducts=0,uvcut=[0,2000],gainfile=None):
        if MSName[-1]=="/":
            self.MSName=MSName[0:-1]
        else:
            self.MSName=MSName
        self.MaxCorrTime=0
        self.SaveDataProducts=SaveDataProducts
        self.ntSol=ntsol
        self.uvcut=uvcut
        self.gainfile=gainfile
        
    def FindWeights(self,tcorr=0,colname=""):
        ms=table(self.MSName,readonly=False)
        # open antennas
        ants=table(ms.getkeyword("ANTENNA"))
        # open antenna tables
        antnames=ants.getcol("NAME")
        nAnt=len(antnames)
        # open uvw to perform uvcut - TEST
        u,v,_=ms.getcol("UVW").T
        # load ant indices
        A0=ms.getcol("ANTENNA1")
        A1=ms.getcol("ANTENNA2")
        Times=ms.getcol("TIME")
        nbl=np.where(Times==Times[0])[0].size
        warnings.filterwarnings("ignore")
        warnings.filterwarnings("default")
        if "RESIDUAL_DATA" not in ms.colnames():
            print "RESIDUAL_DATA column not in measurement set: creating it and filling with RESIDUAL_DATA"
            desc=ms.getcoldesc("CORRECTED_DATA")
            desc["name"]="RESIDUAL_DATA"
            desc['comment']=desc['comment'].replace(" ","_")
            ms.addcols(desc)
            ms.putcol("RESIDUAL_DATA","CORRECTED_DATA")
        flags=ms.getcol("FLAG")
        print "Please ensure that RESIDUAL_DATA or CORRECTED_DATA contains residual visibilities from complete skymodel subtraction."
        residualdata=ms.getcol("RESIDUAL_DATA")
        flags=ms.getcol("FLAG")
        # apply uvcut
        uvlen=np.sqrt(u**2+v**2)
        flags[uvlen>self.uvcut[1]]=1
        flags[uvlen<self.uvcut[0]]=1
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
        ### TODO: the averaging should be done later; try using a filter instead of this ###
        if self.ntSol>1:
            tspill=nt%self.ntSol
            nt1=nt+self.ntSol-tspill
            for i in range(nt1/self.ntSol):
                residualdata[i*self.ntSol:(i+1)*self.ntSol,:,:,:]=np.mean(residualdata[i*self.ntSol:(i+1)*self.ntSol,:,:,:],axis=0)
        A0=A0.reshape((nt,nbl))
        A1=A1.reshape((nt,nbl))
        ant1=np.arange(nAnt)
        # make rms array
        darray=ms.getcol("RESIDUAL_DATA").reshape((nt,nbl,nChan,nPola))
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
            # get rid of NaN
            CoeffArray[np.isnan(CoeffArray)]=np.inf
            tempars=CoeffArray[:,i]
            thres=0.25*np.median(tempars)
            CoeffArray[:,i][tempars<thres]=thres
        if colname=="":
            coeffFilename=self.MSName+"/CoeffArray.ntsol%i.npy"%(ntsol)
        else:
            coeffFilename=self.MSName+"/CoeffArray.%s.ntsol%i.npy"%(colname,ntsol)
        print "Save coefficient array as %s."%coeffFilename
        np.save(coeffFilename,CoeffArray)
        return CoeffArray
                        
    def SaveWeights(self,CoeffArray,colname=None,AverageOverChannels=True,tcorr=0):
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

        # do gains stuff
        ant1gainarray1,ant2gainarray1=self.ReadGainFile()
        for i in range(nbl):
            for j in range(nchan):
                w[:,i,j]=1./(CoeffArray[:,A0ind[i]]*ant2gainarray[i]+CoeffArray[:,A1ind[i]]*ant1gainarray[i]+CoeffArray[:,A0ind[i]]*CoeffArray[:,A1ind[i]] + 0.1)
            PrintProgress(i,nbl)
        warnings.filterwarnings("default")
        w=w.reshape(nt*nbl,nchan)
        w[np.isnan(w)]=0
        w[np.isinf(w)]=0
        # normalise
        w=w/np.mean(w)
        # save in weights column
        if colname!=None:
            ms.putcol(colname,w)
        ants.close()
        ms.close()

    def readGainfile(self):
        if self.gainfile[-4:]==".npy":
            print "Assume reading a kMS sols file"
            gainsnpz=np.load(self.gainfile)
	    gains=gainsnpz["Sols"]
            #gainnames=gainsnpz["StationNames"]
            ant1gainarray=np.ones((nt*nbl))
            ant2gainarray=np.ones((nt*nbl))
            A0arr=ms.getcol("ANTENNA1")
            A1arr=ms.getcol("ANTENNA2")
            print "Build squared gain array"
            for i in range(len(gains)):
                timemask=(tarray>gains[i][0])*(tarray<gains[i][1])
                for j in range(nAnt):
                    shaap1=np.ones_like(ant1gainarray[timemask][A0arr[timemask]==j])
                    shaap2=np.ones_like(ant2gainarray[timemask][A1arr[timemask]==j])
                    mask1=timemask*(A0arr==j)
                    mask2=timemask*(A1arr==j)
                    ant1gainarray[mask1]=np.abs(np.nanmean(gains[i][3][0,j]))
                    ant2gainarray[mask2]=np.abs(np.nanmean(gains[i][3][0,j]))
                PrintProgress(i,len(gains))
            if self.SaveDataProducts:
                np.save("ant1gainarray",ant1gainarray)
                np.save("ant2gainarray",ant2gainarray)
            ant1gainarray=np.load("ant1gainarray.npy")
            ant2gainarray=np.load("ant2gainarray.npy")
            ant1gainarray1=np.ones((nt,nbl,nchan))
            ant2gainarray1=np.ones((nt,nbl,nchan))
            for i in range(nchan):
                ant1gainarray1[:,:,i]=(ant1gainarray.reshape((nt,nbl)))**2
                ant2gainarray1[:,:,i]=(ant2gainarray.reshape((nt,nbl)))**2
        if self.gainfile[-3:]==".h5":
            print "Assume reading losoto h5parm file"
            

        else:
            print "Please convert gain file to either kMS numpy array or LoSoTo h5 format, as these are only ones currently supported."
            ant1gainarray1=np.ones((nt*nbl,nchan))
            ant2gainarray=np.ones_like(ant1gainarray1)
        return ant1gainarray1,ant2gainarray1

        
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
    parser.add_argument("--colname",type=str,help="Name of the weights column name you want to save the weights to. Default is CAL_WEIGHT.",required=False,default="CAL_WEIGHT")
    parser.add_argument("--gainfile",type="str",help="Name of the gain file you want to read to rebuild the calibration quality weights."+\
                        " If no file is given, equivalent to rebuilding weights for phase-only calibration.",required=False,default=True)
    parser.add_argument("--uvcutkm",type=float,nargs=2,default=0,2000,required=False,help="uvcut used during calibration, in km.")
#    parser.add_argument("--nchansol",type=int,help="Solution interval, in channels, for your calibration",required=True)
    args=parser.parse_args()
    return vars(args)



### if program is called as main ###
if __name__=="__main__":
    start_time=time.time()
    args        = readArguments()
    mslist      = args["filename"]
    ntsol       = args["ntsol"]
    colname     = args["colname"]
    gainfile    = args["gainfile"]
    uvcut       = args["uvcut"]
    for msname in mslist:
        print "Finding time-covariance weights for: %s"%msname
        coefficients=covweights.FindWeights(tcorr=0,colname=colname,gainfile=gainfile,uvcut=uvcut)
        covweights=CovWeights(MSName=msname,ntsol=ntsol)
        coefficients=1
        covweights.SaveWeights(coefficients,colname=colname,AverageOverChannels=True,tcorr=0)
        print "Total runtime: %f min"%((time.time()-start_time)/60.)
