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
from numpy.core import multiarray

### main functions ###

class CovWeights:
    def __init__(self,MSName,ntsol=1,nfreqsol=1,MaxCorrTime=0,SaveDataProducts=True,normalise=False,stabilityVariable=0.1,\
                 colname="TEST_WEIGHT1",clipThreshold=0.5,applyWeights=False,SameWeightsForAllPol=True,useGains=False,\
                 solsfile=None):
        self.stabilityVariable=stabilityVariable
        self.ApplyWeights=applyWeights
        self.SameWeightsForAllPol=SameWeightsForAllPol
        self.ClipThreshold=clipThreshold
        if MSName[-1]=="/":
            self.MSName=MSName[0:-1]
        else:
            self.MSName=MSName
        self.MaxCorrTime=MaxCorrTime
        self.solsfile=solsfile
        self.colname=colname
        self.SaveDataProducts=SaveDataProducts
        self.normalise=normalise
        self.nfreqSol=nfreqsol
        #def ReadData(self,tcorr=0):
        self.ms=table(self.MSName,readonly=False,ack=verb)
        self.ants=table(self.ms.getkeyword("ANTENNA"),ack=verb)
        self.weights=np.ones_like(self.ms.getcol("WEIGHT_SPECTRUM"))
        self.tarray=self.ms.getcol("TIME")
        self.nt=len(list(set(self.tarray)))
        self.ntSol=ntsol
        if self.ntSol>self.nt:
            if verb: print "Time solution interval larger than integration time. Setting it to 1 estimate/observation."
            self.nfreqSol=self.nt
        self.tIndices=np.asarray(list(sorted(set(self.tarray))))
        self.dt=self.tIndices[1]-self.tIndices[0]
        # open antenna tables
        self.antnames=self.ants.getcol("NAME")
        self.nAnts=len(self.antnames)
        self.ants.close()
        self.nbl=self.nAnts*(self.nAnts-1)/2
        self.ddescarray=self.ms.getcol("DATA_DESC_ID")
        self.DdescIndices=np.asarray(list(set(self.ddescarray)))
        # load ant indices
        self.A0=self.ms.getcol("ANTENNA1")
        self.A1=self.ms.getcol("ANTENNA2")
        self.antindices=np.array(list(set(np.append(self.A0,self.A1))))
        #nbl=np.where(Times==Times[0])[0].size
        if "RESIDUAL_DATA" not in self.ms.colnames():
            if verb: print "RESIDUAL_DATA not in MS: adding and filling with CORRECTED_DATA - MODEL_DATA"
            desc=self.ms.getcoldesc("CORRECTED_DATA")
            desc["name"]="RESIDUAL_DATA"
            desc['comment']=desc['comment'].replace(" ","_")
            self.ms.addcols(desc)
            try:
                # debug
                self.resid=self.ms.getcol("CORRECTED_DATA")-self.ms.getcol("KMS_MODEL_DATA")
                self.ms.putcol("RESIDUAL_DATA",self.resid)
            except RuntimeError:
                if verb: "Cannot form residual data: please put model visibilities in MODEL_DATA column."
                quit()
        else:
            if verb: print "Reading data from RESIDUAL_DATA column"
            self.resid=self.ms.getcol("RESIDUAL_DATA")
        self.flags=self.ms.getcol("FLAG")
        if self.normalise:
            try: 
                model=self.ms.getcol("MODEL_DATA")
            except RuntimeError:
                if verb: "Cannot load model visibilities to normalise residuals. Please put them in MODEL_DATA column."
                quit()
            warnings.filterwarnings("ignore")
            norm=1.#/model
            warnings.filterwarnings("default")
            norm[np.isnan(norm)]=0
            self.resid=self.resid*norm
            self.resid[np.isnan(self.resid)]=0
        # apply flags to data
        self.resid[self.flags==1]=0
        self.nChan=self.resid.shape[1]
        self.nPola=self.resid.shape[2]
        if self.nfreqSol>self.nChan:
            if verb: print "Frequency solution interval larger than a single subband.We reset it to 1 estimate/subband."
            self.nfreqSol=self.nChan

    def FindIrregWeights(self):
        CoeffArray=np.zeros((self.nt,self.nAnts))
        if verb: print "Begin calculating weights"
        ant1=np.arange(self.nAnts)
        # set up averaging quantities
        if self.nfreqSol>1:
            freqspill=self.nChan%self.nfreqSol
            nfreq1=self.nChan+self.nfreqSol-freqspill
        if self.ntSol>1:
            tspill=self.nt%self.ntSol
            nt1=self.nt+self.ntSol-tspill
        # start weight calculation
        if self.MaxCorrTime==0:
            warnings.filterwarnings("ignore")
            if verb: print "Find variance-only weights"
            prodweights=np.ones_like(self.weights)
            sumweights=np.zeros_like(self.weights)
            for idnum in self.DdescIndices:
                #print "Doing spectral window %i of %i"%(idnum+1,np.max(self.DdescIndices)+1)
                varianceArray=[[] for _ in xrange(len(self.antindices))]#np.empty((len(self.DdescIndices),len(self.antindices)))
                for ant1 in self.antindices:
                    # set up filter
                    arrfilter=(self.ddescarray==idnum)*((self.A0==ant1)+(self.A1==ant1))
                    timefilter=self.tarray[arrfilter] # wait what ?????????????????????????
                    temp=self.resid[arrfilter]
                    #antvar=(temp.ravel()*temp.ravel().conj()).reshape(temp.shape)
                    # TEST TEST TEST
                    temp1=temp.ravel()
                    antvar=np.sqrt((temp1*temp1.conj()).reshape(temp.shape))
                    # average in freq
                    if self.nfreqSol>1:
                        for i in range(nfreq1/self.nfreqSol):
                            upperlim=min((i+1)*self.nfreqSol,self.nChan)
                            freqavg=np.mean(antvar[:,i*self.nfreqSol:upperlim,:],axis=1,dtype=np.float64)
                            for j in range(i*self.nfreqSol,upperlim):
                                antvar[:,j,:]=freqavg
                    # average in time
                    if self.ntSol>1:
                        for i in range(nt1/self.ntSol):
                            sighfilter=np.zeros_like(timefilter).astype(bool)
                            upperlim=min((i+1)*self.ntSol,self.nt)
                            for efwe in range(i*self.ntSol,upperlim):
                                awras=(timefilter==self.tIndices[efwe])
                                sighfilter=sighfilter+awras
                            tavg=np.mean(antvar[sighfilter,:,:],axis=0,dtype=np.float64)
                            antvar[sighfilter,:,:]=tavg[:,:]
                    # average in pol
                    if self.SameWeightsForAllPol==True:
                        polweight=np.mean(antvar,axis=2,dtype=np.float64)
                        for i in range(antvar.shape[2]):
                            antvar[:,:,i]=polweight
                    antvar[np.isnan(antvar)]=0
                    # take root
                    antvar=np.sqrt(antvar)
                    # sigma-clip
                    thres=self.ClipThreshold*np.median(antvar)
                    antvar[antvar<thres]=thres
                    varianceArray[ant1]=antvar
                    prodweights[arrfilter]=prodweights[arrfilter]*antvar
                    sumweights[arrfilter]=sumweights[arrfilter]+antvar
                    if verb: PrintProgress(ant1,np.max(self.antindices)+1,message="Calculating weights for channel %i of %i:"%(idnum+1,np.max(self.DdescIndices)+1),newline=False)
            self.weights=1./np.sqrt(np.abs(prodweights+sumweights+self.stabilityVariable))
            self.weights[sumweights==0]=0
        warnings.filterwarnings("default")
        # apply flags to weights
        if verb: print "Normalising weights per spectral window"
        for idnum in self.DdescIndices:
            arrfilter=(self.ddescarray==idnum)
            norm=np.mean(self.weights[arrfilter][self.weights[arrfilter].astype(np.bool)],dtype=np.float64)
            if np.sum(norm)>0:
                self.weights[arrfilter]=self.weights[arrfilter]/norm
        if self.SaveDataProducts:
            if verb: print "Saving data products in MS"
            np.save(self.MSName+"/VarianceArray.npy",varianceArray)
            np.save(self.MSName+"/weights.npy",self.weights)
        ### do gain stuff

        if verb: print "Saving weights in %s"%self.colname
        # check if weight column in measurement set
        if self.colname not in self.ms.colnames():
            if verb: print "%s not in MS: adding now"%self.colname
            desc=self.ms.getcoldesc("WEIGHT_SPECTRUM")
            desc["name"]=self.colname
            desc['comment']=desc['comment'].replace(" ","_")
            self.ms.addcols(desc)
        try:
            self.ms.putcol(self.colname,self.weights[:,:,0])
        except RuntimeError:
            if verb: print "Current column badly shaped: delete and remake"
            self.ms.removecols(self.colname)
            desc=self.ms.getcoldesc("IMAGING_WEIGHT")
            desc["name"]=self.colname
            desc['comment']=desc['comment'].replace(" ","_")
            self.ms.addcols(desc)
            self.ms.putcol(self.colname,self.weights[:,:,0])
        if verb: "Weights saved to %s"%self.colname
        # apply weights if need be; useful for things like casa which don't read imaging weight columns
        # !!! note that this introduces source suppression as the PSF is not changed accordingly !!!
        if self.ApplyWeights==True:
            try:
                if verb: print "Saving weighted data in WEIGHT_DATA"
                # debug: change this back to corr_data
                self.ms.putcol("WEIGHT_DATA",self.ms.getcol("CORRECTED_DATA")*self.weights)
            except RuntimeError:
                if verb: print "WEIGHT_DATA column missing or ill-shaped; correcting"
                if "WEIGHT_DATA" in self.ms.colnames():
                    self.ms.removecols("WEIGHT_DATA")
                desc=self.ms.getcoldesc("CORRECTED_DATA")
                desc["name"]="WEIGHT_DATA"
                desc['comment']=desc['comment'].replace(" ","_")
                self.ms.addcols(desc)
                # debug: change this back to putting it in WEIGHT_DATA
                self.ms.putcol("WEIGHT_DATA",self.ms.getcol("CORRECTED_DATA")*self.weights)
                if verb: "Saving weighted data in newly-created WEIGHT_DATA"


    def getGainArrays(self,useGains=True,save=True):
        # get the correct shape of the array first
        if useGains==False:
            return 1
        else:
            ### add clause to try and load the saved files before anything else


            # for kMS npz case
            gainsnpz=np.load(self.solsfile)
            gains=gainsnpz["Sols"]
            gainnames=gainsnpz["StationNames"]
            ant1gainarray=np.ones((nt*nbl))
            ant2gainarray=np.ones((nt*nbl))
            A0arr=self.A0
            A1arr=self.A1
            for i in range(len(gains)):
                timemask=(tarray>gains[i][0])*(tarray<gains[i][1])
                for j in range(nAnt):
                    shaap1=np.ones_like(ant1gainarray[timemask][A0arr[timemask]==j])
                    shaap2=np.ones_like(ant2gainarray[timemask][A1arr[timemask]==j])
                    mask1=timemask*(A0arr==j)
                    mask2=timemask*(A1arr==j)
                    ant1gainarray[mask1]=np.abs(np.nanmean(gains[i][3][0,j]))
                    ant2gainarray[mask2]=np.abs(np.nanmean(gains[i][3][0,j]))
                if v: PrintProgress(i,len(gains))
                if save:
                    np.save(self.MSName+"ant1gainarray",ant1gainarray)
                    np.save(self.MSName+"ant2gainarray",ant2gainarray)
                ant1gainarray1=np.ones((nt,nbl,nchan))
                ant2gainarray1=np.ones((nt,nbl,nchan))
                for i in range(nchan):
                    ant1gainarray1[:,:,i]=(ant1gainarray.reshape((nt,nbl)))**2
                    ant2gainarray1[:,:,i]=(ant2gainarray.reshape((nt,nbl)))**2
            return ant1gainarray1,ant2gainarray1
            

                
    def close():
        self.ms.close()
        self.ants.close()


### auxiliary functions ###                                                                                                                                                                  

def readArguments():
    parser=argparse.ArgumentParser("Calculate visibility imagin weights based on calibration quality")
    parser.add_argument("-v","--verbose",help="Be verbose, say everything program does. Default is False",required=False,action="store_true")
    parser.add_argument("--filename",type=str,help="Name of the measurement set for which weights want to be calculated",required=True,nargs="+")
    parser.add_argument("--ntsol",type=int,help="Solution interval, in timesteps, for your calibration",required=True)
    parser.add_argument("--nchansol",type=int,help="Solution interval, in channels, for your calibration",required=True)
#    parser.add_argument("--lambda",type=int,help="Determines how many neighbours covariance is calculated for: default is 0",default=0,required=False)
    parser.add_argument("--save",help="Save dudv data + covariances as numpy arrays in MS",required=False,action="store_true")
    parser.add_argument("--apply",help="Apply weights to the data, saving the result in WEIGHTED_DATA. Default is False.",required=False,action="store_true")
    parser.add_argument("--irregular",help="If stated, do not assume data is regularly-gridded. Use if you have a nasty dataset. Default is False.",\
                        required=False,action="store_true")
    args=parser.parse_args()
    return vars(args)
 
def PrintProgress(currentIter,maxIter,message="Progress:",newline=True):
    sys.stdout.flush()
    sys.stdout.write("\r"+message+" %5.1f %% "%(100*(currentIter+1.)/maxIter))
    if newline:
        if currentIter==(maxIter-1):
            sys.stdout.write("\n")

### if program is called as main ###                                                                                                                                                          
if __name__=="__main__":
    start_time=time.time()
    args        = readArguments()
    msname      = args["filename"]
    irregular   = args["irregular"]
    nchansol    = args["nchansol"]
    ntsol       = args["ntsol"]
    save        = args["save"]
    applythem   = args["apply"]
    global verb
    verb        = args["verbose"]
    for ms in msname:
        if verb: print "Finding time-covariance weights for: %s"%ms
        covweights=CovWeights(MSName=ms,ntsol=ntsol,nfreqsol=nchansol,applyWeights=applythem)
        if irregular:
            covweights.FindIrregWeights()
        else:
            findweights=1
            saveweights=1
        covweights.close
    if verb: print "Total runtime: %f min"%((time.time()-start_time)/60.)
