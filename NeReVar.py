import os
from casacore.tables import table
import numpy as np
import pylab
import sys
import warnings
import time
import argparse

class CovWeights:
    def __init__(self, MSName, dt=0, dfreq=0, SaveDataProducts=1, \
                 uvcut=[0,2000], gainfile=None, phaseonly=True,norm=False, \
                 modelcolname="MODEL_DATA", datacolname="DATA", \
                 weightscolname="IMAGING_WEIGHT"):
        if MSName[-1]=="/":
            self.MSName = MSName[0:-1]
        else:
            self.MSName = MSName
        self.SaveDataProducts = SaveDataProducts
        self.dt               = dt / 2.
        self.dfreq            = int(dfreq / 2)
        self.uvcut            = uvcut
        self.gainfile         = gainfile
        self.phaseonly        = phaseonly
        self.normalise        = norm
        self.ms               = table(self.MSName,readonly=False)
        # open antennas
        self.ants             = table(self.ms.getkeyword("ANTENNA"))
        # open antenna tables
        self.antnames         = self.ants.getcol("NAME")
        self.nAnt             = len(self.antnames)
        # open uvw to perform uvcut
        self.u,self.v,_       = self.ms.getcol("UVW").T
        # load ant indices
        self.A0               = self.ms.getcol("ANTENNA1")
        self.A1               = self.ms.getcol("ANTENNA2")
        # create time array
        self.tarray           = self.ms.getcol("TIME")
        self.t0               = np.min(self.tarray)
        self.tvals            = np.sort(list(set(self.tarray)))
        self.t0               = np.min(self.tvals)
        # set initial time to 0; if MJD needed add self.t0 again
        self.tarray           = self.tarray - self.t0
        self.tvals            = self.tvals  - self.t0
        self.nbl              = np.where(self.tarray==self.tarray[0])[0].size
        self.colnames         = self.ms.colnames()
        self.modelcolname     = modelcolname
        self.datacolname      = datacolname
        self.weightscolname   = weightscolname
        if self.modelcolname in self.colnames and self.datacolname in self.colnames:
            print("Creating RESIDUAL_DATA equivalent from %s - %s"%(self.datacolname,self.modelcolname))
            self.residualdata = ms.getcol(self.datacolname)-ms.getcol(self.datacolname)
        elif "RESIDUAL_DATA" in self.colnames:
            print("Reading RESIDUAL_DATA directly from MS")
            self.residualdata = self.ms.getcol("RESIDUAL_DATA")
        else:
            print("Model, data colnames not present; RESIDUAL_DATA column not in measurement set: reading CORRECTED_DATA")
            self.residualdata = self.ms.getcol("CORRECTED_DATA")
        self.nChan            = self.residualdata.shape[1]
        self.nPola            = self.residualdata.shape[2]
        self.nt               = int(self.residualdata.shape[0]/self.nbl)
        self.flags            = self.ms.getcol("FLAG")
        # apply uvcut
        self.uvlen            = np.sqrt(self.u**2+self.v**2)
        self.flags[self.uvlen>self.uvcut[1]]=1
        self.flags[self.uvlen<self.uvcut[0]]=1
        # apply flags to data
        self.residualdata[self.flags==1]=0

    def FindWeights(self):
        # reshape antennas and data columns
        self.residualdata     = self.residualdata.reshape((self.nt,self.nbl,self.nChan,self.nPola))
        self.flags            = self.flags.reshape((self.nt,self.nbl,self.nChan,self.nPola))
        self.tarray           = self.tarray.reshape((self.nt,self.nbl))
        self.A0               = self.A0.reshape((self.nt,self.nbl))
        self.A1               = self.A1.reshape((self.nt,self.nbl))
        self.ant1             = np.arange(self.nAnt)
        residuals             = np.zeros_like(self.residualdata,dtype=np.complex64)
        # remove crosspols
        residuals[:,:,:,0]    = self.residualdata[:,:,:,0]
        residuals[:,:,:,1]    = self.residualdata[:,:,:,3]
        # antenna coefficient array
        self.CoeffArray       = np.zeros((self.nAnt, self.nt, self.nChan))
        # start calculating the weights
        print("Begin calculating antenna-based coefficients")
        mask = np.zeros_like(residuals).astype(bool)
        for t_i,t_val in enumerate(self.tvals):
            # mask for relevant times within dt
            tmask = ( (t_val+dt  >= self.tvals) * (t_val-dt  <= self.tvals))
            # build weights for each antenna at time t_i
            for ant in self.ant1:
                Resids    = residuals[tmask]
                # build mask for set of vis w/ ant-ant_i and ant_i-ant bls
                antmask   = (self.A0[tmask]==ant) + (self.A1[tmask]==ant)
                AntResids = Resids[antmask]
                AbsResids = np.abs(AntResids)
                # before averaging operation, check if the data is not flagged to save time
                for chan_i in range(self.nChan):
                    chanmin = max(0,chan_i-self.dfreq)
                    vals    = AntResids[:,chanmin:(chan_i+dfreq),:]
                    weights = AbsResids[:,chanmin:(chan_i+dfreq),:]
                    if np.sum(weights) > 0:
                        self.CoeffArray[ant, t_i, chan_i] = np.average( np.real( vals * vals.conj() ), \
                                                                        weights = weights.astype(bool) )
                    else:
                        # if flagged, set var estimates to 0
                        self.CoeffArray[ant, t_i, chan_i] = 0
            PrintProgress(t_i,self.nt)
        for i in range(self.nAnt):
            # flag any potential NaNs
            self.CoeffArray[np.isnan(self.CoeffArray)]=np.inf
            # normalise per antenna if requested
            ### TODO: debug the below as axes have changed
#            if self.normalise==True:
#                self.CoeffArray[:,i,0]=self.CoeffArray[:,i,0]/self.CoeffArray[:,i,1]**2
        # normalise overall to avoid machine errors
        self.CoeffArray = self.CoeffArray /    \
                          np.average( self.CoeffArray, weights = self.CoeffArray.astype(bool))
                if self.weightscolname=="":
            coeffFilename=self.MSName+"/CoeffArray.dt%is.npy"%(dt)
        else:
            coeffFilename=self.MSName+"/CoeffArray.%s.dt%is.npy"%(weightscolname,dt)
        print("Save coefficient array as %s."%coeffFilename)
        np.save(coeffFilename,self.CoeffArray)
    
    def SaveWeights(self):
        print("Begin saving the data")
        if self.weightscolname in self.ms.colnames():
            print("%s column already present; will overwrite"%self.weightscolname)
        else:
            W=np.ones((self.nt*self.nbl,self.nChan,self.nPola))
            desc=self.ms.getcoldesc("WEIGHT_SPECTRUM")
            desc["name"]=self.weightscolname
            desc['comment']=desc['comment'].replace(" ","_")
            self.ms.addcols(desc)
            self.ms.putcol(self.weightscolname,W)
        # create weight array
        w=np.zeros((self.nt,self.nbl,self.nChan,self.nPola))
        ant1=np.arange(self.nAnt)
        print("Fill weights array")
        for i in range(self.nt):
            A0ind = self.A0[i,:]
            A1ind = self.A1[i,:]
            for j in range(self.nbl):
                for k in range(self.nChan):
                    var1=self.CoeffArray[A0ind[j],i,k]
                    var2=self.CoeffArray[A1ind[j],i,k]
                    if var1 and var2 != 0:
                        weights = 1. / ( var1 + var2 )
                    else:
                        weights=0
                    for k1 in range(self.nPola):
                        w[i,j,k,k1] = weights
            PrintProgress(i,self.nt)
        w=w.reshape(self.nt*self.nbl,self.nChan,self.nPola)
#        w[np.isnan(w)]=0
        w = w / np.average(w,weights=w.astype(bool))
        if self.weightscolname!=None:
            self.ms.putcol(self.weightscolname,w)
        else: print("No colname given, so weights not saved in MS.")
#        for i in range(self.nChan):
#            pylab.scatter(np.arange(w.shape[0]),w[:,i,0],s=0.1)
#        pylab.show()

    def close(self):
        # exit files gracefully
        self.ants.close()
        self.ms.close()


def readGainFile(gainfile,ms,nt,nchan,nbl,tarray,nAnt,msname,phaseonly):
    if phaseonly==True or gainfile=="":
        print("Assume amplitude gain values of 1 everywhere")
        ant1gainarray1=np.ones((nt*nbl,nchan))
        ant2gainarray1=np.ones((nt*nbl,nchan))
    else:
        if gainfile[-4:]==".npz":
            print("Assume reading a kMS sols file")
            gainsnpz=np.load(gainfile)
            gains=gainsnpz["Sols"]
            ant1gainarray=np.ones((nt*nbl,nchan))
            ant2gainarray=np.ones((nt*nbl,nchan))
            A0arr=ms.getcol("ANTENNA1")
            A1arr=ms.getcol("ANTENNA2")
            print("Build squared gain array")
            for i in range(len(gains)):
                timemask=(tarray>gains[i][0])*(tarray<gains[i][1])
                for j in range(nAnt):
                    mask1=timemask*(A0arr==j)
                    mask2=timemask*(A1arr==j)
                    for k in range(nchan):
                        ant1gainarray[mask1,:]=np.abs(np.nanmean(gains[i][2][0,j,0]))#np.abs(np.nanmean(gains[i][3][0,j]))
                        ant2gainarray[mask2,:]=np.abs(np.nanmean(gains[i][2][0,j,0]))#np.abs(np.nanmean(gains[i][3][0,j]))
                PrintProgress(i,len(gains))
            np.save(msname+"/ant1gainarray",ant1gainarray)
            np.save(msname+"/ant2gainarray",ant2gainarray)
            ant1gainarray=np.load(msname+"/ant1gainarray.npy")
            ant2gainarray=np.load(msname+"/ant2gainarray.npy")
            #        ant1gainarray1=np.ones((nt,nbl,nchan))
            #        ant2gainarray1=np.ones((nt,nbl,nchan))
            #        for i in range(nchan):
            #            ant1gainarray1[:,:,i]=ant1gainarray**2
            #            ant2gainarray1[:,:,i]=ant2gainarray**2
            ant1gainarray1=ant1gainarray**2#1.reshape((nt*nbl,nchan))
            ant2gainarray1=ant2gainarray**2#1.reshape((nt*nbl,nchan))
            if gainfile[-3:]==".h5":
                print("Assume reading losoto h5parm file")
                import losoto
                solsetName="sol000"
                soltabName="amp000"
                try:
                    gfile=losoto.h5parm.openSoltab(gainfile,solsetName=solsetName,soltabName=soltabName)
                except:
                    print("Could not find amplitude gains in h5parm. Assuming gains of 1 everywhere.")
                    ant1gainarray1=np.ones((nt*nbl,nchan))
                    ant2gainarray1=np.ones((nt*nbl,nchan))
                    return ant1gainarray1,ant2gainarray1
                freqs=table(msname+"/SPECTRAL_WINDOW").getcol("CHAN_FREQ")
                gains=gfile.getValues()[0] # axes: pol, dir, ant, freq, times
                gfreqs=gfile.getValues()[1]["freq"]
                times=fgile.getValues()[1]["time"]
                ant1gainarray=np.zeros((nt*nbl,nchan))
                ant2gainarray=np.zeros((nt*nbl,nchan))
                for j in range(nAnt):
                    mask1=timemask*(A0arr==j)
                    mask2=timemask*(A1arr==j)
                    for k in range(nchan):
                        if freqs[k] in gfreqs:
                            freqmask=(gfreqs==k)
                            ant1gainarray1[mask1,k]=np.mean(gains[:,0,j,freqmask],axis=0)**2
                            ant2gainarray1[mask2,k]=np.mean(gains[:,0,j,freqmask],axis=0)**2
            else:
                print("Gain file type not currently supported. Assume all gain amplitudes are 1.")
                ant1gainarray1=np.ones((nt*nbl,nchan))
                ant2gainarray1=np.ones((nt*nbl,nchan))
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
    parser.add_argument("-v","--verbose",        help="Be verbose, say everything program does. Default is False",required=False,action="store_true")
    parser.add_argument("--filename",  type=str, help="Name of the measurement set for which weights want to be calculated",required=True,nargs="+")
    parser.add_argument("--dt",        type=int, help="Time interval, in seconds, over which to estimate the gain variances. "+\
                        "Default of 0 means an estimate is made for every measurement.",required=False, default=0)
    parser.add_argument("--dfreq",     type=int, help="Frequency interval, in channels, for your calibration. Default of 0, "+\
                        "which solves across all frequency in the dataset.",required=False,default=0)
    parser.add_argument("--weightcol", type=str, help="Name of the weights column name you want to save the weights to. "+\
                        "Default is QUAL_WEIGHT.",required=False,default="QUAL_WEIGHT")
    parser.add_argument("--datacol",   type=str, help="Name of the data column name you want to read to build residual visibilities. "+\
                        "Default is DATA.",required=False,default="DATA")
    parser.add_argument("--modelcol",  type=str, help="Name of the weights column name you want to save the weights to. "+\
                        "Default is MODEL_DATA_CORR.",required=False,default="MODEL_DATA_CORR")
    parser.add_argument("--gainfile",  type=str, help="Name of the gain file you want to read to rebuild the calibration quality weights."+\
                        " If no file is given, equivalent to rebuilding weights for phase-only calibration.",required=False,default="")
    parser.add_argument("--uvcutkm",   type=float,nargs=2,default=[0,3000],required=False,help="uvcut used during calibration, in km.")
    parser.add_argument("--phaseonly",           help="Use if calibration was phase-only; "+\
                        "this means that gain information doesn't need to be read.",required=False,action="store_true")
    parser.add_argument("--normalise",           help="Normalise gains to avoid suppressing long baselines",required=False,action="store_true")
    args=parser.parse_args()
    return vars(args)



### if program is called as main ###
if __name__=="__main__":
    start_time     = time.time()
    args           = readArguments()
    mslist         = args["filename"]
    dt             = args["dt"]
    dfreq          = args["dfreq"]
    weightscolname = args["weightcol"]
    modelcolname   = args["modelcol"]
    datacolname    = args["datacol"]
    gainfile       = args["gainfile"]
    uvcut          = args["uvcutkm"]
    phaseonly      = args["phaseonly"]
    normalise      = args["normalise"]
    for msname in mslist:
        print("Finding time-covariance weights for: %s"%msname)
        covweights=CovWeights(MSName=msname,dt=dt,dfreq=dfreq, gainfile=gainfile,uvcut=uvcut,phaseonly=phaseonly, \
                              norm=normalise, modelcolname=modelcolname, datacolname=datacolname, weightscolname=weightscolname)
        coefficients=covweights.FindWeights()
        covweights.SaveWeights()
        covweights.close()
        print("Total runtime: %f min"%((time.time()-start_time)/60.))
