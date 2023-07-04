import os
from casacore.tables import table
import numpy as np
import pylab
from numpy import ma
import sys
import warnings
import time
import math
import argparse

class CovWeights:
    def __init__(self,MSName,ntsol=1,nfreqsol=1,SaveDataProducts=1,uvcut=[0,2000],gainfile=None,phaseonly=True,norm=False, \
                 modelcolname="MODEL_DATA",datacolname="DATA",weightscolname="IMAGING_WEIGHT"):
        if MSName[-1]=="/":
            self.MSName = MSName[0:-1]
        else:
            self.MSName = MSName
        self.SaveDataProducts = SaveDataProducts
        self.ntSol            = ntsol
        self.nfreqsol         = nfreqsol
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
        # open uvw to perform uvcut - TEST
        self.u,self.v,_       = self.ms.getcol("UVW").T
        # load ant indices
        self.A0               = self.ms.getcol("ANTENNA1")
        self.A1               = self.ms.getcol("ANTENNA2")
        self.tarray           = self.ms.getcol("TIME")
        self.nbl              = np.where(self.tarray==self.tarray[0])[0].size
        self.colnames         = self.ms.colnames()
        self.modelcolname     = modelcolname
        self.datacolname      = datacolname
        self.weightscolname   = weightscolname
        if self.modelcolname in self.colnames and self.datacolname in self.colnames:
            print("Creating RESIDUAL_DATA equivalent from %s - %s"%(self.datacolname,self.modelcolname))
            self.residualdata=ms.getcol(self.datacolname)-ms.getcol(self.datacolname)
        elif "RESIDUAL_DATA" in self.colnames:
            print("Reading RESIDUAL_DATA directly from MS")
            self.residualdata=self.ms.getcol("RESIDUAL_DATA")
        else:
            print("Model, data colnames not present; RESIDUAL_DATA column not in measurement set: reading CORRECTED_DATA")
            self.residualdata=self.ms.getcol("CORRECTED_DATA")
        self.nChan=self.residualdata.shape[1]
        self.nPola=self.residualdata.shape[2]
        self.nt=int(self.residualdata.shape[0]/self.nbl)
        self.flags=self.ms.getcol("FLAG")
        # apply uvcut
        self.uvlen=np.sqrt(self.u**2+self.v**2)
        self.flags[self.uvlen>self.uvcut[1]]=1
        self.flags[self.uvlen<self.uvcut[0]]=1
        # apply flags to data
        self.residualdata[self.flags==1]=0
        # exit files gracefully
        self.ants.close()
        self.ms.close()

    def FindWeights(self):
        warnings.filterwarnings("ignore")
        warnings.filterwarnings("default")
        # reshape antennas and data columns
        self.residualdata=self.residualdata.reshape((self.nt,self.nbl,self.nChan,self.nPola))
        # average residual data within calibration cells
        ### TODO: the averaging should be done later; try using a filter instead of this ###
        self.A0=self.A0.reshape((self.nt,self.nbl))
        self.A1=self.A1.reshape((self.nt,self.nbl))
        self.ant1=np.arange(self.nAnt)
        
#        ant1=np.arange(nAnt)
#        CoeffArray=np.zeros((nt,nAnt))
#        print("Begin calculating antenna-based coefficients")
#        for i,t_i in enumerate(times):
#            indexmax=min(len(times)-1,i+self.ntSol)
#            indexmin=max(0,i-self.ntSol)
#            tmin=times[indexmin]
#            tmax=times[indexmax]
#            tmask=(tarray>tmin)*(tarray<tmax)
#            for ant in ant1:
#                # set of vis for baselines ant-ant_i
#                set1=(A0==ant)
#                # set of vis for baselines ant_i-ant
#                set2=(A1==ant)
#                resmask=tmask*(set1+set2)
#                rarray=residualdata[resmask]
#                CoeffArray[i,ant] = np.mean(np.abs(rarray*rarray.conj()))
#            PrintProgress(i,nt)


### TODO figure out if below is actually useful. Bugged at present
#        if self.ntSol>1:
#            tspill=nt%self.ntSol
#            nt1=nt+self.ntSol-tspill
#            for i in range(int(nt1/self.ntSol)):
#                for j in range(self.nfreqsol):
#                    print(residualdata[i*self.ntSol:(i+1)*self.ntSol,:,self.nfreqsol*j:(j+1)*nfreqsol,:].shape)
#                    print(np.mean(residualdata[i*self.ntSol:(i+1)*self.ntSol,:,:,:],axis=0).shape)
#                    residualdata[i*self.ntSol:(i+1)*self.ntSol,:,self.nfreqsol*j:(j+1)*nfreqsol,:]=np.mean(residualdata[i*self.ntSol:(i+1)*self.ntSol,:,:,:],axis=0)

        residuals=np.zeros_like(self.residualdata,dtype=np.complex64)
        # remove crosspols
        residuals[:,:,:,0]=self.residualdata[:,:,:,0]
        residuals[:,:,:,1]=self.residualdata[:,:,:,3]
        # antenna coefficient array
        self.CoeffArray=np.zeros((self.nt,self.nAnt,2))
        # start calculating the weights
        print("Begin calculating antenna-based coefficients")
        for t_i in range(self.nt):
            # build weights for each antenna at time t_i
            for ant in self.ant1:
                # set of vis for baselines ant-ant_i
                set1=np.where(self.A0[t_i]==ant)[0]
                # set of vis for baselines ant_i-ant
                set2=np.where(self.A1[t_i]==ant)[0]
                self.CoeffArray[t_i,ant,0] = np.mean( np.append(residuals[t_i,set1,:,:],residuals[t_i,set2,:,:]) * np.append(residuals[t_i,set1,:,:],residuals[t_i,set2,:,:]).conj())#, \
#                                                         weights = 1#)flagweights)
                self.CoeffArray[t_i,ant,1] = np.abs( np.mean(np.append(residuals[t_i,set1,:,:],residuals[t_i,set2,:,:])) )
            PrintProgress(t_i,self.nt)
        warnings.filterwarnings("default")
        for i in range(self.nAnt):
            # get rid of NaN
            self.CoeffArray[np.isnan(self.CoeffArray)]=np.inf
            # normalise per antenna
            if self.normalise==True:
                self.CoeffArray[:,i,0]=self.CoeffArray[:,i,0]/self.CoeffArray[:,i,1]**2
        if self.weightscolname=="":
            coeffFilename=self.MSName+"/CoeffArray.ntsol%i.npy"%(ntsol)
        else:
            coeffFilename=self.MSName+"/CoeffArray.%s.ntsol%i.npy"%(weightscolname,ntsol)
        print("Save coefficient array as %s."%coeffFilename)
        np.save(coeffFilename,self.CoeffArray)

        for i in range(self.nAnt):
            pylab.scatter(np.arange(self.nt),self.CoeffArray[:,i,0])
        pylab.show()
        stop
        
        return self.CoeffArray



    
    def SaveWeights(self,colname=None,AverageOverChannels=True):
        print("Begin saving the data")
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
        nbl=int(tarray.shape[0]/nt)
        nchan=darray.shape[1]
        A0=np.array(ms.getcol("ANTENNA1").reshape((nt,nbl)))
        A1=np.array(ms.getcol("ANTENNA2").reshape((nt,nbl)))
        if colname in ms.colnames():
            print("%s column already present; will overwrite"%colname)
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
        print("Fill weights array")
        A0ind=A0[0,:]
        A1ind=A1[0,:]
        warnings.filterwarnings("ignore")

        for i in range(nbl):
            for j in range(nchan):               
                w[:,i,j] = 1. / (
                    CoeffArray[:,A0ind[i],0] + \
                    CoeffArray[:,A1ind[i],0] + \
#                    np.sqrt(CoeffArray[:,A0ind[i],0]*CoeffArray[:,A1ind[i],0]) +\
                    0.005)
                
#                w[:,i,j] = 1. / (
#                    CoeffArray[:,A0ind[i],0]*CoeffArray[:,A1ind[i],1] + \
#                    CoeffArray[:,A1ind[i],0]*CoeffArray[:,A0ind[i],1] + \
#                    CoeffArray[:,A0ind[i],0]*CoeffArray[:,A1ind[i],0] + \
#                    0.1)
            PrintProgress(i,nbl)
        warnings.filterwarnings("default")
        w=w.reshape(nt*nbl,nchan)
        w[np.isnan(w)]=0
        w[np.isinf(w)]=0
        # normalise
        w=w/np.mean(w)
        # check shape of column we are writing to
        if "WEIGHT_SPECTRUM" in ms.colnames():
            if ms.getcol(colname).shape==ms.getcol("WEIGHT_SPECTRUM").shape:
                w1=np.zeros_like(ms.getcol("WEIGHT_SPECTRUM"))
                for i in range(4):
                    w1[:,:,i]=w
                w=w1
        
        if ms.getcol(colname).shape[-1]==4:
            
            # We are writing to a weight col of shape (nbl*nt, nchan, npol) i.e. WEIGHT_SPECTRUM style
            # and not of shape (nbl*nt, nchan) i.e. IMAGING_WEIGHT style
            print()
        # save in weights column
        if colname!=None:
            ms.putcol(colname,w)
        else: print("No colname given, so weights not saved in MS.")
        print(w.shape)
        for i in range(nchan):
            pylab.scatter(np.arange(w.shape[0]),w[:,i])
        pylab.show()
        ants.close()
        ms.close()

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
    parser.add_argument("--ntsol",     type=int, help="Solution interval, in timesteps, for your calibration",required=True)
    parser.add_argument("--nfreqsol",  type=int, help="Frequency interval, in channels, for your calibration. Default is 8.",required=False,default=8)
    parser.add_argument("--weightcol", type=str, help="Name of the weights column name you want to save the weights to. Default is CAL_WEIGHT.",required=False,default="CAL_WEIGHT")
    parser.add_argument("--datacol",   type=str, help="Name of the data column name you want to read to build residual visibilities. Default is DATA.",required=False,default="DATA")
    parser.add_argument("--modelcol",  type=str, help="Name of the weights column name you want to save the weights to. Default is CAL_WEIGHT.",required=False,default="MODEL_DATA_CORR")
    parser.add_argument("--gainfile",  type=str, help="Name of the gain file you want to read to rebuild the calibration quality weights."+\
                        " If no file is given, equivalent to rebuilding weights for phase-only calibration.",required=False,default="")
    parser.add_argument("--uvcutkm",   type=float,nargs=2,default=[0,3000],required=False,help="uvcut used during calibration, in km.")
    parser.add_argument("--phaseonly",           help="Use if calibration was phase-only; this means that gain information doesn't need to be read.",required=False,action="store_true")
    parser.add_argument("--normalise",           help="Normalise gains to avoid suppressing long baselines",required=False,action="store_true")
    args=parser.parse_args()
    return vars(args)



### if program is called as main ###
if __name__=="__main__":
    start_time     = time.time()
    args           = readArguments()
    mslist         = args["filename"]
    ntsol          = args["ntsol"]
    nfreqsol       = args["nfreqsol"]
    weightscolname = args["weightcol"]
    modelcolname   = args["modelcol"]
    datacolname    = args["datacol"]
    gainfile       = args["gainfile"]
    uvcut          = args["uvcutkm"]
    phaseonly      = args["phaseonly"]
    normalise      = args["normalise"]
    for msname in mslist:
        print("Finding time-covariance weights for: %s"%msname)
        covweights=CovWeights(MSName=msname,ntsol=ntsol,nfreqsol=nfreqsol, gainfile=gainfile,uvcut=uvcut,phaseonly=phaseonly, \
                              norm=normalise, modelcolname=modelcolname, datacolname=datacolname, weightscolname=weightscolname)
        coefficients=covweights.FindWeights()
        covweights.SaveWeights(coefficients,colname=weightscolname,AverageOverChannels=True)
        print("Total runtime: %f min"%((time.time()-start_time)/60.))
