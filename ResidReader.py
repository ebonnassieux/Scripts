import os
from pyrap.tables import table
import numpy as np
import pylab
from numpy import ma
import sys
import warnings

class VarianceWeights:
    def __init__(self,MSname,imtype="DDF",ntSol=1,CalcHalfMat=False,useInvSVD=False,DiagOnly=False,SaveDataProducts=True,MaxCorrTime="Full"):
        # DESCRIPTION OF THE INPUT OPTIONS
        # MSname      : this is the measurement set you wish to calculate corrections for
        # imtype      : alpha option, not yet set in stone
        # ntSol       : during calibration, you solved for 1 calibration solution every ntSol timesteps
        # CalcHalfMat : option to calculate only half the covariance matrix, since it is symmetric, rather than use numpy dotproduct
        # useInvSVD   : matrix inversion option, specifies SVD use rather than numpy.linalg.inv
        # MaxCorrTime : if set to an integer, covariance matrix will only be calculated over this time-frame. Units: ntSol
        #               default: "Full" this calculates over the entire timeframe of the MS.
        #               setting MaxCorrTime=0 is equivalent to DiagOnly=True
        if MSname[-1]=="/":
            self.MSName=MSname[0:-2]
        else:
            self.MSName=MSname
        self.imtype=imtype
        self.ntSol=ntSol
        self.CalcHalfMat=CalcHalfMat
        self.useInvSVD=useInvSVD
        self.DiagOnly=DiagOnly
        self.MaxCorrTime=MaxCorrTime
        self.SaveDataProducts=SaveDataProducts


    def FindCovMat(self):
        # load measurement set
        ms=table(self.MSName,readonly=False)
        # open antennas
        ants=table(ms.getkeyword("ANTENNA"))
        # open antenna tables
        nAnt=len(ants.getcol("NAME"))
        A0=ms.getcol("ANTENNA1")
        A1=ms.getcol("ANTENNA2")
        Times=ms.getcol("TIME")
        nbl=np.where(Times==Times[0])[0].size
        # stop python from giving warning here; NaN accounted for right afterwards
        warnings.filterwarnings("ignore")
        norm=1/ms.getcol("PREDICTED_DATA")
        norm[np.isnan(norm)]=0
        # put warnings back on
        warnings.filterwarnings("default")
        if self.imtype=="DDF":
            print "Using DDF products to load flags and construct residual data; loading data from CORRECTED_DATA_BACKUP"
#            residuals=ms.getcol("RESIDUAL_DATA")#(ms.getcol("CORRECTED_DATA")-ms.getcol("PREDICTED_DATA"))*norm
            if "RESIDUAL_DATA" not in ms.colnames():
                desc=ms.getcoldesc("CORRECTED_DATA")
                desc["name"]="RESIDUAL_DATA"
                desc['comment']=desc['comment'].replace(" ","_")
                ms.addcols(desc)
                # TODO; make this into actually taking residuals eh
                ms.putcol("RESIDUAL_DATA",ms.getcol("CORRECTED_DATA"))
            residuals=ms.getcol("RESIDUAL_DATA")
            # bootes test; true flags are in different dir
            flags=ms.getcol("FLAG")#np.load(self.MSName+"/Flagging.npy")
        else:
            print "Not using DDF products: please ensure that CORRECTED_DATA contains residual visibilities from complete skymodel subtraction, normalised by the uncalibrated flux."
            residuals=(ms.getcol("CORRECTED_DATA")*norm)
            flags=ms.getcol("FLAG")
        # apply flags to data
        residuals[flags==1]=0
        # exit files gracefully
        ms.close()
        ants.close()
        # initialise matshape
        nChan=residuals.shape[1]
        nPola=residuals.shape[2]
        nt=residuals.shape[0]/nbl
        MatShape=(nAnt,nAnt,nChan,nPola,nt)
        A=np.zeros(MatShape,np.complex64)
        NPerCell=np.zeros(MatShape,np.complex64) # defined as complex to avoid losing information
        print "Begin creation of residual arrays"
        for ant1 in range(nAnt):
            for ant2 in range(nAnt):
                PrintProgress(ant1*nAnt+ant2,nAnt*nAnt)
    #            print ant1,ant2
                # PARALLELISE HERE
                indA0A1=np.where(((A0==ant1)&(A1==ant2))|((A0==ant2)&(A1==ant1)))[0]
                # skip loop when antenna has no corresponding baselines
                if indA0A1.size==0: continue
                # skip autocorrelations
                if ant1==ant2: continue
                for ichan in range(nChan):
                    for ipol in range(nPola):
                        A[ant1,ant2,ichan,ipol,:]=residuals[indA0A1,ichan,ipol]
                        NPerCell[ant1,ant2,ichan,ipol,:]+=(1-flags[indA0A1,ichan,ipol])
        # reshape in case of a single calsol for multiple times, to mitigate related problems
        if self.ntSol > 1:
            tspill=nt%self.ntSol
            nt=nt+self.ntSol-tspill
            A=np.append(A,np.zeros(A.shape[:-1]+(self.ntSol-tspill,)),axis=4)
            A=A.reshape((nAnt*nAnt*nChan*nPola*self.ntSol,nt/self.ntSol))
            NPerCell=np.append(NPerCell,np.zeros(NPerCell.shape[:-1]+(self.ntSol-tspill,)),axis=4)
            NPerCell=NPerCell.reshape((nAnt*nAnt*nChan*nPola*self.ntSol,nt/self.ntSol))
        else:
            A=A.reshape((nAnt*nAnt*nChan*nPola,nt))
            NPerCell=NPerCell.reshape((nAnt*nAnt*nChan*nPola,nt))
        # create matrix
        tlen=A.shape[1]
        if self.MaxCorrTime=="Full":
            tcorrmax=tlen
        else:
            tcorrmax=np.int(self.MaxCorrTime)
        if self.DiagOnly==True or tcorrmax==0:
            print "Calculating only diagonal elements of the covariance matrix"
            C=np.zeros((tlen,tlen),np.complex64)
            NMat=np.zeros((tlen,tlen),np.complex64)
            for i in range(tlen):
                C[i,i]=np.sum(np.conj(A[:,i]*A[:,i]))
                NMat[i,i]=np.sum(NPerCell[:,i]*NPerCell[:,i])
                PrintProgress(i,tlen)
        else:
            if self.CalcHalfMat==True:
                print "Performing Half-Matrix Calculation and Mirroring"
                C=np.zeros((tlen,tlen),np.complex64)
                NMat=np.zeros((tlen,tlen),np.complex64) 
                for i in range(tlen):
                    C[i,i]=np.sum(A[:,i]*np.conj(A[:,i]))
                    NMat[i,i]=np.sum(NPerCell[:,i]*NPerCell[:,i])
                    for j in range(max(0,i-tcorrmax),i):
                        C[i,j]=np.sum(A[:,i].conj()*A[:,j])
                        C[j,i]=np.conj(C[i,j])
                        NMat[i,j]=np.sum(NPerCell[:,i]*NPerCell[:,j])
                        NMat[j,i]=np.conj(NMat[i,j])
                        PrintProgress((i*tlen+j)/2,tlen*max(0,tlen-tcorrmax)/2)
                    #sys.stdout.write("Progress: %5.1f %%"%(100*(i+1.)/tlen))
                    #sys.stdout.flush()
                    #sys.stdout.write("\r")
            else:
                print "Performing numpy dot-product"
                C=np.dot(A.T.conj(),A)
                NMat=np.dot(NPerCell.T,NPerCell)
        # avoid divide-by-zero errors for zeros in denominator
        NMat[NMat==0]=1
        # save raw matrix
        if self.SaveDataProducts==True:
            np.save(self.MSName+"/C.npy",C)
        # normalise
        C=C/NMat
        # invert C
        if self.DiagOnly==True:
#            print "DEBUG", np.sum(NMat)
            Weight=np.abs(np.diag((np.diag(C)/np.diag(NMat))**(-1)))
        else:
            # normalise
            C=C/NMat
        if self.DiagOnly==False:
            if self.useInvSVD==True:
                Weight=np.abs(invSVD(C))
            else:
                Weight=np.abs(np.linalg.inv(C))
        # pathological case where all data at 1 time is flagged
        Weight[np.isnan(Weight)]=0
        # normalise weights matrix
        Weight=Weight/np.mean(Weight)
        # resize matrix appropriately if needed
        if self.ntSol>1:
            C1=np.zeros((nt,nt),np.complex64)
            Weight1=np.zeros((nt,nt),np.complex64)
            for i in range(tlen):
                for j in range(tlen):
                    C1[i*self.ntSol:(i+1)*self.ntSol,j*self.ntSol:(j+1)*self.ntSol]=C[i,j]
                    Weight1[i*2:(i+1)*self.ntSol,j*self.ntSol:(j+1)*self.ntSol]=Weight[i,j]
            C=C1[0:(nt+tspill-self.ntSol),0:(nt+tspill-self.ntSol)]
            Weight=Weight1[0:(nt+tspill-self.ntSol),0:(nt+tspill-self.ntSol)]
        if self.SaveDataProducts==True:
            np.save(self.MSName+"/Cnorm.npy",C)
            np.save(self.MSName+"/NMat.npy",NMat)
        # invert C
        if self.DiagOnly==True:
            Weight=np.abs(np.diag(1/np.diag(C)))#**(-1)))
            Weight[np.diag(np.diag(C))==0]=0
        else:
            if self.useInvSVD==True:
                Weight=np.abs(invSVD(C))
            else:
                Weight=np.abs(np.linalg.inv(C))
        # normalise weights matrix
#        Weight=Weight/np.sum(Weight)
        # save weights matrix
        if self.SaveDataProducts==True:
            np.save(self.MSName+"/Weight.npy",Weight)
        return Weight

    def SaveWeights(self,CovMatrix,diagonly=True):
#        CovMatrix=CovMatrix/np.mean(CovMatrix)
       # open measurement set to store CTB_DATA
        ms=table(self.MSName,readonly=False)
        # open TIME to assign weights properly
        tarray=ms.getcol("TIME")
        if ("IMAGING_WEIGHT" not in ms.colnames()):
            AddWeightsCol(ms,Colname="IMAGING_WEIGHT")
        W=np.ones_like(ms.getcol("IMAGING_WEIGHT"))
#        print "DEBUG",np.mean(W)
        nrow,nchan=W.shape
        tvalues=np.array(sorted(list(set(tarray))))
        nt=tvalues.shape[0]
        nbl=tarray.shape[0]/nt
        Wr=W.reshape((nt,nbl,nchan))
        if diagonly==True:
            normW=np.diag(CovMatrix)/np.mean(CovMatrix)
        else:
            normW=np.mean(CovMatrix,axis=0)/np.mean(CovMatrix)
#        Wr=np.tile(normW,(nbl,nchan))
        for ibl in range(nbl):
            for ichan in range(nchan):
                Wr[:,ibl,ichan]=normW
                PrintProgress(ibl+ichan,nbl+nchan)
        W=Wr.reshape((nt*nbl,nchan))
        # normalise
#        W=np.copy(W/np.mean(W))
        W=W/np.mean(np.mean(W,axis=0),axis=0)
        print "DEBUG", np.mean(W)
        # store CTB_data in measurement set
        if ("COV_WEIGHT" not in ms.colnames()):
            desc=ms.getcoldesc("IMAGING_WEIGHT")
            desc["name"]="COV_WEIGHT"
            desc['comment']=desc['comment'].replace(" ","_")
            ms.addcols(desc)
        ms.putcol("COV_WEIGHT",W)
        ms.close()


###########################
### AUXILIARY FUNCTIONS ###
###########################

def AddWeightsCol(MSname,Colname="IMAGING_WEIGHT"):
    ms=table(MSname,readonly=False)
    tarray=ms.getcol("TIME")
    darray=ms.getcol("DATA")
    tvalues=np.array(sorted(list(set(tarray))))
    nt=tvalues.shape[0]
    nbl=tarray.shape[0]/nt
    nchan=darray.shape[1]
    W=np.ones((nt*nbl,nchan))
    desc=ms.getcoldesc("WEIGHT")
    desc["name"]=Colname
    desc['comment']=desc['comment'].replace(" ","_")
    ms.addcols(desc)
    ms.putcol(Colname,W)

def CleanImagingWeights(MSname):
    t=table(MSname,readonly=False)
    imw=t.getcol("IMAGING_WEIGHT")
    imw2=np.ones(imw.shape)
    t.putcol("IMAGING_WEIGHT",imw2)
    t.close()

def invSVD(A):
    u,s,v=np.linalg.svd(A)
    print u,s,v
    test=np.allclose(A,np.dot(u,np.dot(np.diag(s),v)))
    print "invSVD validity:",test
    # limit impact of small values
    s[s<1.e-6*s.max()]=1.e-6*s.max()
    ssq=np.abs((1./s))
    # rebuild matrix
    Asq=np.dot(v,np.dot(np.diag(ssq),np.conj(u)))
    print Asq.shape
    v0=v.T*ssq.reshape(1,ssq.size)
    print v0.shape
    return Asq

def testSVD():
    matdim=100
    MatTest=np.random.random((matdim,matdim))+1j*np.random.random((matdim,matdim))
    print MatTest
    invSVD(MatTest)

def PrintProgress(currentIter,maxIter):
    sys.stdout.flush()
    sys.stdout.write("\rProgress: %5.1f %% "%(100*(currentIter+1.)/maxIter))
    if currentIter==(maxIter-1):
        sys.stdout.write("\n")


if __name__=="__main__":
    DiagOnly=False
    maxcorrtime=500
    msname=sys.argv[1]
    print "Finding time-covariance weights for: %s"%msname
    test=VarianceWeights(MSname=msname,imtype="",ntSol=1,CalcHalfMat=True,useInvSVD=True,DiagOnly=DiagOnly,SaveDataProducts=True,MaxCorrTime=maxcorrtime)
    weights=test.FindCovMat()
    test.SaveWeights(weights,diagonly=DiagOnly)
