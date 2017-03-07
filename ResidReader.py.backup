import os
from pyrap.tables import table
import numpy as np
import pylab
from numpy import ma


class VarianceWeights:
def __init__():



def CleanImagingWeights(MSname):
    t=table(MSname,readonly=False)
    imw=t.getcol("IMAGING_WEIGHT")
    imw2=np.ones(imw.shape)
    t.putcol("IMAGING_WEIGHT",imw2)
    t.close()

def invSVD(A):
    u,s,v=np.linalg.svd(A)
    test=np.allclose(A,np.dot(u,np.dot(np.diag(s),v)))
    print "invSVD validity:",test
#    print u.shape, s.shape,v.shape
    # limit impact of small values
    s[s<1.e-6*s.max()]=1.e-6*s.max()
    ssq=np.abs((1./s))
    # rebuild matrix
#    print np.dot(v.T,ssq).shape
#    print u.shape
#    Asq=np.conj(np.dot(np.dot(v.T,ssq),u.T))
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

def FindCovMat(MSname,imtype="DDF",ntSol=1,CalcHalfMat=False,useInvSVD=False,DiagOnly=False,MaxCorrTime="Full"):
    # DESCRIPTION OF THE INPUT OPTIONS
    # MSname      : this is the measurement set you wish to calculate corrections for
    # imtype      : alpha option, not yet set in stone
    # ntSol       : during calibration, you solved for 1 calibration solution every ntSol timesteps
    # CalcHalfMat : option to calculate only half the covariance matrix, since it is symmetric, rather than use numpy dotproduct
    # useInvSVD   : matrix inversion option, specifies SVD use rather than numpy.linalg.inv
    # MaxCorrTime : if set to an integer, covariance matrix will only be calculated over this time-frame. Units: ntSol
    #               default: "Full" this calculates over the entire timeframe of the MS.
    #               setting MaxCorrTime=0 is equivalent to DiagOnly=True
    # load measurement set
    ms=table(MSname,readonly=False)
    # open antennas
    ants=table(ms.getkeyword("ANTENNA"))
    # open antenna tables
    nAnt=len(ants.getcol("NAME"))
    A0=ms.getcol("ANTENNA1")
    A1=ms.getcol("ANTENNA2")
    Times=ms.getcol("TIME")
    nbl=np.where(Times==Times[0])[0].size
    if imtype=="DDF":
        print "Using DDF products to load flags and construct residual data; loading data from CORRECTED_DATA_BACKUP"
        if "RAW_PREDICTED_DATA" not in ms.colnames():
            print "Please run a Predict without applying Jones matrices, saving the result in RAW_PREDICTED_DATA"
            exit
        norm=1/ms.getcol("RAW_PREDICTED_DATA")
        norm[np.isnan(norm)]=0
        residuals=(ms.getcol("CORRECTED_DATA")-ms.getcol("PREDICTED_DATA"))*norm
        if "RESIDUAL_DATA" not in ms.colnames():
            desc=ms.getcoldesc("CORRECTED_DATA")
            desc["name"]="RESIDUAL_DATA"
            desc['comment']=desc['comment'].replace(" ","_")
            ms.addcols(desc)
        ms.putcol("RESIDUAL_DATA",residuals)
        # bootes test; true flags are in different dir
        flags=np.load(MSname+"/Flagging.npy")
    else:
        print "Not using DDF products: please ensure that CORRECTED_DATA contains residual visibilities from complete skymodel subtraction, normalised by the uncalibrated flux."
        residuals=(ms.getcol("CORRECTED_DATA"))
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
    if ntSol > 1:
        tspill=nt%ntSol
        nt=nt+ntSol-tspill
        A=np.append(A,np.zeros(A.shape[:-1]+(ntSol-tspill,)),axis=4)
        A=A.reshape((nAnt*nAnt*nChan*nPola*ntSol,nt/ntSol))
        NPerCell=np.append(NPerCell,np.zeros(NPerCell.shape[:-1]+(ntSol-tspill,)),axis=4)
        NPerCell=NPerCell.reshape((nAnt*nAnt*nChan*nPola*ntSol,nt/ntSol))
    else:
        A=A.reshape((nAnt*nAnt*nChan*nPola,nt))
        NPerCell=NPerCell.reshape((nAnt*nAnt*nChan*nPola,nt))
    # create matrix
    tlen=A.shape[1]
    if MaxCorrTime=="Full":
        tcorrmax=tlen
    else:
        tcorrmax=np.int(MaxCorrTime)
    if DiagOnly==True or tcorrmax==0:
        print "Calculating only diagonal elements of the covariance matrix"
        C=np.zeros((tlen,tlen),np.complex64)
        NMat=np.zeros((tlen,tlen),np.complex64)
        for i in range(tlen):
            C[i,i]=np.sum(np.conj(A[:,i]*A[:,i]))
            NMat[i,i]=np.sum(NPerCell[:,i]*NPerCell[:,i])
    else:
        if CalcHalfMat==True:
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
        else:
            print "Performing numpy dot-product"
            C=np.dot(A.T.conj(),A)
            NMat=np.dot(NPerCell.T,NPerCell)
    # avoid divide-by-zero errors for zeros in denominator
    NMat[NMat==0]=1
    # save raw matrix
    np.save("C.npy",C)
    # normalise                                                                                                                                                                              
    C=C/NMat
    # invert C
    if DiagOnly==True:
        Weight=np.abs(np.diag(np.diag(C)**(-1)))
    else:
        if useInvSVD==True:
            Weight=np.abs(invSVD(C))
        else:
            Weight=np.abs(np.linalg.inv(C))
    # normalise weights matrix
    Weight=Weight/np.sum(Weight)
    # resize matrix appropriately if needed
    if ntSol>1:
        C1=np.zeros((nt,nt),np.complex64)
        Weight1=np.zeros((nt,nt),np.complex64)
        for i in range(tlen):
            for j in range(tlen):
                C1[i*ntSol:(i+1)*ntSol,j*ntSol:(j+1)*ntSol]=C[i,j]
                Weight1[i*2:(i+1)*ntSol,j*ntSol:(j+1)*ntSol]=Weight[i,j]
        C=C1[0:(nt+tspill-ntSol),0:(nt+tspill-ntSol)]
        Weight=Weight1[0:(nt+tspill-ntSol),0:(nt+tspill-ntSol)]
    np.save("Cnorm.npy",C)
    np.save("NMat.npy",NMat)
    # invert C
    if DiagOnly==True:
        Weight=np.abs(np.diag(np.diag(C)**(-1)))
    else:
        if useInvSVD==True:
            Weight=np.abs(invSVD(C))
        else:
            Weight=np.abs(np.linalg.inv(C))
    # normalise weights matrix
    Weight=Weight/np.sum(Weight)
    # save weights matrix
    np.save("Weight.npy",Weight)

def AddImagingWeights(MSname):
    ms=table(MSname,readonly=False)
    tarray=ms.getcol("TIME")
    darray=ms.getcol("DATA")
    tvalues=np.array(sorted(list(set(tarray))))
    nt=tvalues.shape[0]
    nbl=tarray.shape[0]/nt
    nchan=darray.shape[1]
    W=np.ones((nt*nbl,nchan))
    desc=ms.getcoldesc("WEIGHT")
    desc["name"]="IMAGING_WEIGHT"
    desc['comment']=desc['comment'].replace(" ","_")
    ms.addcols(desc)
    ms.putcol("IMAGING_WEIGHT",W)



def SaveWeights(Weight,MSname):
    # open measurement set to store CTB_DATA
    ms=table(MSname,readonly=False)
    # open TIME to assign weights properly
    tarray=ms.getcol("TIME")
    W=ms.getcol("IMAGING_WEIGHT")
    nrow,nchan=W.shape
    tvalues=np.array(sorted(list(set(tarray))))
    nt=tvalues.shape[0]
    nbl=tarray.shape[0]/nt
    Wr=W.reshape((nt,nbl,nchan))
    for ibl in range(nbl):
         for ichan in range(nchan):
#             for i in range(nt):
#                 Wr[i,ibl,ichan]=1/np.abs(2*Weight[i]-np.sum(Weight[:],axis=0))
             Wr[:,ibl,ichan]=np.sum(Weight[:],axis=0)
    W=Wr.reshape((nt*nbl,nchan))
    # normalise
    W=W/np.mean(W)
    # store CTB_data in measurement set
    if ("COV_WEIGHT" not in ms.colnames()):
        desc=ms.getcoldesc("IMAGING_WEIGHT")
        desc["name"]="COV_WEIGHT"
        desc['comment']=desc['comment'].replace(" ","_")
        ms.addcols(desc)
    ms.putcol("COV_WEIGHT",np.abs(W))
    ms.close()
