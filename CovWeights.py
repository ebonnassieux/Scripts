import os
from pyrap.tables import table
import numpy as np
import pylab
from numpy import ma
import sys
import warnings
import time
import math

class CovWeights:
    def __init__(self,MSName,ntsol=1,MaxCorrTime=0,SaveDataProducts=0):
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
        #norm=1/ms.getcol("PREDICTED_VIS")
        warnings.filterwarnings("default")
        # get rid of NaN
        #norm[np.isnan(norm)]=0
        if "RESIDUAL_DATA" not in ms.colnames():
            desc=ms.getcoldesc("CORRECTED_DATA")
            desc["name"]="RESIDUAL_DATA"
            desc['comment']=desc['comment'].replace(" ","_")
            ms.addcols(desc)
            ms.putcol("RESIDUAL_DATA",ms.getcol("CORRECTED_DATA"))
        # bootes test; true flags are in different dir
        flags=ms.getcol("FLAG")
        #flags=np.load(self.MSName+"/Flagging.npy")
        print "Please ensure that RESIDUAL_DATA or CORRECTED_DATA contains residual visibilities from complete skymodel subtraction, and PREDICTED_VIS contains the uncalibrated flux."
        residualdata=ms.getcol("RESIDUAL_DATA")#*norm
        
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
        residuals[:,:,:,0]=darray[:,:,:,0]#residualdata[:,:,:,0]
        residuals[:,:,:,1]=darray[:,:,:,3]#residualdata[:,:,:,3]
        # antenna coefficient array
        # TODO - MAKE COEFFARRAY INTO COMPLEX-VALUED WEIGHTS
        CoeffArray=np.zeros((nt,nAnt))#,dtype=np.complex64)
        # start calculating the weights
        print "Begin calculating antenna-based coefficients"
        if tcorr>1:

            ### SAVE DATA ###

            colname="FULL_WEIGHT"
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
            np.save("fullw.npy",w)
            ant1=np.arange(nAnt)
            print "Fill weights array"
            A0ind=A0[0,:]
            A1ind=A1[0,:]

            ### WEIGHT COL CREATED ###

            warnings.filterwarnings("ignore")
            print "find covariance between %i nearest times"%tcorr
            ### debug ###
            for ant1 in range(nAnt):
                msg="Ant1: %i/%i Progress: "%(ant1+1,nAnt+1)
                for ant2 in range(ant1+1,nAnt):
                    line=0
                    #indA0A1=np.where(((A0==ant1)&(A1==ant2))|((A0==ant2)&(A1==ant1)))[1]
                    #ThisBLresiduals=residuals[:,indA0A1,:,:]
                    #set1=np.where(A0[t_i]==ant1)[0]
                    #set2=np.where(A1[t_i]==ant2)[0]
                    #msg="Ant1: %i/%i Ant2: %i/%i Progress: "%(ant1+1,nAnt+1,ant2+1,nAnt-ant1+1)
                    #print "Antenna %i of %i"%(ant,nAnt)
                    for t_i in range(nt):
                        #set1=A0[t_i]==ant1
                        #set2=A1[t_i]==ant2
                        blset=(A0[t_i]==ant1)*(A1[t_i]==ant2)#set1*set2
                        #set1=np.where(A0[t_i]==ant1)[0]
                        #set2=np.where(A1[t_i]==ant2)[0]
                        t_lo=max(0,t_i-tcorr-1)
                        t_hi=min(nt,t_i+tcorr)
                        cmat=np.zeros((t_hi-t_lo,t_hi-t_lo))
                        # set of vis for baselines ant-ant_i
                        #set1=np.where(A0[t_i]==ant1)[0]
                        #set2=np.where(A1[t_i]==ant1)[0]
                        #ThisBLresiduals=residuals[:,(A0[t_i]==ant)+(A0[t_i]==ant),:,:]
                        # make covmat
                        for iterator1 in range(t_hi-t_lo):
                            for iterator2 in range(t_hi-t_lo):
                                cmat[iterator1,iterator2]= np.mean(residuals[iterator1*self.ntSol,blset,:,:]*residuals[iterator2*self.ntSol,blset,:,:].conj())
#                                cmat[iterator1,iterator2]= np.mean(np.append(residuals[iterator1*self.ntSol,set1,:,:],residuals[iterator1*self.ntSol,set2,:,:])*\
#                                                                   np.append(residuals[iterator2*self.ntSol,set1,:,:],residuals[iterator2*self.ntSol,set2,:,:]).conj())
#                                                            np.append(residuals[iterator2*self.ntSol,set1,:,:],residuals[iterator2*self.ntSol,set2,:,:]).conj())
                                if iterator1==iterator2:
                                    cmat[iterator1,iterator2]=cmat[iterator1,iterator2]-np.std(rmsarray[iterator1,blset,:,:])
                        # invert covmat
                        invcovmat=invSVD(cmat)
                        #invcovmat=np.linalg.inv(cmat)
                        for j in range(nchan):
                            # this is what is going on in current run
                            w[t_i,blset,j]=1/np.max(np.abs(np.mean(invcovmat[line])),0.01)
                            # do a clip later instead
                            #w[t_i,blset,j]=1/(np.abs(np.mean(invcovmat[line]))
                        #    w[t_i,A0ind[i]*A1ind[i],j]=1/np.abs(np.mean(invcovmat[line]))
                        #w[t_i,ant]=np.sqrt(np.abs(np.mean(invcovmat[line])))
                        #print "saving line %i ; "%line,t_hi,t_lo
                        if t_i-tcorr<1:
                            line=line+1
                        elif t_hi==0:
                            line=line-1
                        #msg="Ant1: %i/%i Ant2: %i/%i Progress: "%(ant1,np.max(A0),ant2,np.max(A1))
                    PrintProgress(ant2-ant1,nAnt-ant1,msg=msg)
            warnings.filterwarnings("default")
            # perform clip
            thres=0.1*np.median(w)
            w[w<thres]=thres
            w=w.reshape((nt*nbl,nchan))/npmean(w)
            np.save("fucksake.npy",w)
            ms.putcol(colname,w)
            ms.close()
        else:
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
#                    CoeffArray[t_i,ant] = np.sqrt( np.std( (np.append(residuals[t_i,set1,:,:],residuals[t_i,set2,:,:])))**2  )
                                                  - np.std( (np.append(rmsarray[t_i,set1,:,:], rmsarray[t_i,set2,:,:]))) )
                PrintProgress(t_i,nt)
        warnings.filterwarnings("default")
        for i in range(nAnt):
            thres=0.25*np.median(CoeffArray[:,i])
            CoeffArray[CoeffArray[:,i]<thres,i]=thres
        coeffFilename="CoeffArray.%i.npy"%tcorr
        print "Save coefficient array as %s."%coeffFilename
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
        
        #if tcorr<2:
        warnings.filterwarnings("ignore")
        for i in range(nbl):
            for j in range(nchan):
                w[:,i,j]=1./(CoeffArray[:,A0ind[i]]*CoeffArray[:,A1ind[i]] + 0.01)
                #w[:,i,j]=1./(CoeffArray[:,A0ind[i]]**2 + CoeffArray[:,A1ind[i]]**2 + 0.01)
            PrintProgress(i,nbl)
        warnings.filterwarnings("default")
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
    #test=np.allclose(A,np.dot(u,np.dot(np.diag(s),v)))
    #print "invSVD validity:",test
    # limit impact of small values
    s[s<1.e-6*s.max()]=1.e-6*s.max()
    ssq=np.abs((1./s))
    # rebuild matrix
    Asq=np.dot(v,np.dot(np.diag(ssq),np.conj(u)))
    v0=v.T*ssq.reshape(1,ssq.size)
    return Asq

### if program is called as main ###
if __name__=="__main__":
    start_time=time.time()
    ntsol=7
    corrtime=5
    msname=sys.argv[1]
    print "Finding time-covariance weights for: %s"%msname
    covweights=CovWeights(MSName=msname,ntsol=ntsol)
    coefficients=covweights.FindWeights(tcorr=corrtime)
    #covweights.SaveWeights(coefficients,colname="VAR_WEIGHT",tcorr=corrtime)
    print "Total runtime: %f min"%((time.time()-start_time)/60.)
