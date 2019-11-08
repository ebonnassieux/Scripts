import numpy as np
import pylab
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
from scipy.constants import c


def SimulateSignal(xlen,ndiracs=1,posdiracs=0,ampdiracs=1,ntophats=0,postophats=0,amptophats=0,sigtophat=0):
    x=np.arange(xlen)
    signal=np.zeros_like(x).astype(float)
    for i in range(ndiracs):
        signal[posdiracs[i]]=signal[posdiracs[i]]+ampdiracs[i]
    for i in range(ntophats):
        imin=postophats[i]-sigtophat[i]
        imax=postophats[i]+sigtophat[i]
        signal[imin:imax]=signal[imin:imax]+amptophats[i]
    return signal

def SimulateConvFunctionProper(nphis,freqmin=120e6,freqmax=180e6,nchans=365*8,phimax=40,dolambda0=True):
    deltanu=1.*(freqmax-freqmin)/nchans
    freqvals=np.arange(nchans)*deltanu+freqmin
    lambdavals=np.array(list(np.sort(c/freqvals)))
    if dolambda0:
        lambdavals=lambdavals-np.mean(lambdavals)
    lambd2vals=np.reshape(lambdavals**2,(1,nchans))
    phivals=np.reshape((2.*phimax*np.arange(nphis)/(nphis-1)-phimax),(nphis,1))
    expvals=np.exp(-2*1j*np.pi*np.dot(phivals,lambd2vals))
    weights=np.ones_like(expvals) # this will need to change as function of flags...dims nphis,nchan
    psfvals=np.mean(weights*expvals,axis=1,dtype=float)
    #xvals=np.arange(xlen)-(xlen-1)/2.
    #x0=0
    #psfvals=np.exp( -(xvals-x0)**2/(2*sigma**2) )#*(0.5+np.sinc(np.pi*(xvals-x0)/sigma))/1.5
    #psfvals=psfvals+0.3*np.exp( -(xvals-x0-xlen/4)**2/(2*sigma**2) )
    #psfvals=psfvals+0.3*np.exp( -(xvals-x0+xlen/4)**2/(2*sigma**2) )
    #psfvals=np.abs(np.sinc(xvals/(xlen/10.)))+0.1
    return phivals,psfvals

def SimulateConvFunction(xlen,sigma):
    xvals=np.arange(xlen)-(xlen-1)/2.                                                                                                                                                                      
    x0=0
    x1=xlen/3.
    x2=-xlen/3.
    psfvals=np.exp( -(xvals-x0)**2/(2*sigma**2) )#*(0.5+np.sinc(np.pi*(xvals-x0)/sigma))/1.5
    psfvals=psfvals+.3*np.exp( -(xvals-x1)**2/(2*sigma**2) )
    psfvals=psfvals+.3*np.exp( -(xvals-x2)**2/(2*sigma**2) )
    return psfvals
    
def MakeCovMat(psf):
    xlen=len(psf)
    imat=np.zeros((xlen,xlen))
    imat=imat+np.diag(np.ones(xlen))
    for i in range(xlen):
        imat[i,:]=np.convolve(imat[i],psf,mode="same")
    #pylab.imshow(imat,interpolation="nearest")
    return imat

def invertMat(covmat):
    #invcovmat=np.linalg.inv(covmat)
    #pylab.imshow(np.dot(invcovmat,covmat),interpolation="nearest")
    u,s,v=np.linalg.svd(covmat)
    s=1/s
    s[s<0.]=0.
    #ssq=np.sqrt(np.abs(s))
    v0=v.T*s.reshape(1,s.size)
    Asq=np.conj(np.dot(v0,u.T))

    return Asq


def sqrtSVD(self,A):
    u,s,v=np.linalg.svd(A)
    s[s<0.]=0.
    ssq=np.sqrt(np.abs(s))
    v0=v.T*ssq.reshape(1,ssq.size)
    Asq=np.conj(np.dot(v0,u.T))
    return Asq

    
def MakeFinalPlot(signal,psf,covmat,covsignal,invcovmat,deconvsignal,xvals):
    xlab=r'$\phi$'
    ylab=r"Value"
    xvals=xvals
    pylab.figure(figsize=(14,10))
    pylab.suptitle(r"Showing effect of deconvolution through linear algebra")
    pylab.subplot(2,3,1)
    pylab.title(r"Simulated Signal")
    pylab.xlabel(xlab)
    pylab.ylabel("Signal")
    pylab.plot(xvals,signal)
    pylab.subplot(2,3,2)
    pylab.title(r"Convolution Function")
    pylab.xlabel(xlab)
    pylab.ylabel("RMTF")
    pylab.plot(xvals,psf)
    pylab.subplot(2,3,3)
    pylab.title(r"Convolved Signal")
    pylab.xlabel(xlab)
    pylab.ylabel("Convolved Signal")
    pylab.plot(xvals,covsignal)    
    covmatfig=pylab.subplot(2,3,4)
    pylab.title(r"Covariance matrix")
    pylab.xlabel(xlab)
    pylab.ylabel(xlab)
    arse=pylab.imshow(covmat,interpolation="nearest",extent=[np.min(xvals),np.max(xvals),np.min(xvals),np.max(xvals)])
    divider1 = make_axes_locatable(covmatfig)
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    pylab.colorbar(arse, cax=cax1)
    covmatfig=pylab.subplot(2,3,5)
    pylab.title(r"$abs(C C^{-1}-I)$")
    pylab.xlabel(xlab)
    pylab.ylabel(xlab)
    arse=pylab.imshow(np.abs(invcovmat-np.diag(np.ones(invcovmat.shape[0]))),interpolation="nearest",extent=[np.min(xvals),np.max(xvals),np.min(xvals),np.max(xvals)])
    divider1 = make_axes_locatable(covmatfig)
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    pylab.colorbar(arse, cax=cax1)
    pylab.subplot(2,3,6)
    pylab.title(r"Deconvolved Signal")
    pylab.xlabel(xlab)
    pylab.ylabel("Recovered Signal")
    pylab.plot(xvals,deconvsignal)

    pylab.tight_layout(rect=[0, 0.03, 1, 0.95])
    pylab.savefig("success.png")
    pylab.show()



if __name__=="__main__":
    start = time.time()
    xlen=502
    posdiracs=(np.array([0.4,0.7,0.1])*xlen).astype(int)
    ampdiracs=np.array([1,0.6,1.2])
    ndiracs=len(posdiracs)
    postophats=(np.array([0.5,0.9])*xlen).astype(int)
    amptophats=np.array([0.2,0.4])
    sigtophats=(np.array([0.15,0.1])*xlen).astype(int)
    ntophats=len(postophats)
    signal=SimulateSignal(xlen,ndiracs,posdiracs,ampdiracs,ntophats,postophats,amptophats,sigtophats)
    #pylab.plot(signal)
    #pylab.show()
    xvals,psf=SimulateConvFunctionProper(xlen)
    #psf=SimulateConvFunction(xlen,xlen/15.)
    covmat=MakeCovMat(psf)
    covsignal=np.dot(covmat,signal)
    invcovmat=invertMat(covmat)
    deconvsignal=np.dot(invcovmat,covsignal)
    end = time.time()
    print "Runtime:", end-start
    MakeFinalPlot(signal,psf,covmat,covsignal,invcovmat,deconvsignal,xvals)
