import numpy as np
import scipy.signal
import pylab


#def Gauss(Npix,mu=0,sig=1,Dx=None):
#    if Dx==None:
#        Dx=Npix/2.
#    x=np.mgrid[-Dx+0.5:Dx-0.5:1j*(Npix)]
#    return np.exp(-(x-mu)**2/(2.*sig**2))

def Gauss(Npix,mu=0,sig=1):
    x=np.arange(Npix)
    out=np.exp(-(x-mu)**2/(2.*sig**2))
    return out

class ClassSampleMachine():
    
    def __init__(self,T=10,NPoints=100,V0=2.,sigma=0):
        self.T=T
        self.N=NPoints
        self.V0=V0
        self.sigma=sigma
        self.setCovariance()

    def sqrtSVD(self,A):
        u,s,v=np.linalg.svd(A)
        s[s<0.]=0.
        ssq=np.sqrt(np.abs(s))
        v0=v.T*ssq.reshape(1,ssq.size)
        Asq=np.conj(np.dot(v0,u.T))
        return Asq

    def setCovariance(self):
        V0=self.V0
        T=self.T
        N=self.N

        C=np.zeros((N,N),np.complex64)

        #CDiag=np.diag(np.ones((N,),np.complex64))*V0
        CDiag=np.ones((N,),np.complex64)*V0

        CGauss=np.zeros_like(C)
        for iPix in range(N):
#            for jPix in range(N):
            CGauss[iPix,:]=Gauss(N,mu=iPix,sig=self.sigma)
#                CGauss[iPix,jPix]=V0*np.exp(-(x-mu)**2/(2.*self.sigma**2))#*Gauss(N,mu=iPix,sig=self.sigma)[iPix]
                
        w1=np.zeros(N)
        for iPix in range(N):
            if iPix%T==0:
                w1+=Gauss(N,mu=iPix,sig=T/8.)
            #w1=np.sin(float(iPix)/T*2.*np.pi)-0.3
            #if w1<0.00001: w1=0.00001
            w0=1.-0.5*(np.sin(float(iPix)/T*2.*np.pi)-0.3)#w1[ipix]            
            if w0<0.1: w0=0.1
            #C[iPix,:]+=w0*CDiag[iPix,:]
            CDiag[iPix]=1.*CDiag[iPix]*w0+2.*(1+np.cos(float(iPix)/T*np.pi))+1
            if iPix%150==0:
                CDiag[iPix]+=1000.
            C[iPix,:]+=V0*CGauss[iPix,:]
            #C[:,iPix]+=w0*CGauss[:,iPix]
        #C=(C.T+C)/2.
#        C=np.sqrt(C.T.conj()*C)
        # put oscillation
        rotated = zip(*C[::-1])
        temp=np.zeros_like(rotated)
        for i in range(2*N):
            w1=np.sin(float(i)/T*2.*np.pi)-0.1
            if w1<0.0000: w1=0.0000
            kval=N-i
            temp+=rotated*np.eye(N,k=kval)*w1
        C=zip(*temp[::-1])

        # add diag
        C+=np.diag(CDiag)

        #pylab.imshow(np.abs(C),interpolation="nearest")
        #pylab.show()
                   

        #C=np.sqrt(C.T*C)
        #self.Cov=C
#        self.L=np.linalg.cholesky(C)
        self.L=self.sqrtSVD(C)
        self.Cov=np.dot(self.L,self.L.T)
        #import pylab
        # pylab.clf()
        #pylab.plot(np.abs(C[200]))
        #pylab.plot(np.abs(self.Cov[200]))
        #pylab.imshow(np.abs(self.Cov),interpolation="nearest")
        #pylab.show()
        #stop
        # pylab.show(False)
        # pylab.pause()
    
    
    def GiveSample(self,NSamples=1000):
        
        C=self.Cov
        L=self.L
        N,_=C.shape
        x=(np.random.randn(NSamples,N)+1j*np.random.randn(NSamples,N))/np.sqrt(2.)    
        y=np.zeros_like(x)
        for iSample in range(NSamples):
            y[iSample,:]=np.dot(L,x[iSample].reshape((N,1))).flat[:]
    

            # pylab.clf()
            # pylab.plot(y[iSample,:].real)
            # pylab.plot(x[iSample,:].imag)
            # pylab.draw()
            # pylab.show(False)
            # pylab.pause(0.1)
        return y
    
    
    
    

def test(sigma1=0):
    SM=ClassSampleMachine(sigma=sigma1)
    test2=ClassSampleMachine(T=10,NPoints=100,V0=2.,sigma=sigma1)
    y=SM.GiveSample(NSamples=1000)
    COut=np.dot(y.T.conj(),y)/y.shape[0]

    C=SM.Cov
    pylab.clf()
    pylab.subplot(1,3,1)
    pylab.imshow(C.real,interpolation="nearest")
    pylab.title("Input Covariance")
    pylab.colorbar()
    pylab.subplot(1,3,2)
    pylab.imshow(COut.real,interpolation="nearest")
    pylab.title("Measured from sample")
    pylab.colorbar()
    pylab.subplot(1,3,3)
    pylab.imshow((C-COut).real,interpolation="nearest")
    pylab.title("Difference")
    pylab.colorbar()
    pylab.draw()
    pylab.show(False)
    pylab.pause(0.1)
