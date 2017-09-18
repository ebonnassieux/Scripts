import numpy as np
from pyrap.tables import table
import pylab
import scipy.ndimage
import ClassSampleMachine
from mpl_toolkits.axes_grid1 import make_axes_locatable

class SimulGains:
    def __init__(self,MSname,ant1=0,ant2=55,sigma_sec=5000,realisations=1,timeblock=1,CovInit=True,TimePeriodicity=500,timesteps=0):
        # open measurement set
        self.MS=table(MSname)
        # define antennas
        self.Antenna=table(MSname+"/ANTENNA")
        A0=self.MS.getcol("ANTENNA1")
        A1=self.MS.getcol("ANTENNA2")
        # define single baseline
        self.baseline=np.where((A0==ant1)&(A1==ant2))[0]
        # get UV-tracks
        # debug here
        UV1=self.MS.getcol("UVW")[self.baseline].T
        if timesteps!=0:
            self.UV=UV1[:,0:timesteps]
        else:
            self.UV=UV1
#        stop
        # convert sigma(seconds) into sigma(time-pixels)
        if sigma_sec==0:
            sigma=0.00000001
        else:
            dt=float(self.MS.getcol("INTERVAL")[0])*len(UV1[0,:])/len(self.UV[0,:])
            sigma=1.*sigma_sec/dt
            if sigma<0.00000001:
                sigma=0.00000001
        self.sigma=sigma
# debug: only use first 2 timesteps #1 out of 10 timesteps
#        self.UV=UV1[:,::10]
# end debug
        self.nrow=len(self.UV[0])
        # import time-values for Jacobian
        self.times=self.MS.getcol("TIME")
        # exit gracefully
        self.MS.close()
        self.Antenna.close()
        self.NSample=realisations
        # initiate covariance matrix
        if CovInit==True:
            print "Initialising Covariance"
            self.covar=ClassSampleMachine.ClassSampleMachine(NPoints=self.nrow,sigma=500,T=TimePeriodicity)
            np.save("covmat.npy",self.covar.Cov)
            #fig0=pylab.subplot(1,1,1)
            #pylab.title(r"Example Covariance Matrix")
            #pylab.ylabel(r'$n_t$')
            #pylab.xlabel(r'$n_t$')
            #im0=pylab.imshow(np.abs(self.covar.Cov),interpolation="nearest",cmap="gray")
            #divider0 = make_axes_locatable(fig0)
            #cax0 = divider0.append_axes("right", size="5%", pad=0.05)
            #pylab.colorbar(im0, cax=cax0)
            self.Gains=np.reshape(self.covar.GiveSample(NSamples=self.NSample),(self.NSample,self.nrow,1,1))

    def DFT(self,func,u,v,DiamDeg=1.,Npix=101,loop=False):
        print "in DFT"
        # convert diameter to degrees
        Diam=DiamDeg*np.pi/180
        # create l,m
        l1,m1=np.mgrid[-Diam/2.:Diam/2.:1j*(Npix+1),-Diam/2.:Diam/2.:1j*(Npix+1)]
        l1=l1+0.5*Diam/Npix
        m1=m1+0.5*Diam/Npix
        l=l1[0:-1,0:-1]
        m=m1[0:-1,0:-1]
        # reshape for pythonic operations
        usize=u.size
        u=u.reshape((1,u.size,1,1))
        v=v.reshape((1,v.size,1,1))
        l=l.reshape((1,1,Npix,Npix))
        m=m.reshape((1,1,Npix,Npix))
        # default method for calculating ImStack
        if loop==False:
            print "in pythonic loop"
            # make cube, stack it along time-axis
            CubeIm=np.exp(2.*np.pi*1j*(u*l+v*m))*func
            ImStack=np.mean(CubeIm,axis=1)
        # this loop is for when func is too big, to limit RAM use
        if loop==True:
            print "in iterator loop"#,func.shape
            # initialise cube
            ImStack=np.zeros((self.NSample,Npix,Npix),np.complex64)
            print "begin iteration"
            # iterate over pixels
            for ipix in range(Npix):
                for jpix in range(Npix):
#                    print "so far, at: %f / %f, %f / %f"%(ipix,Npix,jpix,Npix)
                    ImStack[:,ipix,jpix]=np.mean(func[:,:,0,0]*np.exp(2.*np.pi*1j*(u[0,:,0,0]*l[0,0,ipix,jpix]+v[0,:,0,0]*m[0,0,ipix,jpix])),axis=1)
        # return outputs and exit gracefully
        return np.copy(ImStack)

    def MakeCovMapSimul(self,residual_gains,sigma=0,diamdeg=1.,npix=101):
        print "in simulated covmap"
        # find UV tracks
        us,vs,_=self.UV
        # make image
        image=self.DFT(residual_gains,us,vs,DiamDeg=diamdeg,Npix=npix,loop=False)
        varimage=np.var(image,axis=0)
        ax1=residual_gains[:,:,0,0]
        nmatrix=np.dot((ax1.conj()).T,ax1)/self.NSample
        return np.copy(image),np.copy(varimage),np.copy(nmatrix)
 
    def AnalyticCovMatrix(self):
        CTh=self.covar.Cov
        return np.copy(CTh)

    def MakeCovMapTheoretical(self,residual_gains,sigma=0,diamdeg=1.,npix=101,weights=None):
        # convolve residual gains
        print "in theoretical covmap"
        # find number of timesteps in a given realisation
        NSample=len(np.array(residual_gains)[0,:,0,0])
        # make delta-u,delta-v tracks
        us,vs,_=self.UV
        dus=us.reshape((1,us.size))-us.reshape((us.size,1))
        dvs=vs.reshape((1,vs.size))-vs.reshape((vs.size,1))
        dus=dus.flatten()
        dvs=dvs.flatten()
        print nmatrix
        for j in range(nmatrix.shape[0]):
            for i in range(nmatrix.shape[1]):
                if weights != None:
                    nmatrix[j,i]=nmatrix[j,i]*(weights[j]*weights[i])/np.mean(weights)**2
#        print "after weight-apply:\n",np.abs(np.round(nmatrix,3))
        dudvdata=nmatrix.flatten()
        dudvdata=dudvdata.reshape(1,len(dudvdata),1,1)
        # make image
        image=self.DFT(dudvdata,dus,dvs,DiamDeg=diamdeg,Npix=npix,loop=True)
        return np.copy(image),np.copy(nmatrix)

    def NoisePsfCrossSection(self,PixSizeDeg=.002,Npix=101,loop=True):
        # find UV tracks
        us,_,_=self.UV
        # find theoretical covmat
        TheoCovMat=self.AnalyticCovMatrix()
        print "TheoCovMat initialised"
        # create corrected matrices
        diagcorrTheoCovMat=np.copy(TheoCovMat)
        fullcorrTheoCovMat=np.copy(TheoCovMat)
        complexcorrTheoCovMat=np.copy(TheoCovMat)
        print "copy made"
        invCovMat=np.linalg.inv(TheoCovMat)
        print "invert matrix found"
        #diagweights=np.abs(np.diag(invCovMat)/np.mean(np.diag(invCovMat)))
        diagweights=(np.mean(np.diag(TheoCovMat))/np.diag(TheoCovMat))
#        stop
        fullweights=np.abs(np.sum(invCovMat,axis=1))/np.abs(np.mean(np.sum(invCovMat,axis=1)))
        complexweights=np.sum(invCovMat,axis=1)/np.abs(np.mean(np.sum(invCovMat,axis=1)))
        print "Matrices, weights initialised"
        for i in range(TheoCovMat.shape[0]):
            diagcorrTheoCovMat[i,:]=diagcorrTheoCovMat[i,:]*diagweights
            diagcorrTheoCovMat[:,i]=diagcorrTheoCovMat[:,i]*diagweights
            fullcorrTheoCovMat[i,:]=fullcorrTheoCovMat[i,:]*fullweights
            fullcorrTheoCovMat[:,i]=fullcorrTheoCovMat[:,i]*fullweights
            complexcorrTheoCovMat[i,:]=fullcorrTheoCovMat[i,:]*complexweights
            complexcorrTheoCovMat[:,i]=fullcorrTheoCovMat[:,i]*complexweights
        # initialise theofuncs
        theofunc=np.reshape(TheoCovMat.ravel(),(1,TheoCovMat.ravel().size,1))
        diagtheofunc=np.reshape(diagcorrTheoCovMat.ravel(),(1,TheoCovMat.ravel().size,1))
        fulltheofunc=np.reshape(fullcorrTheoCovMat.ravel(),(1,TheoCovMat.ravel().size,1))
        complextheofunc=np.reshape(complexcorrTheoCovMat.ravel(),(1,TheoCovMat.ravel().size,1))
        # initialise simufuncs
        simufunc=np.reshape(np.copy(self.Gains),(self.Gains.shape[0],self.Gains.shape[1],1))
        diagsimufunc=np.copy(simufunc)
        fullsimufunc=np.copy(simufunc)
        complexsimufunc=np.copy(simufunc)
        for i in range(simufunc.shape[1]):
            diagsimufunc[:,i,0]=simufunc[:,i,0]*diagweights[i]
            fullsimufunc[:,i,0]=simufunc[:,i,0]*fullweights[i]
            complexsimufunc[:,i,0]=simufunc[:,i,0]*complexweights[i]
        pixsize=PixSizeDeg*np.pi/180

#        diagsimufunc=diagsimufunc/np.mean(diagsimufunc)*np.mean(simufunc)
#        fullsimufunc=diagsimufunc/np.mean(fullsimufunc)*np.mean(simufunc)
#        diagtheofunc=diagtheofunc/np.mean(diagtheofunc)*np.mean(theofunc)
#        fulltheofunc=diagtheofunc/np.mean(fulltheofunc)*np.mean(theofunc)
        

        print "quantities initialised"
        # create l,m
        l=np.arange(Npix)*pixsize
        # reshape for pythonic operations
        usize=us.size
        us=us.reshape((1,us.size,1))
        dus=us.reshape((1,us.size))
        dus=dus-dus.T
        dus=dus.ravel().reshape((1,dus.ravel().size,1))
        l=l.reshape((1,1,Npix))
        print "Calculating images"
        # default method for calculating ImStack
        if loop==False:
            print "in pythonic loop"
            # make cube, stack it along time-axis
#            SimuCubeIm=np.exp(2.*np.pi*1j*(us*l))*simufunc
            TheoCubeIm=np.exp(2.*np.pi*1j*(dus*l))*theofunc
            nocorrTheoImStack=np.mean(CubeIm,axis=1)
        # this loop is for when func is too big, to limit RAM use
        if loop==True:
            print "in iterator loop"#,func.shape
            fullcorrSimuImStack=np.zeros((Npix),np.complex64)
            nocorrTheoImStack=np.zeros((Npix),np.complex64)
            diagcorrTheoImStack=np.zeros((Npix),np.complex64)
            fullcorrTheoImStack=np.zeros((Npix),np.complex64)
            nocorrSimuImStack=np.zeros((Npix),np.complex64)
            diagcorrSimuImStack=np.zeros((Npix),np.complex64)
            fullcorrSimuImStack=np.zeros((Npix),np.complex64)
            complexcorrSimuImStack=np.zeros((Npix),np.complex64)
            complexcorrTheoImStack=np.zeros((Npix),np.complex64)
            
            print "begin iteration"
            # iterate over pixels
            for ipix in range(Npix):
                print ipix
                nocorrTheoImStack[ipix]=np.mean(theofunc[0,:,0]*np.exp(2.*np.pi*1j*(dus[0,:,0]*l[0,0,ipix])))
                diagcorrTheoImStack[ipix]=np.mean(diagtheofunc[0,:,0]*np.exp(2.*np.pi*1j*(dus[0,:,0]*l[0,0,ipix])))
                fullcorrTheoImStack[ipix]=np.mean(fulltheofunc[0,:,0]*np.exp(2.*np.pi*1j*(dus[0,:,0]*l[0,0,ipix])))
                complexcorrTheoImStack[ipix]=np.mean(complextheofunc[0,:,0]*np.exp(2.*np.pi*1j*(dus[0,:,0]*l[0,0,ipix])))
                nocorrSimuImStack[ipix]=np.var(np.mean(simufunc[:,:,0]*np.exp(2.*np.pi*1j*(us[0,:,0]*l[0,0,ipix])),axis=1))
                diagcorrSimuImStack[ipix]=np.var(np.mean(diagsimufunc[:,:,0]*np.exp(2.*np.pi*1j*(us[0,:,0]*l[0,0,ipix])),axis=1))
                fullcorrSimuImStack[ipix]=np.var(np.mean(fullsimufunc[:,:,0]*np.exp(2.*np.pi*1j*(us[0,:,0]*l[0,0,ipix])),axis=1))
                complexcorrSimuImStack[ipix]=np.var(np.mean(complexsimufunc[:,:,0]*np.exp(2.*np.pi*1j*(us[0,:,0]*l[0,0,ipix])),axis=1))
        # plot cross-sections
        pylab.clf()
        pylab.suptitle("Noise-PSF Cross-Section, m=0")
        pylab.plot(l[0,0,:],nocorrTheoImStack,label="Uncorrected",color="b")
        pylab.plot(l[0,0,:],diagcorrTheoImStack,label="Sensitivity-optimal",color="g")
        pylab.plot(l[0,0,:],fullcorrTheoImStack,label="Artefact-optimal",color="r")
        #pylab.plot(l[0,0,:],complexcorrTheoImStack,label="Complex artefact-optimal",color="c")
        pylab.plot(l[0,0,:],nocorrSimuImStack,alpha=0.7,color="b")
        pylab.plot(l[0,0,:],diagcorrSimuImStack,alpha=0.7,color="g")
        pylab.plot(l[0,0,:],fullcorrSimuImStack,alpha=0.7,color="r")
        #pylab.plot(l[0,0,:],complexcorrSimuImStack,alpha=0.7,color="c")

        pylab.legend()
        pylab.ylabel("Variance in pixel at l [Jy]")
        pylab.xlabel("l [degrees]")
        pylab.savefig("ctime%3.3i-NoisePsfCrossections.png"%self.sigma)
        print "Figure saved in ctime%3.3i-NoisePsfCrossections.png"%self.sigma
        pylab.show()
        
        # return cross-section
        return np.copy(nocorrTheoImStack),np.copy(diagcorrTheoImStack),np.copy(fullcorrTheoImStack),np.copy(nocorrSimuImStack),np.copy(diagcorrSimuImStack),np.copy(fullcorrSimuImStack)



def test(ctime1=800):
    pixels=11
#    inst=SimulGains(MSname="/data/tasse/BOOTES/BOOTES24_SB140-149.2ch8s.ms",sigma_sec=ctime1)
    inst=SimulGains(MSname="/data/etienne.bonnassieux/VINCE//BOOTES24_SB140-149.2ch8s.ms",sigma_sec=ctime1)
    inst.NoisePsfCrossSection()
    stop





    gg_corr=inst.Gains
    # make unweighted images
    imsize=0.4/60 # angular size of image
    nocorrSimuImage,nocorrSimuVarImage,nocorrSimuCovmat=inst.MakeCovMapSimul(residual_gains=gg_corr,sigma=ctime1,diamdeg=imsize,npix=pixels)
    nocorrTheoVarianceImage,nocorrTheoCovmat=inst.MakeCovMapTheoretical(residual_gains=gg_corr,sigma=ctime1,diamdeg=imsize,npix=pixels)
    np.save("ctime%3.3i.nocorr.theo.ImStack"%ctime1,nocorrTheoVarianceImage)
    nocorrTheoVarImage=np.abs(np.mean(nocorrTheoVarianceImage,axis=0))
    # calculate weights
    Cinv=np.linalg.inv(nocorrTheoCovmat)
    fullweights=np.abs(np.sum(Cinv,axis=0))
    fullweights=fullweights/np.mean(fullweights)#*fullweights.shape[0]
    diagweights=np.abs(np.diag(Cinv))
    diagweights=diagweights/np.mean(diagweights)#*diagweights.shape[0]
    print "TEST DIFFERENCE BETWEEN WEIGHTS:",fullweights-diagweights
    # apply weights 
    pylab.save("ctime%3.3i.diagweights"%ctime1,diagweights)
    pylab.save("ctime%3.3i.fullweights"%ctime1,fullweights)
    diag_gg=gg_corr[:,:,0,0]*diagweights
    full_gg=gg_corr[:,:,0,0]*fullweights
    # check that it is cast correctly
#    test1=np.zeros_like(test)
#    for i in range(gg_corr.shape[0]): 
#        for j in range(gg_corr.shape[1]): 
#            test1[i,j]=gg_corr[i,j,0,0]/diagweights[j]
    # recast in original shape
    diag_gg=diag_gg.reshape(gg_corr.shape)
    full_gg=full_gg.reshape(gg_corr.shape)
    # find weighted images!
#    print "test1",nocorrTheoCovmat
    diagSimuImage,diagSimuVarImage,diagSimuCovmat=inst.MakeCovMapSimul(residual_gains=diag_gg,sigma=ctime1,diamdeg=imsize,npix=pixels)
    diagTheoVarianceImage,diagTheoCovmat=inst.MakeCovMapTheoretical(residual_gains=diag_gg,sigma=ctime1,diamdeg=imsize,npix=pixels,weights=diagweights)
#    print "test",nocorrTheoCovmat
#    print "test",diagTheoCovmat
    diagTheoVarImage=np.abs(np.mean(diagTheoVarianceImage,axis=0))
    fullSimuImage,fullSimuVarImage,fullSimuCovmat=inst.MakeCovMapSimul(residual_gains=full_gg,sigma=ctime1,diamdeg=imsize,npix=pixels)
    fullTheoVarianceImage,fullTheoCovmat=inst.MakeCovMapTheoretical(residual_gains=full_gg,sigma=ctime1,diamdeg=imsize,npix=pixels,weights=fullweights)
    fullTheoVarImage=np.abs(np.mean(fullTheoVarianceImage,axis=0))

    # save image nparrays
    np.save("ctime%3.3i.not_corr.theo.VarIm"%ctime1,nocorrTheoVarImage)
    np.save("ctime%3.3i.diagcorr.theo.VarIm"%ctime1,diagTheoVarImage)
    np.save("ctime%3.3i.fullcorr.theo.VarIm"%ctime1,fullTheoVarImage)
#
    np.save("ctime%3.3i.not_corr.simu.VarIm"%ctime1,nocorrSimuVarImage)
    np.save("ctime%3.3i.diagcorr.simu.VarIm"%ctime1,diagSimuVarImage)
    np.save("ctime%3.3i.fullcorr.simu.VarIm"%ctime1,fullSimuVarImage)
#
    np.save("ctime%3.3i.not_corr.simu.ImStack"%ctime1,nocorrSimuImage)
    np.save("ctime%3.3i.diagcorr.simu.ImStack"%ctime1,diagSimuImage)
    np.save("ctime%3.3i.fullcorr.simu.ImStack"%ctime1,fullSimuImage)
#
    np.save("ctime%3.3i.not_corr.simu.covmat"%ctime1,nocorrSimuCovmat)
    np.save("ctime%3.3i.diagcorr.simu.covmat"%ctime1,diagSimuCovmat)
    np.save("ctime%3.3i.fullcorr.simu.covmat"%ctime1,fullSimuCovmat)
#
    np.save("ctime%3.3i.not_corr.theo.covmat"%ctime1,nocorrTheoCovmat)
    np.save("ctime%3.3i.diagcorr.theo.covmat"%ctime1,diagTheoCovmat)
    np.save("ctime%3.3i.fullcorr.theo.covmat"%ctime1,fullTheoCovmat)
#
    np.save("ctime%3.3i.nocorr.vis"%ctime1,gg_corr)
    np.save("ctime%3.3i.diagcorr.vis"%ctime1,diag_gg)
    np.save("ctime%3.3i.fullcorr.vis"%ctime1,full_gg)    

    PlotMaker(ctime1,cmapchoice=None)
#    stop

def PlotMaker(time=0,cmapchoice="gray"):

    # load data
    not_corrTheoCovmatrix         = np.load("ctime%3.3i.not_corr.theo.covmat.npy"%time)
    not_corrSimuCovmatrix         = np.load("ctime%3.3i.not_corr.simu.covmat.npy"%time)
    not_corrTheoVarImage          = np.load("ctime%3.3i.not_corr.theo.VarIm.npy"%time)/np.sum(not_corrTheoCovmatrix)*(not_corrTheoCovmatrix).size
    not_corrSimuVarImage          = np.load("ctime%3.3i.not_corr.simu.VarIm.npy"%time)/np.sum(not_corrSimuCovmatrix)*(not_corrSimuCovmatrix).size
#
    diagcorrTheoCovmatrix         = np.load("ctime%3.3i.diagcorr.theo.covmat.npy"%time)
    diagcorrSimuCovmatrix         = np.load("ctime%3.3i.diagcorr.simu.covmat.npy"%time)
    diagcorrTheoVarImage          = np.load("ctime%3.3i.diagcorr.theo.VarIm.npy"%time)/np.sum(not_corrTheoCovmatrix)*(diagcorrTheoCovmatrix).size
    diagcorrSimuVarImage          = np.load("ctime%3.3i.diagcorr.simu.VarIm.npy"%time)/np.sum(not_corrSimuCovmatrix)*(diagcorrSimuCovmatrix).size
#
    fullcorrTheoCovmatrix         = np.load("ctime%3.3i.fullcorr.theo.covmat.npy"%time)
    fullcorrSimuCovmatrix         = np.load("ctime%3.3i.fullcorr.simu.covmat.npy"%time)
    fullcorrTheoVarImage          = np.load("ctime%3.3i.fullcorr.theo.VarIm.npy"%time)/np.sum(not_corrTheoCovmatrix)*(fullcorrTheoCovmatrix).size
    fullcorrSimuVarImage          = np.load("ctime%3.3i.fullcorr.simu.VarIm.npy"%time)/np.sum(not_corrSimuCovmatrix)*(fullcorrSimuCovmatrix).size

    ##################
    ### MAKE PLOTS ###
    ##################
    pylab.clf()
    pylab.cla()
    pylab.close()
    pylab.figure(figsize=(12,8))
    pylab.suptitle(r"Noise-PSF for $\sigma_\tau=%3.3is$"%time)
    fig0=pylab.subplot(2,3,1)
    pylab.title(r"Uncorrected Simulated")
    pylab.ylabel("Dec [arcsec]")
    pylab.xlabel("RA [arcsec]")
    im0=pylab.imshow(np.abs(not_corrSimuVarImage),interpolation="nearest",extent=[-12,12,-12,12],cmap=cmapchoice)
    #,cmap="gray",vmin=0,vmax=1.05,
    divider0 = make_axes_locatable(fig0)
    cax0 = divider0.append_axes("right", size="5%", pad=0.05)
    pylab.colorbar(im0, cax=cax0)
    fig0=pylab.subplot(2,3,2)
    pylab.ylabel("Dec [arcsec]")
    pylab.xlabel("RA [arcsec]")
    pylab.title(r"Diag-corrected Simulated")
    im0=pylab.imshow(np.abs(diagcorrSimuVarImage),interpolation="nearest",extent=[-12,12,-12,12],cmap=cmapchoice)
    divider0 = make_axes_locatable(fig0)
    cax0 = divider0.append_axes("right", size="5%", pad=0.05)
    pylab.colorbar(im0, cax=cax0)
    fig0=pylab.subplot(2,3,3)
    pylab.ylabel("Dec [arcsec]")
    pylab.xlabel("RA [arcsec]")
    pylab.title(r"Full-corrected Simulated")
    im0=pylab.imshow(np.abs(fullcorrSimuVarImage),interpolation="nearest",extent=[-12,12,-12,12],cmap=cmapchoice)
    divider0 = make_axes_locatable(fig0)
    cax0 = divider0.append_axes("right", size="5%", pad=0.05)
    pylab.colorbar(im0, cax=cax0)
    fig0=pylab.subplot(2,3,4)
    pylab.title(r"Uncorrected Theoretical")
    pylab.ylabel("Dec [arcsec]")
    pylab.xlabel("RA [arcsec]")
    im1=pylab.imshow(np.abs(not_corrTheoVarImage),interpolation="nearest",cmap=cmapchoice,vmin=0,vmax=1.05*np.max(np.abs(not_corrTheoVarImage)),extent=[-12,12,-12,12])
    #,cmap="gray",vmin=0,vmax=1.05,
    divider0 = make_axes_locatable(fig0)
    cax0 = divider0.append_axes("right", size="5%", pad=0.05)
    pylab.colorbar(im1, cax=cax0)
    fig0=pylab.subplot(2,3,5)
    pylab.title(r"Diag-Corrected Theoretical")
    pylab.ylabel("Dec [arcsec]")
    pylab.xlabel("RA [arcsec]")
    im1=pylab.imshow(np.abs(diagcorrTheoVarImage),interpolation="nearest",cmap=cmapchoice,vmin=0,vmax=1.05*np.max(np.abs(diagcorrTheoVarImage)),extent=[-12,12,-12,12])
    #,cmap="gray",vmin=0,vmax=1.05,
    divider0 = make_axes_locatable(fig0)
    cax0 = divider0.append_axes("right", size="5%", pad=0.05)
    pylab.colorbar(im1, cax=cax0)
    fig0=pylab.subplot(2,3,6)
    pylab.title(r"Full-Corrected Theoretical")
    pylab.ylabel("Dec [arcsec]")
    pylab.xlabel("RA [arcsec]")
    im0=pylab.imshow(np.abs(diagcorrTheoVarImage),interpolation="nearest",cmap=cmapchoice,vmin=0,vmax=1.05*np.max(np.abs(fullcorrTheoVarImage)),extent=[-12,12,-12,12])
    #,cmap="gray",vmin=0,vmax=1.05,
    divider0 = make_axes_locatable(fig0)
    cax0 = divider0.append_axes("right", size="5%", pad=0.05)
    pylab.colorbar(im0, cax=cax0)
    #pylab.show()
    pylab.tight_layout()#pad=1.5)
    pylab.savefig("SimulationsCorrectionResultsCtime%3.3i.png"%time)
    print "Saved figured in file SimulationsCorrectionResultsCtime%3.3i.png"%time
    pylab.clf()
    pylab.cla()
    pylab.close("all")

def lambdaplot(ctime1=0):
    pixels=1009
    testval=5
    inst=SimulGains(MSname="/data/tasse/BOOTES/BOOTES24_SB140-149.2ch8s.ms",sigma_sec=ctime1,timesteps=testval)
    residuals=inst.Gains
    # make unweighted images
    imsize=0.4/60. # angular size of image
#    nocorrSimuImage,nocorrSimuVarImage,nocorrSimuCovmat=inst.MakeCovMapSimul(residual_gains=residuals,sigma=ctime1,diamdeg=imsize,npix=pixels)
    nocorrTheoVarianceImage,nocorrTheoCovmat=inst.MakeCovMapTheoretical(residual_gains=residuals,sigma=ctime1,diamdeg=imsize,npix=pixels)
    nocorrTheoVarImage=np.abs(np.mean(nocorrTheoVarianceImage,axis=0))/np.mean(nocorrTheoCovmat)
    # calculate weights
    Cinv=np.linalg.inv(nocorrTheoCovmat)
    fullweights=np.abs(np.sum(Cinv,axis=0))#/np.abs(np.mean(np.sum(Cinv,axis=0)))
    diagweights=np.abs(np.diag(Cinv))#/np.abs(np.mean(np.diag(Cinv)))
    Lambda=np.arange(200)/20.
#    Lambda[0]=0
#    Lambda[1]=500000
    minvals=[]
    maxvals=[]
    initMin=[]
    initMax=[]
#    stop
    for i in Lambda:
#        print "in iteration:",i
#        print np.abs(np.round(nocorrTheoCovmat,3))
#        corrTheoVarImage,corrTheoCovmat,corrTheoVarImage=0,0,0
        weights=i*fullweights+diagweights
        weights=weights/np.sum(weights)*weights.shape[0]
        flagweights=np.ones(weights.shape)#np.array([0,1,1,1,1])
#        print "weight:",weights
        corr_residuals=np.copy((np.copy(residuals[:,:,0,0])*weights).reshape(residuals.shape))
#        print residuals-corr_residuals
        corrSimuImage,corrSimuVarImage,corrSimuCovmat=inst.MakeCovMapSimul(residual_gains=corr_residuals,sigma=ctime1,diamdeg=imsize,npix=pixels)
        corrSimuImage=corrSimuImage/np.sum(nocorrSimuCovmat)*(nocorrSimuCovmat).size
        corrTheoVarImage,corrTheoCovmat=inst.MakeCovMapTheoretical(residual_gains=residuals,sigma=ctime1,diamdeg=imsize,npix=pixels,weights=weights)
        corrTheoVarImage=corrTheoVarImage/np.mean(nocorrTheoCovmat)#*((inst.covar.Cov).size)#**2
        minvals.append(np.min(np.abs(corrTheoVarImage)))
        maxvals.append(np.max(np.abs(corrTheoVarImage)))
        print "in loop, iter %i"%i,np.mean(weights)#np.sum(nocorrTheoCovmat-inst.covar.Cov)
        print maxvals[-1],minvals[-1]
        print np.mean(corrSimuCovmat)/np.mean(nocorrSimuCovmat)
        # make nocorr min, max arrays
        initMin.append(np.min(np.abs(nocorrTheoVarImage)))
        initMax.append(np.max(np.abs(nocorrTheoVarImage)))
#    stop
    pylab.clf()
    pylab.suptitle(r"$\sigma_t$: %3.3i"%ctime1)
    pylab.plot(Lambda,minvals,label="MinCorrected Var")
    pylab.plot(Lambda,maxvals,label="Max Corrected Var")
    pylab.plot(Lambda,initMax,"--",label="Max Uncorrected Var")
    pylab.plot(Lambda,initMin,"--",label="Min Uncorrected Var")
    pylab.ylabel("Variance")
    pylab.xlabel(r"$\lambda$")
    pylab.legend()
    pylab.savefig("ctime%3.3i.png.CorrectionMinMax.Lambdaplot.png"%ctime1)
    pylab.clf()
    print "Saved figure in file ctime%3.3i.png.CorrectionMinMax.Lambdaplot.png"%ctime1
#    pylab.show()
#    stop

    
def AddNoiseToData(MSName,colname,ctime):
    #open MS
    ms=table(MSName,readonly=True)
    ants=table(ms.getkeyword("ANTENNA"))
    nAnt=len(ants.getcol("NAME"))
    ants.close()
    data=ms.getcol(colname)
    Times=ms.getcol("TIME")
    nbl=np.where(Times==Times[0])[0].size
    nt=data.shape[0]/nbl
    nChan=data.shape[1]
    nPola=data.shape[2]
    ms.close()
    # find number of baselines
    inst=SimulGains(MSname=MSName,sigma_sec=ctime,realisations=nbl*nChan*nPola)
#    data=data.reshape(nt,nbl,nChan,nPola)
    # add noise to data; inst.Gains contains time-correlated noise
    data=data+np.swapaxes(inst.Gains[:,:,0,0],0,1).reshape(nt*nbl,nChan,nPola)
    ms=table(MSName,readonly=False)
    ms.putcol(colname,data)
    ms.close()

if __name__=="__main__":
    test()
    
