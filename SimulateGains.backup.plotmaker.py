import numpy as np
from pyrap.tables import table
import pylab
import scipy.ndimage
import ClassSampleMachine
from mpl_toolkits.axes_grid1 import make_axes_locatable

class SimulGains:
    def __init__(self,MSname,ant1=0,ant2=55,sigma_sec=0,realisations=2000,timeblock=1,CovInit=True,TimePeriodicity=100):
        # open measurement set
        self.MS=table(MSname)
        # define antennas
        self.Antenna=table(MSname+"/ANTENNA")
        A0=self.MS.getcol("ANTENNA1")
        A1=self.MS.getcol("ANTENNA2")
        # define single baseline
        self.baseline=np.where((A0==ant1)&(A1==ant2))[0]
        # get UV-tracks
        UV1=self.MS.getcol("UVW")[self.baseline].T
        self.UV=UV1[:,:]
        # convert sigma(seconds) into sigma(time-pixels)
        if sigma_sec==0:
            sigma=0.00000001
        else:
            dt=float(self.MS.getcol("INTERVAL")[0])*len(UV1[0,:])/len(self.UV[0,:])
            sigma=sigma_sec/dt
            if sigma<0.00000001:
                sigma=0.00000001
# debug: only use 1 out of 10 timesteps
#        UV1=self.MS.getcol("UVW")[self.baseline].T
#        self.UV=UV1[:,::10]
# end debug
        self.nrow=len(self.UV[0])
        # exit gracefully
        self.MS.close()
        self.Antenna.close()
        self.NSample=realisations
        # initiate covariance matrix
        if CovInit==True:
            self.covar=ClassSampleMachine.ClassSampleMachine(NPoints=self.nrow,sigma=sigma,T=TimePeriodicity)
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
                    print "so far, at: %f / %f, %f / %f"%(ipix,Npix,jpix,Npix)
                    ImStack[:,ipix,jpix]=np.mean(func[:,:,0,0]*np.exp(2.*np.pi*1j*(u[0,:,0,0]*l[0,0,ipix,jpix]+v[0,:,0,0]*m[0,0,ipix,jpix])),axis=1)
        # return outputs and exit gracefully
        return ImStack

    def MakeCovMapSimul(self,residual_gains,sigma=0,diamdeg=1.,npix=101):
        print "in simulated covmap"
        # find UV tracks
        us,vs,_=self.UV
        # make image
        image=self.DFT(residual_gains,us,vs,DiamDeg=diamdeg,Npix=npix,loop=False)
        varimage=np.var(image,axis=0)
        ax1=residual_gains[:,:,0,0]
        nmatrix=np.dot((ax1.conj()).T,ax1)/self.NSample
        return image,varimage,nmatrix
 
    def AnalyticCovMatrix(self,sigma=0):
        CTh=self.covar.Cov
        return CTh

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
        nmatrix=self.AnalyticCovMatrix(sigma=sigma)
        if weights == None:
            weights=np.ones((nmatrix.shape[1]))
        #            nmatrix=np.dot(np.dot(weights,nmatrix),weights.T)
        for j in range(nmatrix.shape[0]):
            for i in range(nmatrix.shape[1]):
                nmatrix[j,i]=nmatrix[j,i]*(weights[j]*weights[i])
        dudvdata=nmatrix.flatten()
        dudvdata=dudvdata.reshape(1,len(dudvdata),1,1)
        # make image
        image=self.DFT(dudvdata,dus,dvs,DiamDeg=diamdeg,Npix=npix,loop=True)
#        stop
        print "peak:",np.max(image)*(nmatrix.flatten()).size/np.sum(nmatrix)
#        image=image*(nmatrix.flatten()).size/np.sum(nmatrix)
        return image,nmatrix

def test(ctime1=0):
    pixels=31
    inst=SimulGains(MSname="/data/tasse/BOOTES/BOOTES24_SB140-149.2ch8s.ms",sigma_sec=ctime1)
    gg_corr=inst.Gains
    # make unweighted images
    imsize=0.4/60 # angular size of image
    nocorrSimuImage,nocorrSimuVarImage,nocorrSimuCovmat=inst.MakeCovMapSimul(residual_gains=gg_corr,sigma=ctime1,diamdeg=imsize,npix=pixels)
    nocorrTheoVarianceImage,nocorrTheoCovmat=inst.MakeCovMapTheoretical(residual_gains=gg_corr,sigma=ctime1,diamdeg=imsize,npix=pixels)
    np.save("ctime%3.3i.nocorr.theo.ImStack"%ctime1,nocorrTheoVarianceImage)
    nocorrTheoVarImage=np.abs(np.mean(nocorrTheoVarianceImage,axis=0))
    # calculate weights
    Cinv=np.linalg.inv(nocorrTheoCovmat)
    fullweights=np.abs(np.sum(Cinv,axis=0))/np.abs(np.sum(Cinv))
    diagweights=np.abs(np.diag(Cinv))/np.abs(np.sum(np.diag(Cinv)))
    print "TEST DIFFERENCE BETWEEN WEIGHTS:",fullweights-diagweights
    # apply weights 
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
    diagSimuImage,diagSimuVarImage,diagSimuCovmat=inst.MakeCovMapSimul(residual_gains=diag_gg,sigma=ctime1,diamdeg=imsize,npix=pixels)
    diagTheoVarianceImage,diagTheoCovmat=inst.MakeCovMapTheoretical(residual_gains=diag_gg,sigma=ctime1,diamdeg=imsize,npix=pixels,weights=diagweights)
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

#    stop

def PlotMaker(cmapchoice="gray"):

    # load 0
    not_corrTheoCovmatrix000         = np.load("ctime000.not_corr.theo.covmat.npy")
    not_corrSimuCovmatsim000         = np.load("ctime000.not_corr.simu.covmat.npy")
    not_corrTheoVarImage000          = np.load("ctime000.not_corr.theo.VarIm.npy")/np.sum(not_corrTheoCovmatsim000)*(not_corrTheoCovmatsim000).size
    not_corrSimuVarImage000          = np.load("ctime000.not_corr.simu.VarIm.npy")/np.sum(not_corrSimuCovmatsim000)*(not_corrSimuCovmatsim000).size
#
    diagcorrTheoCovmatrix000         = np.load("ctime000.diagcorr.theo.covmat.npy")
    diagcorrSimuCovmatsim000         = np.load("ctime000.diagcorr.simu.covmat.npy")
    diagcorrTheoVarImage000          = np.load("ctime000.diagcorr.theo.VarIm.npy")/np.sum(diagcorrTheoCovmatsim000)*(diagcorrTheoCovmatsim000).size
    diagcorrSimuVarImage000          = np.load("ctime000.diagcorr.simu.VarIm.npy")/np.sum(diagcorrSimuCovmatsim000)*(diagcorrSimuCovmatsim000).size
#
    fullcorrTheoCovmatrix000         = np.load("ctime000.fullcorr.theo.covmat.npy")
    fullcorrSimuCovmatsim000         = np.load("ctime000.fullcorr.simu.covmat.npy")
    fullcorrTheoVarImage000          = np.load("ctime000.fullcorr.theo.VarIm.npy")/np.sum(fullcorrTheoCovmatsim000)*(fullcorrTheoCovmatsim000).size
    fullcorrSimuVarImage000          = np.load("ctime000.fullcorr.simu.VarIm.npy")/np.sum(fullcorrSimuCovmatsim000)*(fullcorrSimuCovmatsim000).size
    # load 400
    not_corrTheoCovmatrix400         = np.load("ctime400.not_corr.theo.covmat.npy")
    not_corrSimuCovmatsim400         = np.load("ctime400.not_corr.simu.covmat.npy")
    not_corrTheoVarImage400          = np.load("ctime400.not_corr.theo.VarIm.npy")/np.sum(not_corrTheoCovmatsim400)*(not_corrTheoCovmatsim400).size
    not_corrSimuVarImage400          = np.load("ctime400.not_corr.simu.VarIm.npy")/np.sum(not_corrSimuCovmatsim400)*(not_corrSimuCovmatsim400).size
#
    diagcorrTheoCovmatrix400         = np.load("ctime400.diagcorr.theo.covmat.npy")
    diagcorrSimuCovmatsim400         = np.load("ctime400.diagcorr.simu.covmat.npy")
    diagcorrTheoVarImage400          = np.load("ctime400.diagcorr.theo.VarIm.npy")/np.sum(diagcorrTheoCovmatsim400)*(diagcorrTheoCovmatsim400).size
    diagcorrSimuVarImage400          = np.load("ctime400.diagcorr.simu.VarIm.npy")/np.sum(diagcorrSimuCovmatsim400)*(diagcorrSimuCovmatsim400).size
#
    fullcorrTheoCovmatrix400         = np.load("ctime400.fullcorr.theo.covmat.npy")
    fullcorrSimuCovmatsim400         = np.load("ctime400.fullcorr.simu.covmat.npy")
    fullcorrTheoVarImage400          = np.load("ctime400.fullcorr.theo.VarIm.npy")/np.sum(fullcorrTheoCovmatsim400)*(fullcorrTheoCovmatsim400).size
    fullcorrSimuVarImage400          = np.load("ctime400.fullcorr.simu.VarIm.npy")/np.sum(fullcorrSimuCovmatsim400)*(fullcorrSimuCovmatsim400).size
    # load 800
    not_corrTheoCovmatrix800         = np.load("ctime800.not_corr.theo.covmat.npy")
    not_corrSimuCovmatsim800         = np.load("ctime800.not_corr.simu.covmat.npy")
    not_corrTheoVarImage800          = np.load("ctime800.not_corr.theo.VarIm.npy")/np.sum(not_corrTheoCovmatsim800)*(not_corrTheoCovmatsim800).size
    not_corrSimuVarImage800          = np.load("ctime800.not_corr.simu.VarIm.npy")/np.sum(not_corrSimuCovmatsim800)*(not_corrSimuCovmatsim800).size
#
    diagcorrTheoCovmatrix800         = np.load("ctime800.diagcorr.theo.covmat.npy")
    diagcorrSimuCovmatsim800         = np.load("ctime800.diagcorr.simu.covmat.npy")
    diagcorrTheoVarImage800          = np.load("ctime800.diagcorr.theo.VarIm.npy")/np.sum(diagcorrTheoCovmatsim800)*(diagcorrTheoCovmatsim800).size
    diagcorrSimuVarImage800          = np.load("ctime800.diagcorr.simu.VarIm.npy")/np.sum(diagcorrSimuCovmatsim800)*(diagcorrSimuCovmatsim800).size
#
    fullcorrTheoCovmatrix800         = np.load("ctime800.fullcorr.theo.covmat.npy")
    fullcorrSimuCovmatsim800         = np.load("ctime800.fullcorr.simu.covmat.npy")
    fullcorrTheoVarImage800          = np.load("ctime800.fullcorr.theo.VarIm.npy")/np.sum(fullcorrTheoCovmatsim800)*(fullcorrTheoCovmatsim800).size
    fullcorrSimuVarImage800          = np.load("ctime800.fullcorr.simu.VarIm.npy")/np.sum(fullcorrSimuCovmatsim800)*(fullcorrSimuCovmatsim800).size

    ###############################
    ### MAKE PLOT FOR CTIME=000 ###
    ###############################
    pylab.clf()
    pylab.cla()
    pylab.close()
    pylab.figure(4124,figsize=(12,8))
    pylab.suptitle(r"Noise-PSF for $\sigma_\tau=0s$")
    fig0=pylab.subplot(2,3,1)
    pylab.title(r"Uncorrected Simulated")
    pylab.ylabel("Dec [arcsec]")
    pylab.xlabel("RA [arcsec]")
    im0=pylab.imshow(np.abs(not_corrSimuVarImage000),interpolation="nearest",extent=[-12,12,-12,12],cmap=cmapchoice)
    #,cmap="gray",vmin=0,vmax=1.05,
    divider0 = make_axes_locatable(fig0)
    cax0 = divider0.append_axes("right", size="5%", pad=0.05)
    pylab.colorbar(im0, cax=cax0)
    fig0=pylab.subplot(2,3,2)
    pylab.ylabel("Dec [arcsec]")
    pylab.xlabel("RA [arcsec]")
    pylab.title(r"Diag-corrected Simulated")
    im0=pylab.imshow(np.abs(diagcorrSimuVarImage000),interpolation="nearest",extent=[-12,12,-12,12],cmap=cmapchoice)
    divider0 = make_axes_locatable(fig0)
    cax0 = divider0.append_axes("right", size="5%", pad=0.05)
    pylab.colorbar(im0, cax=cax0)
    fig0=pylab.subplot(2,3,3)
    pylab.ylabel("Dec [arcsec]")
    pylab.xlabel("RA [arcsec]")
    pylab.title(r"Full-corrected Simulated")
    im0=pylab.imshow(np.abs(fullcorrSimuVarImage000),interpolation="nearest",extent=[-12,12,-12,12],cmap=cmapchoice)
    divider0 = make_axes_locatable(fig0)
    cax0 = divider0.append_axes("right", size="5%", pad=0.05)
    pylab.colorbar(im0, cax=cax0)
    fig0=pylab.subplot(2,3,4)
    pylab.title(r"Uncorrected Theoretical")
    pylab.ylabel("Dec [arcsec]")
    pylab.xlabel("RA [arcsec]")
    im1=pylab.imshow(np.abs(not_corrTheoVarImage000),interpolation="nearest",cmap=cmapchoice,vmin=0,vmax=1.05,extent=[-12,12,-12,12])
    #,cmap="gray",vmin=0,vmax=1.05,
    divider0 = make_axes_locatable(fig0)
    cax0 = divider0.append_axes("right", size="5%", pad=0.05)
    pylab.colorbar(im1, cax=cax0)
    fig0=pylab.subplot(2,3,5)
    pylab.title(r"Diag-Corrected Theoretical")
    pylab.ylabel("Dec [arcsec]")
    pylab.xlabel("RA [arcsec]")
    im1=pylab.imshow(np.abs(diagcorrTheoVarImage000),interpolation="nearest",cmap=cmapchoice,vmin=0,vmax=1.05,extent=[-12,12,-12,12])
    #,cmap="gray",vmin=0,vmax=1.05,
    divider0 = make_axes_locatable(fig0)
    cax0 = divider0.append_axes("right", size="5%", pad=0.05)
    pylab.colorbar(im1, cax=cax0)
    fig0=pylab.subplot(2,3,6)
    pylab.title(r"Full-Corrected Theoretical")
    pylab.ylabel("Dec [arcsec]")
    pylab.xlabel("RA [arcsec]")
    im0=pylab.imshow(np.abs(diagcorrTheoVarImage000),interpolation="nearest",cmap=cmapchoice,vmin=0,vmax=1.05,extent=[-12,12,-12,12])
    #,cmap="gray",vmin=0,vmax=1.05,
    divider0 = make_axes_locatable(fig0)
    cax0 = divider0.append_axes("right", size="5%", pad=0.05)
    pylab.colorbar(im0, cax=cax0)
    #pylab.show()
    pylab.tight_layout()#pad=1.5)
    pylab.savefig("SimulationsCorrectionResultsCtime000.png")

    ###############################
    ### MAKE PLOT FOR CTIME=400 ###
    ###############################
    pylab.clf()
    pylab.figure(463,figsize=(12,8))
    pylab.suptitle(r"Noise-PSF for $\sigma_\tau=400s$")
    fig0=pylab.subplot(2,3,1)
    pylab.title(r"Uncorrected Simulated")
    pylab.ylabel("Dec [arcsec]")
    pylab.xlabel("RA [arcsec]")
    im0=pylab.imshow(np.abs(simulated400),interpolation="nearest",extent=[-12,12,-12,12],cmap=cmapchoice)
    #,cmap="gray",vmin=0,vmax=1.05,
    divider0 = make_axes_locatable(fig0)
    cax0 = divider0.append_axes("right", size="5%", pad=0.05)
    pylab.colorbar(im0, cax=cax0)
    fig0=pylab.subplot(2,3,2)
    pylab.ylabel("Dec [arcsec]")
    pylab.xlabel("RA [arcsec]")
    pylab.title(r"Diag-corrected Simulated")
    im0=pylab.imshow(np.abs(diagsimulated400),interpolation="nearest",extent=[-12,12,-12,12],cmap=cmapchoice)
    divider0 = make_axes_locatable(fig0)
    cax0 = divider0.append_axes("right", size="5%", pad=0.05)
    pylab.colorbar(im0, cax=cax0)
    fig0=pylab.subplot(2,3,3)
    pylab.ylabel("Dec [arcsec]")
    pylab.xlabel("RA [arcsec]")
    pylab.title(r"Full-corrected Simulated")
    im0=pylab.imshow(np.abs(fullsimulated400),interpolation="nearest",extent=[-12,12,-12,12],cmap=cmapchoice)
    divider0 = make_axes_locatable(fig0)
    cax0 = divider0.append_axes("right", size="5%", pad=0.05)
    pylab.colorbar(im0, cax=cax0)
    fig0=pylab.subplot(2,3,4)
    pylab.title(r"Uncorrected Theoretical")
    pylab.ylabel("Dec [arcsec]")
    pylab.xlabel("RA [arcsec]")
    im0=pylab.imshow(np.abs(predicted400),interpolation="nearest",extent=[-12,12,-12,12],cmap=cmapchoice)
    #,cmap=cmapchoice,vmin=0,vmax=1.05,
    divider0 = make_axes_locatable(fig0)
    cax0 = divider0.append_axes("right", size="5%", pad=0.05)
    pylab.colorbar(im0, cax=cax0)
    fig0=pylab.subplot(2,3,5)
    pylab.title(r"Diag-Corrected Theoretical")
    pylab.ylabel("Dec [arcsec]")
    pylab.xlabel("RA [arcsec]")
    im0=pylab.imshow(np.abs(diagpredicted400),interpolation="nearest",extent=[-12,12,-12,12],cmap=cmapchoice)
    #,cmap=cmapchoice,vmin=0,vmax=1.05,
    divider0 = make_axes_locatable(fig0)
    cax0 = divider0.append_axes("right", size="5%", pad=0.05)
    pylab.colorbar(im0, cax=cax0)
    fig0=pylab.subplot(2,3,6)
    pylab.title(r"Full-Corrected Theoretical")
    pylab.ylabel("Dec [arcsec]")
    pylab.xlabel("RA [arcsec]")
    im0=pylab.imshow(np.abs(fullpredicted400),interpolation="nearest",extent=[-12,12,-12,12],cmap=cmapchoice)
    #,cmap=cmapchoice,vmin=0,vmax=1.05,
    divider0 = make_axes_locatable(fig0)
    cax0 = divider0.append_axes("right", size="5%", pad=0.05)
    pylab.colorbar(im0, cax=cax0)
    #pylab.show()
    pylab.tight_layout()#pad=1.5)
    pylab.savefig("SimulationsCorrectionResultsCtime400.png")

    ###############################
    ### MAKE PLOT FOR CTIME=800 ###
    ###############################
    pylab.clf()
    pylab.figure(613,figsize=(12,8))
    pylab.suptitle(r"Noise-PSF for $\sigma_\tau=800s$")
    fig0=pylab.subplot(2,3,1)
    pylab.title(r"Uncorrected Simulated")
    pylab.ylabel("Dec [arcsec]")
    pylab.xlabel("RA [arcsec]")
    im0=pylab.imshow(np.abs(simulated800),interpolation="nearest",extent=[-12,12,-12,12],cmap=cmapchoice)
    #,cmap=cmapchoice,vmin=0,vmax=1.05,
    divider0 = make_axes_locatable(fig0)
    cax0 = divider0.append_axes("right", size="5%", pad=0.05)
    pylab.colorbar(im0, cax=cax0)
    fig0=pylab.subplot(2,3,2)
    pylab.ylabel("Dec [arcsec]")
    pylab.xlabel("RA [arcsec]")
    pylab.title(r"Diag-corrected Simulated")
    im0=pylab.imshow(np.abs(diagsimulated800),interpolation="nearest",extent=[-12,12,-12,12],cmap=cmapchoice)
    divider0 = make_axes_locatable(fig0)
    cax0 = divider0.append_axes("right", size="5%", pad=0.05)
    pylab.colorbar(im0, cax=cax0)
    fig0=pylab.subplot(2,3,3)
    pylab.ylabel("Dec [arcsec]")
    pylab.xlabel("RA [arcsec]")
    pylab.title(r"Full-corrected Simulated")
    im0=pylab.imshow(np.abs(fullsimulated800),interpolation="nearest",extent=[-12,12,-12,12],cmap=cmapchoice)
    divider0 = make_axes_locatable(fig0)
    cax0 = divider0.append_axes("right", size="5%", pad=0.05)
    pylab.colorbar(im0, cax=cax0)
    fig0=pylab.subplot(2,3,4)
    pylab.title(r"Uncorrected Theoretical")
    pylab.ylabel("Dec [arcsec]")
    pylab.xlabel("RA [arcsec]")
    im0=pylab.imshow(np.abs(predicted800),interpolation="nearest",extent=[-12,12,-12,12],cmap=cmapchoice)
    #,cmap=cmapchoice,vmin=0,vmax=1.05,
    divider0 = make_axes_locatable(fig0)
    cax0 = divider0.append_axes("right", size="5%", pad=0.05)
    pylab.colorbar(im0, cax=cax0)
    fig0=pylab.subplot(2,3,5)
    pylab.title(r"Diag-Corrected Theoretical")
    pylab.ylabel("Dec [arcsec]")
    pylab.xlabel("RA [arcsec]")
    im0=pylab.imshow(np.abs(diagpredicted800),interpolation="nearest",extent=[-12,12,-12,12],cmap=cmapchoice)
    #,cmap=cmapchoice,vmin=0,vmax=1.05,
    divider0 = make_axes_locatable(fig0)
    cax0 = divider0.append_axes("right", size="5%", pad=0.05)
    pylab.colorbar(im0, cax=cax0)
    fig0=pylab.subplot(2,3,6)
    pylab.title(r"Full-Corrected Theoretical")
    pylab.ylabel("Dec [arcsec]")
    pylab.xlabel("RA [arcsec]")
    im0=pylab.imshow(np.abs(fullpredicted800),interpolation="nearest",extent=[-12,12,-12,12],cmap=cmapchoice)
    #,cmap=cmapchoice,vmin=0,vmax=1.05,
    divider0 = make_axes_locatable(fig0)
    cax0 = divider0.append_axes("right", size="5%", pad=0.05)
    pylab.colorbar(im0, cax=cax0)
    #pylab.show()
    pylab.tight_layout()#pad=1.5)
    pylab.savefig("SimulationsCorrectionResultsCtime800.png")
    pylab.clf()
    pylab.cla()
    pylab.close("all")
    
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
