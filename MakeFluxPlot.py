import numpy as np
import pylab
from scipy.optimize import curve_fit
import pyregion
from astropy.io import fits
from matplotlib import rc
#rc('font',**{'family':'serif','serif':['Palatino']})
#rc('text', usetex=True)
from matplotlib import gridspec
from matplotlib import ticker
from matplotlib import font_manager
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import astropy.units as u
from scipy.optimize import curve_fit
import emcee
from scipy.special import erf,gamma
import astropy.constants as const
import math as m
import astropy.cosmology as astrocosm
import pickle


### pickle functions
def save_dict(filename_,di_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di

### set frequency units for entire program to be GHz.
global nu0
nu0=1.e9
np.random.seed(420)

###############################################################################################
###############################################################################################
###################################### SCIENCE FUNCTIONS ######################################
###############################################################################################
###############################################################################################

### read flux values from images
def GetFluxValues(images,regs,fluxsigmacutoff=5.):
    """ Program to read flux values from fits files.
        images is a list of .fits images to read.
        regs is the set of region files to analyse. lobes vs hotspot hardcoded.
        fluxsigmacutoff is the flux sigma threshold, in units of std, calculated in the function.
        beam area is currently hardcoded. 
    """
    DicoOutput={}
    DicoOutput["Freq"]=[]
    for reg in regs:
        DicoOutput[reg.split(".reg")[0]]={}
        DicoOutput[reg.split(".reg")[0]]["FLUX"]=[]
        DicoOutput[reg.split(".reg")[0]]["FLUX_ERR"]=[]
    # get WCS from thing. ffs
    HDUfitsfile=fits.open("SpecIndexMapsPB/LOFAR-5GHz.fits")[0]
    # WCS requires 8 characters in these fields, so control loop for them
    if len(HDUfitsfile.header["CTYPE1"])!=8:
        HDUfitsfile.header["CTYPE1"]="DEC---RA"
    if len(HDUfitsfile.header["CTYPE2"])!=8:
        HDUfitsfile.header["CTYPE2"]="DEC--SIN"
    for image in images:
        fitsfile=fits.open(image)
        header=fitsfile[0].header
        # get beam area
        try:
            if "BMAJ" in header.keys():
                bmaj=header["BMAJ"]/(2*np.sqrt(2*np.log(2))) # convert from FWHM to sigma
            else:
                print("BMAJ not found in fits header. Attempting to read from history.")
                for head in header["HISTORY"]:
                    if "bmaj" in head.lower():
                        terms=head.lower().split()
                        for i,val in enumerate(terms):
                            if "bmaj" in val:
                                bmaj=float(terms[i+1])
            if "BMIN" in header.keys():
                bmin=header["BMIN"]/(2*np.sqrt(2*np.log(2)))
            else:
                print("BMIN not found in fits header. Attempting to read from history.")
                for head in header["HISTORY"]:
                    if "bmin" in head.lower():
                        terms=head.lower().split()
                        for i,val in enumerate(terms):
                            if "bmin" in val:
                                bmin=float(terms[i+1])
        except:
            print("resolution not given as bmin, bmax. damn you for this. put it in your header as BMIN BMAX.")
        # get pixel area
        cr1=header["CDELT1"]
        cr2=header["CDELT2"]
        pixarea=np.abs(cr1*cr2)    # take absolute values bcos cdelt can be negative
        beamarea=2*np.pi*bmaj*bmin # beam is 2D gaussian, area is (gaussian normalisation)**2
        beam2pix=pixarea/beamarea  # Jy/bm * beam2pix = Jy/pix
        d=fits.open(image)[0].data[0,0,:,:]*beam2pix
        DicoOutput["Freq"].append(fits.open(image)[0].header["CRVAL3"])
        sigmathresh=fluxsigmacutoff*np.std(d)
        nhotvals=0
        shotvals=0
        for reg in regs:
            keyval=reg.split(".reg")[0]
            region=pyregion.open(reg)
            for regn in region:
                regmask=pyregion.ShapeList([regn]).get_mask(hdu=HDUfitsfile)
            vals=d[regmask]
            vals=vals[vals>sigmathresh]
            DicoOutput[keyval]["Area [\"]"]=np.sum(regmask)*pixarea*3600**2
            if keyval=="Nhotspot":
                flx=np.sum(vals)
                DicoOutput[keyval]["FLUX"].append(flx)
                DicoOutput[keyval]["FLUX_ERR"].append(0.15*flx)
                nhotvals=flx
            if keyval=="Nlobe":
                flx=np.sum(vals)-nhotvals
                DicoOutput[keyval]["FLUX"].append(flx)
                DicoOutput[keyval]["FLUX_ERR"].append(0.15*flx)
            if keyval=="Shotspot":
                flx=np.sum(vals)
                DicoOutput[keyval]["FLUX"].append(flx)
                DicoOutput[keyval]["FLUX_ERR"].append(0.15*flx)
                shotvals=flx
            if keyval=="Slobe":
                flx=np.sum(vals)-shotvals
                DicoOutput[keyval]["FLUX"].append(flx)
                DicoOutput[keyval]["FLUX_ERR"].append(0.15*flx)
        print("Done with %s"%image)
    DicoOutput["Freq"]=np.array(DicoOutput["Freq"])
    for reg in regs:
        keyval=reg.split(".reg")[0]
        DicoOutput[keyval]["FLUX"]=np.array(DicoOutput[keyval]["FLUX"])
        DicoOutput[keyval]["FLUX_ERR"]=np.array(DicoOutput[keyval]["FLUX_ERR"])
    return DicoOutput


### Define spectral functions to be fitted, along with convenience printout functions.
# define Free-Free Absorption spectrum
def FFA(freqs,S0,tau,alphaFFA):
    """ Free-free absorption 
    """
    freqs=(freqs/nu0)
    expnum=-tau*(freqs**-2.1)
    expvals=np.exp(expnum.astype(np.float128))
    S=S0*(freqs**alphaFFA)*expvals
    return S.astype(float)
def FFA_prior(params):
    """ Function to contain the limits for each parameter 
    """
    S0,tau,alpha=params
    if not(0. <= S0 <= 80):
        return -np.inf
    if not(0. <= tau <= 50 ):
        return -np.inf
    if not(-2 <= alpha <= 2):
        return -np.inf
    return 0
def PrintFFAParams(*mcmcfit):
    """ FFA printout function 
    """
    S0,tau,alpha=mcmcfit
    print("Fitted FFA parameters.")
    print("S0            : %f +- %f"%(S0[0],np.mean([S0[1:2]])))
    print("Optical Depth : %f +- %f"%(tau[0],np.mean([tau[1:2]])))
    print("Free-free spi : %f +- %f"%(alpha[0],np.mean([alpha[1:2]])))
# define Synchrontron Self Absorption spectrum
def SSA(freqs,S0,nuc,tau,alphaSSA):
    """ synchrotron self-absorption spectrum """
    freqs=freqs/nu0/nuc
    expnum=-tau*(freqs**(alphaSSA-2.5))
    expvals=np.exp(expnum.astype(np.float128))
    S=S0*(freqs**2.5)*(1-expvals)
    return S.astype(float)
def SSA_prior(params):
    """ Function to contain the limits for each of the parameters
    """
    S0,nuc,tau,alpha = params
    if not (0 <= S0 <= 50):
        return -np.inf
    if not (0 <= nuc <= 0.2):
        return -np.inf
    if not (-20 <= tau <= 20):
        return -np.inf
    if not (-2 <= alpha <= 1):
        return -np.inf
    return 0
def PrintSSAParams(*mcmcfit):
    """ SSA printout function 
    """
    S0,nuc,tau,alpha=mcmcfit
    print("Fitted SSA parameters.")
    print("S0              : %f +- %f"%(S0[0],np.mean([S0[1:2]])))
    print("nu_critical     : %f +- %f GHz"%(nuc[0],np.mean([nuc[1:2]])))
    print("Optical Depth   : %f +- %f"%(tau[0],np.mean([tau[1:2]])))
    print("sync. s.-a. spi : %f +- %f"%(alpha[0],np.mean([alpha[1:2]])))
# define simple power-law spectrum
def powerlaw(freqs,*params):
    """ power law spectrum 
    """
    S0,alphaPL=params
    freqs=freqs/nu0
    S=S0*freqs**alphaPL
    return S.astype(float)
def powerlaw_prior(params):
    S0,alpha=params
    if not (0 <= S0 <= 50):
        return -np.inf
    if not (-2 <= alpha <= 1):
        return -np.inf
    return 0
def PrintPLParams(*mcmcfit):
    """ power law printout function
    """
    S0,alpha=mcmcfit
    print("Fitted power law parameters.")
    print("S0              : %f +- %f"%(S0[0],np.mean([S0[1:2]])))
    print("alpha           : %f +- %f"%(alpha[0],np.mean([alpha[1:2]])))

### Define likelihood functions
def lnlike(theta, DicoFlux, key, model):
    """Likelihood function 
    theta ({np.array}) -- parameters from emcee
    """
    flux     = DicoFlux[key]["FLUX"]
    flux_err = DicoFlux[key]["FLUX_ERR"]
    nu       = DicoFlux["Freq"]
    # Calculate the log likelihood for real measurements
    model_flux = model(nu, *theta)
    chi2 = np.sum( ((flux-model_flux)/flux_err)**2. + np.log(2.*np.pi*flux_err**2.))
    # Calculate the modified liklihood for the limits. Taken from MrMoose
    model = model(nu, *theta)
    rms = flux
    mod_chi2 = np.sum(-2.*np.sum(np.log((np.pi/2.)**0.5*rms*(1.+erf(((rms-model)/((2**0.5)*rms)))))))    
    return -0.5*(chi2 + mod_chi2)            
def lnprob(theta, df, key, model_prior, model):
    theta_prior = model_prior(theta)
    if theta_prior == -np.inf:
        return -np.inf
    theta_model = lnlike(theta, df, key, model)
    return theta_model

### Define fitting function, including MCMC
def FitSpectra(DicoFlux,DicoModel,fluxregkey,spectra=["pl","ffa","ssa"]):
    """ Function to fit spectra to a set of points along frequency 
    """
    # define bounds
    ssabounds=((0,0.0,-20,-2.),(50,0.2,20,0.))
    ffabounds=((0,-20,-2.),(50,20,1.))
    plbounds =((0,-2.),(50,1))
    itermax=20000
    DicoModel[fluxregkey]={}
    DicoModel[fluxregkey]["FreqFitCurve"]=np.arange(0.8*np.min(DicoFlux["Freq"]),1.1*np.max(DicoFlux["Freq"]),501)
    if "pl" in spectra:
        DicoModel[fluxregkey]["PL"]={}
        ### fit power law
        print("Beginning powerlaw fit")
        # curve_fit to initialise MCMC
        p0_PL=(np.median(DicoFlux[fluxregkey]["FLUX"]),-0.5)
        fit_PL=curve_fit(powerlaw,DicoFlux["Freq"],DicoFlux[fluxregkey]["FLUX"],sigma=DicoFlux[fluxregkey]["FLUX_ERR"],
                         bounds=plbounds,maxfev=itermax,p0=p0_PL)
        # do MCMC fit
        ndim, nwalkers = len(p0_PL), 500
        pos = [fit_PL[0] + 1e-3*np.random.randn(ndim) for i in range(nwalkers)]
        PL_sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(DicoFlux, fluxregkey, powerlaw_prior, powerlaw))
        PL_sampler.run_mcmc(pos, 500)
        PL_samples = PL_sampler.chain[:, -50, :].reshape((-1,ndim))
        y_sample=[]
        for j in range(PL_samples.shape[0]):
            y_sample.append(powerlaw(DicoFlux["Freq"], *(PL_samples[j,i] for i in range(ndim))))
        y_sample=np.array(y_sample)
        PL_fill_range = np.nanpercentile(y_sample, [16, 84], axis=0)
        PL_S0, PL_alpha=map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),\
                             zip(*np.percentile(PL_samples, [16,50,84],
                                                axis=0)))
        DicoModel[fluxregkey]["PL"]["S0"]=PL_S0
        DicoModel[fluxregkey]["PL"]["alpha"]=PL_alpha
        DicoModel[fluxregkey]["PL"]["fill_range"]=PL_fill_range
        DicoModel[fluxregkey]["PL"]["FitCurve"]=powerlaw(DicoModel[fluxregkey]["FreqFitCurve"],PL_S0[0],PL_alpha[0])
        print(DicoModel[fluxregkey]["PL"])
        print("Power law fit done to %s"%fluxregkey)
    if "ffa" in spectra:
        DicoModel[fluxregkey]["FFA"]={}
        ### fit free-free spectrum
        print("Begin FFA fit")
        # curve_fit to initialise MCMC
        p0_FFA=(np.median(DicoFlux[fluxregkey]["FLUX"]),1.,-0.5)
        fit_FFA=curve_fit(FFA,DicoFlux["Freq"],DicoFlux[fluxregkey]["FLUX"],sigma=DicoFlux[fluxregkey]["FLUX_ERR"],
                          bounds=ffabounds,maxfev=itermax,p0=p0_FFA)
        # do MCMC fit
        ndim, nwalkers = len(p0_FFA), 500
        pos = [fit_FFA[0] + 1e-3*np.random.randn(ndim) for i in range(nwalkers)]
        FFA_sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(DicoFlux, fluxregkey, FFA_prior, FFA))
        FFA_sampler.run_mcmc(pos, 500)
        FFA_samples = FFA_sampler.chain[:, -50, :].reshape((-1,ndim))
        y_sample=[]
        for j in range(FFA_samples.shape[0]):
            y_sample.append(FFA(DicoFlux["Freq"], *(FFA_samples[j,i] for i in range(ndim))))
        y_sample=np.array(y_sample)
        FFA_fill_range = np.nanpercentile(y_sample, [16, 84], axis=0)
        FFA_S0, FFA_tau, FFA_alpha=map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                                       zip(*np.percentile(FFA_samples, [16,50,84],axis=0)))
        DicoModel[fluxregkey]["FFA"]["S0"]=FFA_S0
        DicoModel[fluxregkey]["FFA"]["Tau"]=FFA_tau
        DicoModel[fluxregkey]["FFA"]["alpha"]=FFA_alpha
        DicoModel[fluxregkey]["FFA"]["fill_range"]=FFA_fill_range
        DicoModel[fluxregkey]["FFA"]["FitCurve"]=FFA(DicoModel[fluxregkey]["FreqFitCurve"],FFA_S0[0],FFA_tau[0],FFA_alpha[0])
        print(DicoModel[fluxregkey]["FFA"])
        print("FFA fit done to %s"%fluxregkey)
    if "ssa" in spectra:
        DicoModel[fluxregkey]["SSA"]={}
        ### fit synchrotron self-abs. spectrum
        print("Begin SSA fit")
        # curve_fit to initialise MCMC
        p0_SSA=(np.median(DicoFlux[fluxregkey]["FLUX"]),0.1,1.,-0.5)
        fit_SSA=curve_fit(SSA,DicoFlux["Freq"],DicoFlux[fluxregkey]["FLUX"],sigma=DicoFlux[fluxregkey]["FLUX_ERR"],
                          bounds=ssabounds,maxfev=itermax,p0=p0_SSA)
        # do MCMC fit
        ndim, nwalkers = len(p0_SSA), 500
        pos = [fit_SSA[0] + 1e-3*np.random.randn(ndim) for i in range(nwalkers)]
        SSA_sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(DicoFlux, fluxregkey, SSA_prior, SSA))
        SSA_sampler.run_mcmc(pos, 500)
        SSA_samples = SSA_sampler.chain[:, -50, :].reshape((-1,ndim))
        y_sample=[]
        for j in range(SSA_samples.shape[0]):
            y_sample.append(SSA(DicoFlux["Freq"], *(SSA_samples[j,i] for i in range(ndim))))
        y_sample=np.array(y_sample)
        SSA_fill_range = np.nanpercentile(y_sample, [16, 84], axis=0)
        SSA_S0, SSA_nuc, SSA_tau, SSA_alpha=map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),\
                             zip(*np.percentile(SSA_samples, [16,50,84],
                                                axis=0)))
        DicoModel[fluxregkey]["SSA"]["S0"]=SSA_S0
        DicoModel[fluxregkey]["SSA"]["nuc"]=SSA_nuc
        DicoModel[fluxregkey]["SSA"]["Tau"]=SSA_tau
        DicoModel[fluxregkey]["SSA"]["alpha"]=SSA_alpha
        DicoModel[fluxregkey]["SSA"]["fill_range"]=SSA_fill_range
        DicoModel[fluxregkey]["SSA"]["FitCurve"]=SSA(DicoModel[fluxregkey]["FreqFitCurve"],SSA_S0[0],SSA_nuc[0],SSA_tau[0],FFA_alpha[0])
        print(DicoModel[fluxregkey]["SSA"])
        print("SSA fit done to %s"%fluxregkey)
    return DicoModel

### Define plotting function
def PlotSpectra(DicoFlux, DicoModel,fluxregkeys,spectra=["pl","ffa","ssa"],scale="log"):
    """ Function to plot the spectral fits acquired
    """
    # find limits
    ylims=[]
    xlims=[]
    for key in fluxregkeys:
        ylims.append(np.max(DicoModel[key]["PL"]["FitCurve"]))
        ylims.append(np.min(DicoModel[key]["PL"]["FitCurve"]))
    if scale=="log":
        ymax=4*np.max(ylims)
        ymin=0.3*np.min(ylims)
    else:
        ymax=1.1*np.max(ylims)
        ymin=.9*np.min(ylims)
    # create initial figure
    #pylab.figure(figsize=(8,8))
    i=0

    fig=pylab.figure(figsize=(8,8))
    for key in fluxregkeys:
        i+=1
        #fig,ax=pylab.subplots(2,2,i)
        ax=fig.add_subplot(int("22"+str(i)))
        ax.errorbar(DicoFlux["Freq"],DicoFlux[key]["FLUX"],yerr=DicoFlux[key]["FLUX_ERR"],linestyle="",markersize=1)
        for spectr in spectra:
            spectr=str.upper(spectr)
            ax.plot(DicoModel[key]["FreqFitCurve"],DicoModel[key][spectr]["FitCurve"],label=spectr,linewidth=0.8)
            ax.fill_between(DicoFlux["Freq"], DicoModel[key][spectr]["fill_range"][0],DicoModel[key][spectr]["fill_range"][0],alpha=0.15)
        if i==1:
            ax.legend()
        pylab.xscale("log")
        pylab.yscale("log")
        pylab.ylim((ymin,ymax))
        pylab.xlabel("Freq [Hz]")
        pylab.ylabel("Flux [Jy]")
        if key=="Nhotspot":
            ftitle="North Hotspot"
        if key=="Shotspot":
            ftitle="South Hotspot"
        if key=="Nlobe":
            ftitle="North Lobe"
        if key=="Slobe":
            ftitle="South Lobe"
        pylab.title(ftitle)

    pylab.tight_layout()
    fig.savefig("SpectralFit")

def PrintParams(DicoModel,DicoFlux,fluxregkeys,spectra=["pl","ffa","ssa"]):
    print(DicoModel)
    for reg in fluxregkeys:
        angsize=DicoFlux[reg]["Area [\"]"]#*1e6
        print(angsize)
        print()
        print("Fits for %s"%reg)
        if "pl" in spectra:
            PrintPLParams(DicoModel[reg]["PL"]["S0"],DicoModel[reg]["PL"]["alpha"])
        if "ffa" in spectra:
            PrintFFAParams(DicoModel[reg]["FFA"]["S0"],DicoModel[reg]["FFA"]["Tau"],DicoModel[reg]["FFA"]["alpha"])
        if "ssa" in spectra:
            PrintSSAParams(DicoModel[reg]["SSA"]["S0"],DicoModel[reg]["SSA"]["nuc"],DicoModel[reg]["SSA"]["Tau"],DicoModel[reg]["SSA"]["alpha"])
            # do B-field estimate
            fluxmax=np.max(DicoModel[reg]["SSA"]["FitCurve"])
            freqfluxmax=np.mean(DicoModel[reg]["FreqFitCurve"][np.where(DicoModel[reg]["SSA"]["FitCurve"]==fluxmax)])
            Bfield=(freqfluxmax*1e-9/8.)**5 / (fluxmax**2) * angsize**4 / (1.464)
            freqerr=8.**-5 * 5*(freqfluxmax*1e-9/8.)**4*24e-6
            fluxerr=2*(fluxmax**-2)*np.mean(DicoModel[reg]["SSA"]["S0"][1:])
            Bfielderror=Bfield*np.sqrt(freqerr+fluxerr)
            print("Bfield estimate for %s: %f +- %f microG"%(reg,Bfield*1e6,Bfielderror*1e6))



###############################################################################################
###############################################################################################
#################################### ANALYSIS STARTS HERE #####################################
###############################################################################################
###############################################################################################



### list of images and region files to be analysed
images    = ["3C295_SCIENCE_IMAGES_PBSCALE_2LOFARFREQS/3C295.lofar.redo-0000-image.fits.rescaled.fitsrephased.fits",
             "3C295_SCIENCE_IMAGES_PBSCALE_2LOFARFREQS/3C295.lofar.redo-0001-image.fits.rescaled.fitsrephased.fits",
             "3C295_SCIENCE_IMAGES_PBSCALE_2LOFARFREQS/3C295.emerlin.rescaled.fitsrephased.fits",
             "3C295_SCIENCE_IMAGES_PBSCALE_2LOFARFREQS/wsclean.scienceimage.5GHz.VLA-MFS-image.fits.rescaled.fits",
             "3C295_SCIENCE_IMAGES_PBSCALE_2LOFARFREQS/wsclean.scienceimage.15GHz.VLA-MFS-image.fits.rescaled.fits"]

### regions in images
regs      = ["Nhotspot.reg","Shotspot.reg","Nlobe.reg","Slobe.reg"]

### params to see what we redo
readdata = False
fitdata  = False
plotdata = True

### create dico containing params and such
if readdata:
    DicoFlux  = GetFluxValues(images,regs,fluxsigmacutoff=5.)
    save_dict("DicoFlux",DicoFlux)
else:
    DicoFlux=load_dict("DicoFlux")
if fitdata:
    print("Fitting data")
    DicoModel={}
    for key in ["Nhotspot","Shotspot","Nlobe","Slobe"]:
        DicoModel = FitSpectra(DicoFlux,DicoModel,key,spectra=["pl","ffa","ssa"])
    save_dict("DicoModel",DicoModel)
else:
    DicoModel=load_dict("DicoModel")

### plot the data, fits
if plotdata==True:
    PlotSpectra(DicoFlux, DicoModel,fluxregkeys=["Nhotspot","Shotspot","Nlobe","Slobe"],spectra=["pl","ffa","ssa"],scale="log")
    ### print out parameters and b-field estimate
    PrintParams(DicoModel,DicoFlux,fluxregkeys=["Nhotspot","Shotspot","Nlobe","Slobe"],spectra=["pl","ffa","ssa"])

