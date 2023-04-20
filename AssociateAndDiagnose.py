import numpy as np
import astropy
from regions import Regions
from astropy.io import fits
from aplpy import FITSFigure 
import os
import matplotlib.pyplot as pylab
from astropy import wcs
from regions import PixCoord
import pyregion
import astropy.units as u

import warnings
warnings.filterwarnings("ignore")

def fix_aplpy_fits(aplpy_obj, dropaxis=2):
    """This removes the degenerated dimensions in APLpy 2.X...
    The input must be the object returned by aplpy.FITSFigure().
    `dropaxis` is the index where to start dropping the axis (by default it assumes the 3rd,4th place).
    """
    temp_wcs = aplpy_obj._wcs.dropaxis(dropaxis)
    temp_wcs = temp_wcs.dropaxis(dropaxis)
    aplpy_obj._wcs = temp_wcs

def GetMaskWithinRegion(regfile,hdu):
    reg=pyregion.open(regfile)
    mask=reg.get_mask(hdu[0])
    return mask

if __name__=="__main__":

    regionfilename    = "soft+supersoft.reg"
    imagefilename     = "M31-lowres-LOFAR.fits"
    outdirname        = "SSS_lowres_AssociationResults"
    pybdsfregfilename = "M31_lowres.pybdsf.reg"
    pybdsfcatfilename = "M31_lowres.pybdsf.fits"
    
    outdir=os.getcwd()+"/"+outdirname
    if os.path.exists(os.getcwd()+outdirname)==False:
        os.system("mkdir "+outdir)

    print("Reading pybdsf catalogue")
    pybdsfhdul=fits.open(pybdsfcatfilename)
    pybdsfcat=pybdsfhdul[1].data
    pybdsfhdul.close()
        
    print("Finding primary beam of radio image")
    hdul=fits.open(imagefilename)
    bmaj,bmin,bpa = hdul[0].header["BMAJ"],hdul[0].header["BMIN"],hdul[0].header["BPA"]
    dpix=np.abs(hdul[0].header["CDELT1"])
    fwhm_to_sigma = 1. / (8 * np.log(2))**0.5
    omega_B = 2*np.pi * (bmaj*fwhm_to_sigma)*(bmin*fwhm_to_sigma)
    omega_pix = np.abs(hdul[0].header["CDELT1"]*hdul[0].header["CDELT2"])
    beams2jy=omega_B/omega_pix
    imwcs=wcs.WCS(hdul[0].header)
    imwcs=imwcs.dropaxis(2)
    imwcs=imwcs.dropaxis(2)
    hdul[0].wcs=imwcs
    hdul[0].data=hdul[0].data[0,0,:,:]

    print("Setting angular separation expected")
    catresolution=1./3600
    angsep=np.sqrt(catresolution**2+(bmaj*bmin))*u.deg
    print(angsep.to("arcsec"))
    
    print("Starting association")

    regions=Regions.read(regionfilename,format="ds9")
    image=fits.open(imagefilename)
    pybdsfregions=Regions.read(pybdsfregfilename,format="ds9")
    

    forcev=False
    sizedeg=240./3600
    NSigmaVmin=-1.5
    NSigmaVmax=2.5
    d=hdul[0].data
    ind=np.int64(np.random.rand(10000)*d.size)
    A=d.flat[ind]
    A=A[np.isnan(A)==0]
    std=np.std(A)
    
    if forcev==False:
        vmin=NSigmaVmin*std
        vmax=NSigmaVmax*std
    else:
        vmin=NSigmaVmin
        vmax=NSigmaVmax
    print("noise is ", std)
    
    pol=0
    chan=0
    overlay=True

    i=0

    # make basic primary beam file
    f=open("primarybeam.reg","w")
    f.write("# Region file format: DS9 version 4.1\n")
    f.write("global color=blue dashlist=8 3 width=1"+\
            " font=\"helvetica 10 normal roman\" select=1"+\
            " highlite=1 dash=0 fixed=0 edit=1 move=1"+\
            " delete=1 include=1 source=1\n")
    f.write("fk5\n")
    f.write("ellipse %s %s %s %s %s"%("00h00m00s","00d00m00s", 0.5*bmaj, 0.5*bmin, bpa))
    f.close()
    
    for region in regions:
        # put primary beam around position
        nustring="ellipse %s %s %s %s %s"%(region.center.to_string("hmsdms").split()[0],
                                      region.center.to_string("hmsdms").split()[1],
                                      0.5*bmaj, 0.5*bmin, bpa)
        os.system("sed -i '4s/.*/%s/' primarybeam.reg"%nustring)
        
        test=pylab.figure(figsize=(12,6))
        ax=pylab.subplot(121)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        temp = FITSFigure(image,slices=[pol,chan],figure=test,subplot=(1,2,1))#[0,0,1,1])
        fix_aplpy_fits(temp)
        temp.add_beam()
        temp.show_grayscale(vmin=vmin,vmax=vmax,invert=False)
        temp.recenter(region.center.ra,region.center.dec,width=sizedeg,height=sizedeg)
        temp.add_colorbar()
        temp.colorbar.set_width(0.2)
        temp.colorbar.set_location("right")
        temp.show_regions(regionfilename)
        temp.show_regions("primarybeam.reg")


        # filter pybdsf region file
        pybdsfflux=0
        E_pybdsfflux=0
        for pbdsreg in pybdsfregions:
            sep=pbdsreg.center.separation(region.center)
            if (sep.to("arcsec")<=20*angsep.to("arcsec")):
                pbdsreg.write("temp.reg",overwrite=True,format="ds9")
                os.system("sed -i '3s/.*/fk5/' temp.reg") # change the stupid j2000 to stupid fk5
                temp.show_regions("temp.reg")
                if (sep.to("arcsec")<=angsep.to("arcsec")):
                    print("pybdsf fit for %i found"%i)
                    sourceid=int(pbdsreg.meta["text"].split("_")[-2].split("i")[1])
                    pybdsfflux   = pybdsfcat[pybdsfcat["source_id"]==sourceid]["Total_flux"]
                    E_pybdsfflux = pybdsfcat[pybdsfcat["source_id"]==sourceid]["E_Total_flux"]
        ax1=pylab.subplot(122)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        
        regmask=GetMaskWithinRegion("primarybeam.reg",hdul)
        fluxval = np.sum(d[regmask]) * beams2jy
        sigval  = np.sum(d[regmask])/std
        
        columns=("RA","Dec")
        rows=[""]
        vals=[[region.center.to_string("hmsdms").split()[0],region.center.to_string("hmsdms").split()[1]]]
        ax1.table(cellText=vals,colLabels=columns, bbox= [0.1,0.75,0.8,0.2])

        columns     = (r"$S_{peak}$ [mJy]", r"$S_{total}$ [mJy]", "rms [mJy]")
        fluxval     = np.sum(d[regmask]) * beams2jy
        peakfluxval = np.max(d[regmask]) * beams2jy
        rms         = std                * beams2jy
        vals        = [["%2.5f"%(1e3*peakfluxval),
                        "%2.5f"%(1e3*fluxval),
                        "%2.5f"%(1e3*rms)]]
        ax1.table(cellText=vals,colLabels=columns, bbox= [0.1,0.50,0.8,0.1])
        
        columns      = (r"$S_{PyBDSF}$ [mJy]", r"$\delta S_{PyBDSF}$ [mJy]")
        vals=[["%2.5f"%(1e3*pybdsfflux),"%2.3f"%(1e3*E_pybdsfflux)]]
        ax1.table(cellText=vals,colLabels=columns, bbox= [0.1,0.25,0.8,0.1])

        temp.save(outdir+"/"+"point%i"%i+".png")
        
        i=i+1
        pylab.clf()
        pylab.close()
        print("Done with ",i, "of", len(regions))
