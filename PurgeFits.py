import glob
import numpy as np
from astropy.io import fits

# Create a list of your fits files
# You'll need to change the glob statement depending on whether you've got .FITS or .fits files
fitslist = list(["3C295.0.175.ICLN.J2000.fits"])#sorted(glob.glob('PAPER_FITS/*.fits'))

fitslist = list(["~/m31.core.pass0.test1.app.restored.fits"])


#fitslist=list(["3C295_SCIENCE_IMAGES_5GHz_15GHz_PERLEY_FINAL/wsclean.scienceimage.5GHz.VLA.perley-image.fits.rescaled.fitsrephased.fits",
#          "3C295_SCIENCE_IMAGES_5GHz_15GHz_PERLEY_FINAL/wsclean.scienceimage.15GHz.VLA.perley-image.fits.rescaled.fitsrephased.fits"])

for f in fitslist:
    fh = fits.getheader(f)
    fd = fits.getdata(f)


    
    # If you've got more than two axes, remove FREQUENCY and STOKES axes as APLPy.FITSFigure doesn't like 3- or 4-D for some reason...
    if len(fd.shape) > 2:
        hitems = ['PC1_3','PC1_4','PC2_3','PC2_4','PC3_1','PC3_2','PC3_3','PC3_4','PC4_1','PC4_2','PC4_3','PC4_4',\
            'PC01_03','PC01_04','PC02_03','PC02_04','PC03_01','PC03_02','PC03_03','PC03_04','PC04_01','PC04_02','PC04_03','PC04_04',\
            'CROTA3','CROTA4','CRPIX4','CDELT4','CRVAL4','CTYPE4','CRPIX3','CDELT3','CRVAL3','CTYPE3','CUNIT3','NAXIS4','NAXIS3','CUNIT4']
        # For each of the extra coordinates that FITSFigure doesn't like, check if it's there and remove it if so.
        for hi in hitems:
            try:
                fh.remove(hi)
            except:
                pass
        fh.set('NAXIS',2)
        fh.set('WCSAXES',2)
        fh.set('BUNIT', 'mJy/beam')

        # You'll need to change this .fits to .FITS if you have that formatting
        fits.writeto(f.replace('.fits','.PURGED.fits'), fd[0,0,:,:], header=fh, overwrite=True)
    else:
        # If this activates, you're already dealing with a 2D image so that's fine.
        # You'll need to change this .fits to .FITS if you have that formatting
        fits.writeto(f.replace('.fits','.mjy.fits'), 1e3*fd[:,:], header=fh, overwrite=True)

