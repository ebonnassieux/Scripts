import numpy as np
from astropy.io import fits
import argparse


def MakeStokesP(stokesQfile,stokesUfile,outfilename):
    hdul    = fits.open(stokesQfile)
    stokesQ = fits.open(stokesQfile)[0].data
    stokesU = fits.open(stokesUfile)[0].data
    stokesP = np.sqrt(stokesQ**2+stokesU**2)  # should be Q**2 + U**2 in principle. In RL may need to be V**2 + Q**2
    hdul[0].data = stokesP
    hdul[0].header['CVALUE4'] = 'STOKES-P'
    hdul.writeto(outfilename,overwrite=True)
    print("Created pol map %s"%outfilename)

def MakeStokesPfrac(stokesIfile, stokesQfile,stokesUfile,outfilename):
    hdul    = fits.open(stokesQfile)
    stokesI = fits.open(stokesIfile)[0].data
    stokesQ = fits.open(stokesQfile)[0].data
    stokesU = fits.open(stokesUfile)[0].data
    stokesp = np.sqrt(stokesQ**2+stokesU**2)/np.abs(stokesI)  # should be Q**2 + U**2 in principle.
    hdul[0].data = stokesp
    hdul[0].header['CVALUE4'] = 'STOKES-p'
    hdul.writeto(outfilename,overwrite=True)
    print("Created fractional pol map %s"%outfilename)


def readArguments():
    parser=argparse.ArgumentParser("Create polarisation fraction maps from wsclean basename")
    parser.add_argument("--basename",  type=str,   help="basename used in wsclean", \
                        required=True,nargs="+")    
    args=parser.parse_args()
    return vars(args)

if __name__=="__main__":
    print("Running as main")
    args      = readArguments()
    basenames = args["basename"]

    for basename in basenames:
        Ifile   = basename+'-MFS-I-image.fits'
        Qfile   = basename+'-MFS-Q-image.fits'
        Ufile   = basename+'-MFS-U-image.fits'
        Vfile   = basename+'-MFS-V-image.fits'
        #Ufile   = 'IMAGES/OJ287.LOFAR.FinalScienceImage.tryStokes.switchpols.nojointpols-MFS-U-image.fits'
        outfile = basename+'-MFS-P-image.fits'
        outfile1= basename+'-MFS-p-image.fits'
        MakeStokesP(Qfile,Ufile,outfile)
#        MakeStokesPfrac(Ifile, Qfile,Ufile,outfile1)
