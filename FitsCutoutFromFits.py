import sys
import argparse
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS as astropyWCS
from astLib.astWCS import WCS
from astLib.astCoords import calcRADecSearchBox
from astLib.astImages import clipUsingRADecCoords
from astLib.astImages import saveFITS

def readArguments():
    # create parser
    parser=argparse.ArgumentParser(description="Make a FITS cutout from a FITS image.")
    parser.add_argument("--filename",type=str,help="Name of the .fits file you want a cutout of",required=True)
    parser.add_argument("--outname",type=str,help="Name of the output .fits file, optional",required=False)#,nargs=argparse.REMAINDER)
    parser.add_argument("--RA",metavar="HH:MM:SS",type=str, help="Right Ascension in hexadecimal hour angle",default=None)
    parser.add_argument("--Dec",metavar="HH:MM:SS",type=str,help="Declination in hexadecimal degrees",default=None)
    parser.add_argument("--size",metavar="arcmin",type=float,default=5,help="Size of cutout, in arcmin. Default is 5")
    # parse
    args=parser.parse_args()
    return vars(args)

def MakeCutout(filename,RA,dec,ArcMinSize,cutoutname=None):
    # open fits files
    imhdu = fits.open(filename)
    astropy_imwcs = astropyWCS(header=imhdu[0].header)
    imwcs = WCS(filename)
    imdata = imhdu[0].data[0,0,:,:]
    # parse RA, Dec coordinates
    if RA==None or Dec==None:
        print("RA or Dec not provided; cutout will be at image centre")
        # find central pixel coordinates
        fits_npix = max(imdata.shape)
        centrepixcoord = round(fits_npix/2)
        centreskycoords = astropy_imwcs.pixel_to_world(centrepixcoord,centrepixcoord,0,0)
        fits_centrecoords = centreskycoords[0]
        print(centrepixcoord,fits_centrecoords)
        RAdeg = fits_centrecoords.ra.deg
        Decdeg = fits_centrecoords.dec.deg
    if type(RA)==str:
        print("Cropped image centre:")
        print("RA   : %s"%RA)
        print("Dec  : %s"%dec)
        print("Size : %s '"%ArcMinSize)
        RAdeg  = HHMMSStoDegrees(RA)*15. # this converts RA from hours to degrees
        Decdeg = HHMMSStoDegrees(dec)
    print("Cropped image centre:")
    print("RA   : %s"%RA)
    print("Dec  : %s"%dec)
    print("Size : %s '"%ArcMinSize)
    # make cutout box
    rmin,rmax,dmin,dmax=calcRADecSearchBox(RAdeg,Decdeg,ArcMinSize/60.)
    cutout = clipUsingRADecCoords(imdata,imwcs,rmin,rmax,dmin,dmax)
    im=cutout["data"]
    if cutoutname==None:
        cutoutname=filename+".cutout.fits"
    saveFITS(cutoutname,cutout['data'],cutout['wcs'])
    print("Cutout is: %s"%cutoutname)
    imhdu.close()


def HHMMSStoDegrees(HHMMSS):
   # convert HHMMSS string to angular value float
   HH,MM,SS=np.array(HHMMSS.split(":")).astype(float)
   degrees=HH+MM/60.+SS/3600.
   return degrees

if __name__=="__main__":
    # parse arguments for this function
    args=readArguments()
    # assign variables
    filename=args["filename"]
    outname=args["outname"]
    ra=args["RA"]
    dec=args["Dec"]
    arcminsize=args["size"]
    # launch script
    MakeCutout(filename,ra,dec,arcminsize,outname)
    
