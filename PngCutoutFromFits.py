# This is a script that reads hexadecimal RA, Dec and a list of fits files
# and outputs one png cutout per fits centred at the coordinate. By default,
# these cutouts will be 1 degree by 1 degree.

# import libraries
import argparse
import matplotlib
matplotlib.use("Agg")
from aplpy import FITSFigure 
import numpy as np
import sys
import glob
from pyrap.images import image
import warnings
warnings.filterwarnings("ignore")

def readArguments():
    # create parser
    parser=argparse.ArgumentParser(description="Make a png cutout from a FITS image.")
    parser.add_argument("-v","--verbose",help="Be verbose, say everything program does. Default is True",required=False,action="store_true")
    parser.add_argument("--filename",type=str,help="Name of the .fits file you want a cutout of",required=True,nargs="+")
    parser.add_argument("--RA",metavar="HH:MM:SS",type=str, help="Right Ascension in hexadecimal hour angle",required=True)
    parser.add_argument("--Dec",metavar="HH:MM:SS",type=str,help="Declination in hexadecimal degrees",required=True)
    parser.add_argument("--vmin",type=float,default=-2,help="Minimum value for display, in units of rms. Default -2.",required=False)
    parser.add_argument("--vmax",type=float,default=40,help="Maximum value for display, in units of rms. Default 40.",required=False)
    parser.add_argument("--size",metavar="arcmin",type=float,default=5,help="Size of cutout, in arcmin. Default is 5")
    parser.add_argument("--noise",metavar="all/first",choices=["all","first"],type=str,default="first",help="Set of images over which to estimate noise. Default is \"first\", only the first image in the list.",required=False)
    # parse
    args=parser.parse_args()
    return vars(args)


def MakeCutout(fitsfiles,RA,Dec,xDegrees=0.1,yDegrees=0.1,NSigmaVmax=10,NSigmaVmin=-10,outname=None,SetVminToAverageNoise="first",verb=0):
    RAdeg  = HHMMSStoDegrees(RA)*15. # this converts RA from hours to degrees
    Decdeg = HHMMSStoDegrees(Dec)
    # find vmin, vmax
    if SetVminToAverageNoise=="all":
        stdarray=[]
        for ffile in fitsfiles:
            im=image(ffile)
            d=im.getdata()
            ind=np.int64(np.random.rand(10000)*d.size)
            A=d.flat[ind]
            A=A[np.isnan(A)==0]
            std=np.std(A)
            stdarray.append(std)
#            vmin=NSigmaVmin*std
#            vmax=NSigmaVmax*std
            im.close()
        vmin=NSigmaVmin*np.mean(stdarray)
        vmax=NSigmaVmax*np.mean(stdarray)        
    else:
        # calculate noise in first image, use that as a reference
        if verb: print "Calculating noise in %s, use as reference for all images"%fitsfiles[0]
        im=image(fitsfiles[0])
        d=im.getdata()
        ind=np.int64(np.random.rand(10000)*d.size)
        A=d.flat[ind]
        A=A[np.isnan(A)==0]
        std=np.std(A)
        vmin=NSigmaVmin*std
        vmax=NSigmaVmax*std

    for ffile in fitsfiles:
        temp = FITSFigure(ffile)
        temp.show_grayscale(vmin=vmin,vmax=vmax)
        temp.recenter(RAdeg,Decdeg,width=xDegrees,height=xDegrees)
        # do grid
#        temp.add_grid()
#        temp.grid.set_alpha(0.8)
#        temp.grid.set_color("red")
        # do colorbar
        temp.add_colorbar()
        temp.colorbar.set_width(0.2)
        temp.colorbar.set_location("right")
        temp.save(ffile+"."+RA+"."+Dec+".png")
        print "Created new cutout: %s"%ffile+"."+RA+"."+Dec+".png"
        temp.close()

def HHMMSStoDegrees(HHMMSS):
   # convert HHMMSS string to angular value float
   HH,MM,SS=np.array(HHMMSS.split(":")).astype(float)
   degrees=HH+MM/60.+SS/3600.
   return degrees



if __name__=="__main__":
    args=readArguments()
    fitsfiles=args["filename"]
    RA=args["RA"]
    Dec=args["Dec"]
    vmin=args["vmin"]
    vmax=args["vmax"]
    size=args["size"]
    verb=args["verbose"]
    noisearg=args["noise"]
    if verb:
        print "Making cutouts of size %2.1f' at (RA=%s,Dec=%s) for the following files:"%(size,RA,Dec)
        for i in fitsfiles:
            print i
    MakeCutout(fitsfiles,RA,Dec,xDegrees=size/60.,yDegrees=size/60.,NSigmaVmin=vmin,NSigmaVmax=vmax,outname=None,verb=verb,SetVminToAverageNoise=noisearg)
