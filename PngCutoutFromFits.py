# This is a script that reads hexadecimal RA, Dec and a list of fits files
# and outputs one png cutout per fits centred at the coordinate. By default,
# these cutouts will be 1 degree by 1 degree.

# import libraries
import argparse
import matplotlib
matplotlib.use("Agg")
#import aplpy
import numpy as np
import sys
import glob
from pyrap.images import image
import warnings
warnings.filterwarnings("ignore")

def readArguments():
    # create parser
    parser=argparse.ArgumentParser(description="Make a png cutout from a FITS image.")
    parser.add_argument("--filename",type=str,help="Name of the .fits file you want a cutout of",required=True,nargs=argparse.REMAINDER)
    parser.add_argument("--RA",metavar="HH:MM:SS",type=str, help="Right Ascension in hexadecimal hour angle",required=True)
    parser.add_argument("--Dec",metavar="HH:MM:SS",type=str,help="Declination in hexadecimal degrees",required=True)
    parser.add_argument("--vmin",type=str,default=-2,help="Minimum value for display, in units of rms. Default -2.",required=False)
    parser.add_argument("--vmax",type=str,default=40,help="Maximum value for display, in units of rms. Default 40.",required=False)
    parser.add_argument("--size",metavar="arcmin",type=float,default=5,help="Size of cutout, in arcmin. Default is 5")
    # parse
    args=parser.parse_args()
    return vars(args)


def MakeCutout(fitsfiles,RA,Dec,xDegrees=0.1,yDegrees=0.1,NSigmaVmax=10,NSigmaVmin=-10,outname=None,SetVminToAverageNoise=False):
    RAdeg  = HHMMSStoDegrees(RA)*15. # this converts RA from hours to degrees
    Decdeg = HHMMSStoDegrees(Dec)
    # find vmin, vmax
    if SetVminToAverageNoise==True:
        stdarray=[]
        for ffile in fitsfiles:
            im=image(ffile)
            d=im.getdata()
            ind=np.int64(np.random.rand(10000)*d.size)
            A=d.flat[ind]
            A=A[np.isnan(A)==0]
            std=np.std(A)
            stdarray.append(std)
            vmin=-10*std
            vmax=40*std
            im.close()
        vmin=NSigmaVmin*np.mean(stdarray)
        vmax=NSigmaVmax*np.mean(stdarray)        
    else:
        # calculate noise in first image, use that as a reference
        print "Calculating noise in %s, use as reference for all images"%fitsfiles[0]
        im=image(fitsfiles[0])
        d=im.getdata()
        ind=np.int64(np.random.rand(10000)*d.size)
        A=d.flat[ind]
        A=A[np.isnan(A)==0]
        std=np.std(A)
        vmin=NSigmaVmin*std
        vmax=NSigmaVmax*std
        

    for ffile in fitsfiles:
        temp = aplpy.FITSFigure(ffile)
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
    print args
    fitsfiles=args["filename"]
    RA=args["RA"]
    print fitsfiles
    print RA

    #fitsfiles=sys.argv[1:-2]
    #print "List of .fits: ",fitsfiles
    #RA=sys.argv[-2]
    #Dec=sys.argv[-1]
    #print "RA  : %s"%RA
    #print "Dec : %s"%Dec
    #vmin=-1  # in units of sigma
    #vmax= 1.2  # in units of sigma 

    # make some control loop to say that RA, Dec need to be given in hexadecimal
    
    #MakeCutout(fitsfiles,RA,Dec,xDegrees=4,yDegrees=4,NSigmaVmin=vmin,NSigmaVmax=vmax,outname=None)
