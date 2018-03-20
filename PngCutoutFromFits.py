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
import pylab
from astropy.io import fits
import matplotlib as plt

def readArguments():
    # create parser
    parser=argparse.ArgumentParser(description="Make a png cutout from a FITS image.")
    parser.add_argument("-v","--verbose",help="Be verbose, say everything program does. Default is False",required=False,action="store_true")
    parser.add_argument("--filename",type=str,help="Name of the .fits file you want a cutout of",required=True,nargs="+")
    parser.add_argument("--RA",metavar="HH:MM:SS",type=str, help="Right Ascension in hexadecimal hour angle",required=True)
    parser.add_argument("--Dec",metavar="HH:MM:SS",type=str,help="Declination in hexadecimal degrees",required=True)
    parser.add_argument("--vmin",type=float,default=-2,help="Minimum value for display, in units of rms. Default -2.",required=False)
    parser.add_argument("--vmax",type=float,default=40,help="Maximum value for display, in units of rms. Default 40.",required=False)
    parser.add_argument("--size",metavar="arcmin",type=float,default=5,help="Size of cutout, in arcmin. Default is 5")
    parser.add_argument("--noise",metavar="all/first",choices=["all","first"],type=str,default="first",\
                        help="Set of images over which to estimate noise. Default is \"first\", only the first image in the list.",required=False)
    parser.add_argument("--pol",metavar="I/Q/U/V",choices=["I","Q","U","V"],type=str,default="I",\
                        help="Polarisation to choose, in case of polarimetric fits file. Default is \"I\".",required=False)
    parser.add_argument("--chan",metavar="channel",type=int,default=0,\
                        help="Frequency channel to choose, in case of mutli-frequency image cube. Default is 0."\
                        ,required=False)
    parser.add_argument("--histogram",help="Add histogram of the pixel distribution below the image",required=False,default=False,action="store_true")
    parser.add_argument("--invert",help="Invert colourmap",required=False,default=False,action="store_true")
    parser.add_argument("--normalise",help="Normalise average pixel value to 1 in all images",required=False,default=False,action="store_true")
    # parse
    args=parser.parse_args()
    return vars(args)


def MakeCutout(fitsfiles,RA,Dec,xDegrees=0.1,yDegrees=0.1,NSigmaVmax=10,NSigmaVmin=-10,outname=None,\
               SetVminToAverageNoise="first",verb=0,pol=0,chan=0,histogram=0,invertCmap=False):
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
#            im.close()
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

    if histogram:
        if verb: print "Initialising overall histogram bounds"
        blarg=[]
        blorg=[]
        narray=[]
        binsarray=[]
        for ffile in fitsfiles:
            hdulist=fits.open(ffile)
            pixelvals=hdulist[0].data[chan,pol].ravel()
            n,bins,_=pylab.hist(pixelvals,bins=np.sqrt(len(pixelvals)).astype(np.int),normed=True)
            blarg.append([np.max(pixelvals),np.min(pixelvals)])
            blorg.append(np.max(n))
            hdulist.close()
            narray.append(n)
            binsarray.append(bins)
        minval=np.min(blarg)
        maxval=np.max(blarg)
        ymax=1.1*np.max(blorg)
        pylab.clf()
        if verb: print "Initialisation complete: making cutouts"
    i=0
    if histogram:
        ytickvals=np.arange(6)/5.*np.log(np.max(narray))
        ytickvals=(10*ytickvals).astype(np.int)/10.
    for ffile in fitsfiles:
        test=pylab.figure(figsize=(8,8))
        ax=pylab.subplot(111)
        temp = FITSFigure(ffile,slices=[chan,pol],figure=test,subplot=[0,0,1,1])
        temp.show_grayscale(vmin=vmin,vmax=vmax,invert=invertCmap)
        temp.recenter(RAdeg,Decdeg,width=yDegrees,height=xDegrees)
        temp.add_colorbar()
        temp.colorbar.set_width(0.2)
        temp.colorbar.set_location("right")
        if histogram:
#            rect=pylab.Rectangle((0.2,0.2),.6,.6,facecolor="red",edgecolor="black")#,transform=ax.transFigure)
#            ax.add_patch(rect)
            rectsuperplot=pylab.axes([-0.19,0.79,0.515,0.18])
            rectsuperplot.tick_params(labelsize=0)
            rectsuperplot.tick_params(size=0)
            rectsuperplot.add_patch(plt.patches.Rectangle((0,0),1,1,facecolor="white",edgecolor="black"))
            histsuperplot=pylab.axes([-.1,0.85,0.4,0.1])
            logarray=np.log(narray[i])
            logarray[logarray<0]=0
            pylab.bar(binsarray[i][0:-1],logarray,align="edge",width=(binsarray[i][1]-binsarray[i][0]),facecolor="white")
            pylab.xlim((-std*2,std*2))
            pylab.yticks(ytickvals)
            pylab.ylabel("log(count)")            
            pylab.xlabel("Flux")
            i=i+1
#        temp.save(ffile+"."+RA+"."+Dec+".png")
        temp.save(ffile+".png")
        print "Created new cutout: %s"%ffile+".png"
        temp.close()
    print "Done, thank you for using this script"
            
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
    chan=args["chan"]
    pol=args["pol"]
    hist=args["histogram"]
    cmapInv=args["invert"]
    normalise=args["normalise"]
    for i,j in enumerate(["I","Q","U","V"]):
        if pol==j:
            polnum=i
    if verb:
        print "Making cutouts of size %2.1f' centred at (RA=%s,Dec=%s) for the following files:"%(size,RA,Dec)
    if normalise:
        normfiles=[]
    for i in fitsfiles:
        if normalise:
            if verb: "Making normalised fits images, then cutouts"
            data,header=fits.getdata(i,header=True)
            data=data/np.mean(data,dtype=np.float64)
            fits.writeto(i[:-4]+"normalised.fits",data,header,clobber=True)
            normfiles.append(i[:-4]+"normalised.fits")
        print i
    if normalise:
        MakeCutout(normfiles,RA,Dec,xDegrees=size/60.,yDegrees=size/60.,NSigmaVmin=vmin,NSigmaVmax=vmax,outname=None,\
                   verb=verb,SetVminToAverageNoise=noisearg,pol=polnum,chan=chan,histogram=hist,invertCmap=cmapInv)
    else:
        MakeCutout(fitsfiles,RA,Dec,xDegrees=size/60.,yDegrees=size/60.,NSigmaVmin=vmin,NSigmaVmax=vmax,outname=None,\
                   verb=verb,SetVminToAverageNoise=noisearg,pol=polnum,chan=chan,histogram=hist,invertCmap=cmapInv)
