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
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
import pyregion
from scipy.special import erf
from scipy.ndimage.filters import minimum_filter

def readArguments():
    # create parser
    parser=argparse.ArgumentParser(description="Make a png cutout from a FITS image.")
    parser.add_argument("-v","--verbose",help="Be verbose, say everything program does. Default is False",required=False,action="store_true")
    parser.add_argument("--filename",type=str,help="Name of the .fits file you want a cutout of",required=True,nargs="+")
    parser.add_argument("--overlay",type=str,help="Name of .fits file you want to overlay on your cutout",required=False,default=None)
    parser.add_argument("--RA",metavar="HH:MM:SS",type=str, help="Right Ascension in hexadecimal hour angle",required=True)
    parser.add_argument("--Dec",metavar="HH:MM:SS",type=str,help="Declination in hexadecimal degrees",required=True)
    parser.add_argument("--vmin",type=float,default=-2,help="Minimum value for display, in units of rms. Default -2.",required=False)
    parser.add_argument("--vmax",type=float,default=40,help="Maximum value for display, in units of rms. Default 40.",required=False)
    parser.add_argument("--forcev",help="vmin, vmax values are used strictly rather than as multiples of noise level",required=False,default=False,action="store_true")
    parser.add_argument("--size",metavar="arcmin",type=float,default=5,help="Size of cutout, in arcmin. Default is 5")
    parser.add_argument("--overlaymin",type=float,default=5,help="Number of noise values at which overlays being. Default is 5.")
    parser.add_argument("--nlevels",type=int,default=5,help="Number of layers in overlay. Default is 5.")
    parser.add_argument("--noise",metavar="all/first",choices=["all","first"],type=str,default="first",\
                        help="Set of images over which to estimate noise. Default is \"first\", only the first image in the list.",required=False)
    parser.add_argument("--pol",metavar="I/Q/U/V",choices=["I","Q","U","V"],type=str,default="I",\
                        help="Polarisation to choose, in case of polarimetric fits file. Default is \"I\".",required=False)
    parser.add_argument("--overlaychan",metavar="channel",type=int,default=0,\
                        help="Frequency channel to choose for overlay, in case of mutli-frequency image cube. Default is 0."\
                        ,required=False)
    parser.add_argument("--chan",metavar="channel",type=int,default=0,\
                        help="Frequency channel to choose, in case of mutli-frequency image cube. Default is 0."\
                        ,required=False)
    parser.add_argument("--histogram",help="Add histogram of the pixel distribution below the image",required=False,default=False,action="store_true")
    parser.add_argument("--invert",help="Invert colourmap",required=False,default=False,action="store_true")
    parser.add_argument("--normalise",help="Normalise average pixel value to 1 in all images",required=False,default=False,action="store_true")
    parser.add_argument("--append",help="String to add to output name",required=False,default="")
    parser.add_argument("--reg",help="Name of ds9 region to superimpose on the png cutout", required=False,default="")
    # parse
    args=parser.parse_args()
    return vars(args)


def MakeCutout(fitsfiles,overlay,RA,Dec,xDegrees=0.1,yDegrees=0.1,NSigmaVmax=10,NSigmaVmin=-10,overlaymin=3,nlevels=5\
               ,outname=None,SetVminToAverageNoise="first",verb=0,pol=0,chan=0,histogram=0,invertCmap=False,forcev=False,\
               ochan=0,append="",reg=""):
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
        if forcev==False:
            vmin=NSigmaVmin*np.mean(stdarray)
            vmax=NSigmaVmax*np.mean(stdarray)        
        else:
            vmin=NSigmaVmin
            vmax=NSigmaVmax
    else:
        # calculate noise in first image, use that as a reference
        if verb: print("Calculating noise in %s, use as reference for all images"%fitsfiles[0])
        im=image(fitsfiles[0])
        d=im.getdata()
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

    if histogram:
        if verb: print("Initialising overall histogram bounds")
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
        if verb: print("Initialisation complete: making cutouts")
    i=0
    if histogram:
        ytickvals=np.arange(6)/5.*np.log(np.max(narray))
        ytickvals=(10*ytickvals).astype(np.int)/10.
    if overlay!=None:
#        overlay=CheckB1950toJ2000(overlay,verb=verb)
        if verb:("Making overlay from file: %s"%overlay)
        hduOverlay=fits.open(overlay)
        

    for ffile in fitsfiles:
#        ffile=CheckB1950toJ2000(ffile)
        test=pylab.figure(figsize=(8,8))
        ax=pylab.subplot(111)
        temp = FITSFigure(ffile,slices=[pol,chan],figure=test,subplot=[0,0,1,1])
        fix_aplpy_fits(temp)
        temp.show_grayscale(vmin=vmin,vmax=vmax,invert=invertCmap)
#        print RAdeg,Decdeg
        temp.recenter(RAdeg,Decdeg,width=yDegrees,height=xDegrees)
        temp.add_colorbar()
        temp.colorbar.set_width(0.2)
        temp.colorbar.set_location("right")
        if overlay!=None:
            if len(hduOverlay[0].shape)==4:
                pixelvals=hduOverlay[0].data[ochan,pol].ravel()
            else:
                pixelvals=hduOverlay[0].data.ravel()
            pixelvals=pixelvals[np.isnan(pixelvals)==False]
            ind=np.int64(np.random.rand(100000)*pixelvals.size)
            
#            x=np.linspace(-10,10,len(pixelvals))
#            f=0.5*(1.+erf(x/np.sqrt(2.)))
#            n=90
#            F=1.-(1.-f)**n
#            ratio=np.abs(np.interp(0.5,F,x))
#            overstd=-minimum_filter(pixelvals,x)/ratio

            A=pixelvals.flat[ind]
            A=A[np.isnan(A)==0]
            overstd=np.std(A)
            print("overstd is",overstd)
            pixelmaxval=np.max(pixelvals)
#            print pixelmaxval
            levelmax=np.log10(pixelmaxval)
#            print levelmax
            minval=min(overlaymin*overstd,0.6*pixelmaxval)
            print("min level val, minval in std, minval as frac of peak", minval,overlaymin*overstd,0.6*pixelmaxval)
            levelmin=np.log10(minval)
            levels=10**(np.linspace(levelmin,levelmax,nlevels))
#            print levels
            temp.show_contour(hduOverlay,linewidths=1,cmap="autumn",origin="lower",levels=levels,alpha=0.4,overlap=True,slices=[0,ochan])
        if histogram:
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
            temp.tick_labels.set_yformat("dd:mm:ss")
            temp.tick_labels.set_xformat("hh:mm:ss")
        if reg!="":
            temp.show_regions(reg)
        if append=="":
            temp.save("./"+ffile.split("/")[-1]+".png")
            print("Created new cutout: %s"%("./"+ffile.split("/")[-1]+".png"))
        else:
            #temp.save("./"+ffile.split("/")[-1]+append+".png")
            #print "Created new cutout: %s"%("./"+ffile.split("/")[-1]+append+".png")
            temp.save("./"+append+".png")
            print("Created new cutout: %s"%("./"+append+".png"))
        temp.close()
    print("Done, thank you for using this script")
            
def HHMMSStoDegrees(HHMMSS):
   # convert HHMMSS string to angular value float
   HH,MM,SS=np.array(HHMMSS.split(":")).astype(float)
   degrees=HH+MM/60.+SS/3600.
   return degrees

def fix_aplpy_fits(aplpy_obj, dropaxis=2):
    """This removes the degenerated dimensions in APLpy 2.X...
    The input must be the object returned by aplpy.FITSFigure().
    `dropaxis` is the index where to start dropping the axis (by default it assumes the 3rd,4th place).
    """
    temp_wcs = aplpy_obj._wcs.dropaxis(dropaxis)
    temp_wcs = temp_wcs.dropaxis(dropaxis)
    aplpy_obj._wcs = temp_wcs
   

def CheckB1950toJ2000(filename,verb=False):
    hdulist=fits.open(filename)
    for keys in hdulist[0].header:
        if "OBSRA"in keys:
            RAkey=keys
        if "OBSDEC" in keys:
            DECkey=keys
        if "EPOCH" in keys:
            Epochkey=keys
        else: Epochkey=0
        if "CRVAL1" in keys:
            crval1key=keys
        if "CRVAL2" in keys:
            crval2key=keys

    if hdulist[0].header["EPOCH"]==1950.:
        b1950ra=hdulist[0].header[RAkey]
        b1950dec=hdulist[0].header[DECkey]
        if verb: print("Converting %s into a new icrs-WCS file"%filename)
        hdulist[0].header["EPOCH"]=2000.0
        c=SkyCoord(hdulist[0].header[RAkey],hdulist[0].header[DECkey],frame="fk4",unit="deg")
        cJ2000=c.transform_to("icrs")
        hdulist[0].header[RAkey]=cJ2000.ra.deg
        hdulist[0].header[DECkey]=cJ2000.dec.deg
        hdulist[0].header["EQUINOX"]=2000.
        hdulist[0].header[crval1key]=cJ2000.ra.deg
        hdulist[0].header[crval2key]=cJ2000.dec.deg
        newfilename=filename[:-4]+"J2000.fits"
        if verb: print("Saving icrs file as: %s"%newfilename)
        hdulist.writeto(newfilename,clobber=True)
        return newfilename
    else:
        return filename

if __name__=="__main__":
    args=readArguments()
    fitsfiles=args["filename"]
    overlay=args["overlay"]
    RA=args["RA"]
    Dec=args["Dec"]
    vmin=args["vmin"]
    vmax=args["vmax"]
    forcev=args["forcev"]
    overlaymin=args["overlaymin"]
    size=args["size"]
    verb=args["verbose"]
    noisearg=args["noise"]
    chan=args["chan"]
    pol=args["pol"]
    hist=args["histogram"]
    cmapInv=args["invert"]
    normalise=args["normalise"]
    nlevels=args["nlevels"]
    append=args["append"]
    ochan=args["overlaychan"]
    regfile=args["reg"]
    #    for i in fitsfiles:
#        CheckB1950toJ2000(i,verb=False)
        

    for i,j in enumerate(["I","Q","U","V"]):
        if pol==j:
            polnum=i
    if verb:
        print("Making cutouts of size %2.1f' centred at (RA=%s,Dec=%s) for the following files:"%(size,RA,Dec))
    if normalise:
        normfiles=[]
    testvar=0
    normvals=[6.7053,3.79805]
    for i in fitsfiles:        
        if normalise:
            if verb: print("Making normalised fits images, then cutouts")
            data,header=fits.getdata(i,header=True)
            data=data/normvals[testvar]#np.amax(data)#,dtype=np.float64)
            testvar=testvar+1
            fits.writeto(i[:-4]+"normalised.fits",data,header,clobber=True)
            normfiles.append(i[:-4]+"normalised.fits")
        print(i)

        #    stop
    if normalise:
        MakeCutout(normfiles,overlay,RA,Dec,xDegrees=size/60.,yDegrees=size/60.,NSigmaVmin=vmin,NSigmaVmax=vmax,overlaymin=overlaymin,outname=None,\
                   verb=verb,SetVminToAverageNoise=noisearg,pol=polnum,chan=chan,histogram=hist,invertCmap=cmapInv,nlevels=nlevels,forcev=forcev,\
                   ochan=ochan,append=append,reg=regfile)
    else:
        MakeCutout(fitsfiles,overlay,RA,Dec,xDegrees=size/60.,yDegrees=size/60.,NSigmaVmin=vmin,NSigmaVmax=vmax,overlaymin=overlaymin,outname=None,\
                   verb=verb,SetVminToAverageNoise=noisearg,pol=polnum,chan=chan,histogram=hist,invertCmap=cmapInv,nlevels=nlevels,forcev=forcev,\
                   ochan=ochan,append=append,reg=regfile)
