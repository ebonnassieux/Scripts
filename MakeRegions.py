import numpy as np
from astropy.io import fits
from astropy import wcs
import argparse
import os
from regions import read_ds9
from astropy.coordinates import Angle
from astropy import units as u


def readArguments():
    # function to read arguments, making program more user-friendly
    parser=argparse.ArgumentParser("Calculate visibility imagin weights based on calibration quality")
    parser.add_argument("-v","--verbose",help="Be verbose, say everything program does. Default is False",required=False,action="store_true")
    parser.add_argument("--filename",type=str,help="Name of the fits image you want to investigate",required=True)
    parser.add_argument("--region",type=str,help="Name of the region(s) over which you want to investigate your image. Must be rectangle, or set of rectangles. If set of rectangles, one directory will be created per region.",required=True)
    parser.add_argument("--outdir",type=str,help="Name of the directory into which to save output regions. Default is Regions",required=False,default="Regions")
    args=parser.parse_args()
    return vars(args)


def findPrimaryBeam(hdul,fitstype="MIRIAD"):
    # function which takes hdul and returns size of restoring beam in arcsec. Assumes normalised gaussian with 2 fwhm params.
    test_list=hdul[0].header["HISTORY"]
    print hdul[0].header
    if fitstype=="MIRIAD":
        # read header assuming miriad header format. FWHM given in units of arcsec
        fwhm = np.array(list(filter(lambda x: "fwhm" in x, hdul[0].header["HISTORY"]))[0].split("=")[1].split(",")).astype(float)
        ang  = float(filter(lambda x: "pa" in x, hdul[0].header["HISTORY"])[0].split("=")[1])
    return fwhm,ang


def PixelCoords(hdul,regfilename):
    # function which takes hdul and region file, and returns set of pixel+world coordinates of all image pixels inside the region.
    # assumes rectangular regions ONLY.
    # start by reading the wcs info and creating the ra,dec grid
    w=wcs.WCS(hdul[0].header)
    x=np.arange(hdul[0].header["NAXIS1"])
    y=np.arange(hdul[0].header["NAXIS1"])
    X,Y=np.meshgrid(x,y)
    ra, dec = w.all_pix2world(X, Y, 1)
    # now read the info from the region file
    regions=read_ds9(regfilename)
    # assume regions are all rectangles. Do not assume anything on rotation.
    ravals=np.array([])
    decvals=np.array([])
    xcovals=np.array([])
    ycovals=np.array([])
    for reg in regions:
        ang=Angle(reg.angle,u.radian).degree
        centRA,centDec=(reg.center.ra.degree,reg.center.dec.degree)
        deltara=Angle(reg.width,u.radian).degree
        deltadec=Angle(reg.height,u.radian).degree
        if ang%360.:
            # do rotation
            rotang=ang%360*np.pi/180 # numpy uses values of radians
#            rap=ra-centRA
#            decp=ra-centDec
            ra1=(ra-centRA)*np.cos(rotang)-(dec-centDec)*np.sin(rotang)+centRA
            dec1=(ra-centRA)*np.sin(rotang)+(dec-centDec)*np.cos(rotang)+centDec
        else:
            ra1=ra
            dec1=dec
        # now see which pixels are in the region
        mask=(ra1>=centRA-0.5*deltara)*(ra1<=centRA+0.5*deltara)*(dec1>=centDec-0.5*deltadec)*(dec1<=centDec+0.5*deltadec)
        ravals=np.append(ravals,ra[mask])
        decvals=np.append(decvals,dec[mask])
        xcovals=np.append(xcovals,X[mask])
        ycovals=np.append(ycovals,Y[mask])
    # put everything on a single axis
    pixelwcs=np.zeros((len(ravals),2))
    pixelwcs[:,0]=ravals
    pixelwcs[:,1]=decvals
    pixelcoords=np.zeros_like(pixelwcs)
    pixelcoords[:,0]=xcovals
    pixelcoords[:,1]=ycovals
    return pixelcoords,pixelwcs



def writeRegions(filename,pixelcoords,worldcoords,primarybeam,dirname="Regions"):
    # this function takes the information to write a region file per pixel in the input
    # region file. Each region file is centred on 1 pixel, and is a gaussian of fwhm
    # provided. The name of each region file includes the filename and pixel coordinates
    # associated to the file, built as follows:
    #          dirname/filename.xPixCoord.yPixCoord.reg
    # this is for ease of later reconstruction.
    if os.path.isdir(dirname)==False:
        os.mkdir(dirname)
    debugf=open(filename+".allregions.reg","w+")
    debugf.write("# Region file format: DS9 version 4.1\n")
    debugf.write("global color=green dashlist=8 3 width=1 font=\"helvetica 10 normal roman\" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n")
    debugf.write("fk5\n")
    for i in range(pixelcoords.shape[0]):
        fname=dirname+"/"+filename+"."+str(pixelcoords[i,0])+"."+str(pixelcoords[i,1])+".reg"
        f=open(fname,"w+")
        f.write("# Region file format: DS9 version 4.1\n")
        f.write("global color=green dashlist=8 3 width=1 font=\"helvetica 10 normal roman\" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n")
        f.write("fk5\n")
        ra=Angle(worldcoords[i,0],u.degree).to_string(sep=":",unit=u.hour)
        dec=Angle(worldcoords[i,1],u.degree).to_string(sep=":",unit=u.degree)
        f.write("ellipse(%s,%s,%f\",%f\",%f)"%(ra,dec,primarybeam[0][0],primarybeam[0][1],primarybeam[1]))
        debugf.write("\nellipse(%s,%s,%f\",%f\",%f)"%(ra,dec,primarybeam[0][0]/25,primarybeam[0][1]/20,primarybeam[1]))
        f.close()
    debugf.close()



if __name__=="__main__":
    args=readArguments()
    filename=args["filename"]
    regname=args["region"]
    dirname=args["outdir"]
    v=args["verbose"]
    # open fits file
    if v: print "Opening fits file"
    hdul=fits.open(filename)
    # read header for primary beam info, assuming MIRIAD convention
    if v: print "Reading header information to find restoring beam characteristics"
    primarybeam=findPrimaryBeam(hdul)
    # get pixel & world coords for things inside the input region
    if v: print "Finding world/pixel coordinates inside region"
    pixelcoords,worldcoords=PixelCoords(hdul,regname)
    # write region for each set of coordinates found above
    if v: print "writing primary beam region files in directory %s"%dirname
    writeRegions(filename,pixelcoords,worldcoords,primarybeam,dirname)
    # exit gracefully
    if v: "All done. Thank you for using this script!"
    hdul.close()
