### Script to make freqlist file for RM-Tools from an input list of fits files

import numpy as np
from astropy.io import fits
import argparse

def readArguments():
    # create parser                                                                                                                                           
    parser=argparse.ArgumentParser(description="Make a png cutout from a FITS image.")
    parser.add_argument("-v","--verbose",help="Be verbose, say everything program does. Default is False",required=False,action="store_true")
    parser.add_argument("--filenames",type=str,help="Name of the .fits file you want a cutout of",required=True,nargs="+")
    parser.add_argument("--output",type=str,help="Name of the frequency list to use for RM-Tools",required=False,default='freqlist.txt')
    args=parser.parse_args()
    return vars(args)

def FillFreqList(filenames,outname):
    outf = open(outname,'w')
    for filename in filenames:
        # read frequency                                                                                                                                      
        freq = fits.open(filename)[0].header['CRVAL3']
        outf.write(str(freq)+'\n')
    outf.close()

if __name__=="__main__":
    args=readArguments()
    fitsfiles = args["filenames"]
    output    = args["output"]
    FillFreqList(fitsfiles,output)
