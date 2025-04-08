import numpy as np
import pandas as pd
from casacore.tables import table
from astropy.io import fits
from astropy.coordinates import SkyCoord, Distance
from astropy.time import Time
import astropy.units as u
import pylab
import requests
import os
import time
import argparse

### this script gets cutouts from DR3 for the CARMENES coordinates and places them at the request outdir
# example call:
# python3 GetCutouts_DR3.py -v --catalog /home/ebonnassieux/Project_CARMENES_LoTSS_CrossMatch/carmencita.107.csv --cutoutdir Cutouts_DR3

class GetCutouts:
    """
    A class that takes in an input catalog (assumes .csv) and attempts to create a cutout for each
    source in the LoTSS-DR3 field. These are written as .fits files to a specified output directory,
    and subsequently have their headers adapted to pybdsf processing.

    Attributes:
    -------------
    catalog    : string
        value pointing to the input catalog
    outdir     : string 
        value for the directory to write the output files
    size       : float
        size of the cutout fields in arcmin
    obstime    : str
        ISO time of observation for proper motion correction
    decfilter  : float
        minimum declination to use, in degrees.
    namekey    : str
        Key for the column containing source names in your catalog.
    RAkey      : str
        Key for the column containing RA in your catalog.
    Deckey     : str
        Key for the column containing Dec in your catalog.
    RA_pm_key  : str
        Key for the column containing RA proper motion in your catalog.
    Dec_pm_key : str
        Key for the column containing RA proper motion in your catalog.
    distkey    : str
        Key for the column containing distances in pc in your catalog.
    verbose          : bool
        flag to print out stdout information. Default: True     
    """

    def __init__(self,
                 catalog,
                 outdir,
                 size,
                 obstime,
                 decfilter,
                 namekey,
                 rakey,
                 deckey,
                 ra_pm_key,
                 dec_pm_key,
                 distkey,
                 verbose=False):
        self.verbose         = verbose
        self.j2000_reference = Time("2000-01-01 12:00:00")
        self.filename        = catalog
        # only csv supported right now
        if catalog.split(".")[-1]=="csv":
            if self.verbose:
                print("Catalog read as csv")
            self.catalog     = pd.read_csv(catalog)
        else:
            if self.verbose:
                print("Catalog not in .csv format : currently not supported")
                exit()
        self.cutoutsdir      = outdir
        self.size            = size
        self.time            = Time(obstime)
        self.decfilter       = decfilter
        if self.verbose:
            print("Source count in catalog :",len(self.catalog))
        decs                 = self.catalog["DE_deg"]
        cat_dec_filter       = (decs>self.decfilter)
        self.catalog         = self.catalog[cat_dec_filter]
        if self.verbose:
            print("Source count above declination limit of %3.1f :"%self.decfilter,len(self.catalog))
        self.namekey         = namekey
        self.rakey           = rakey
        self.deckey          = deckey
        self.ra_pm_key       = ra_pm_key
        self.dec_pm_key      = dec_pm_key
        self.distkey         = distkey
        
    def SetLoTSSReferenceTime(self):
        # since we are looking at moving objects, use upper and lower time limits here.
        # Taken from https://science.astron.nl/sdc/astron-data-explorer/data-releases/lotss-dr2/
        j2000_reference = Time("2000-01-01 12:00:00")
        lotss_t0 = Time("2014-05-23 00:00:00")
        lotss_dr2_end = Time("2020-02-05 23:59:59")  # this is the end time of DR2 observations
        lotss_t1 = Time("2024-02-05 23:59:59")  # this is an estimate of end of DR3 obs: need a better one.
        lotss_yrs   = lotss_t1.jyear-lotss_t0.jyear
        lotss_avgyr = lotss_t0.jyear + lotss_yrs/2
        self.lotss_yrs = lotss_yrs
        self.time = Time(lotss_avgyr,format="jyear")
        if self.verbose:
            print("Reference time set to LoTSS average time :",self.time.isot)

    def PrintCatHeader(self):
        # print keys for debug use
        if self.verbose:
            for key in self.cat.keys():
                print(key)

    def UpdatePositions(self,outname="carmencita.107.lotss_positions.csv"):
        RA_array=[]
        dec_array=[]
        err_array=[]
        for i in range(len(self.catalog)):
             # correct CARMENES coords by average estimated proper motion
            Coords = SkyCoord(ra       = self.catalog[self.rakey].values[i]*u.deg,
                              dec      = self.catalog[self.deckey].values[i]*u.deg,
                              distance = Distance(self.catalog.iloc[i][self.distkey]*u.pc),
                              pm_ra_cosdec = self.catalog.iloc[i][self.ra_pm_key]*u.mas/u.yr,
                              pm_dec   = self.catalog.iloc[i][self.dec_pm_key]*u.mas/u.yr,
                              obstime  = self.j2000_reference ).apply_space_motion(self.time)
            propmotionerror = np.sqrt(self.catalog.iloc[i][self.ra_pm_key]**2 + self.catalog.iloc[i][self.dec_pm_key]**2) * self.lotss_yrs / 1000
            RA_array.append(Coords.ra.deg)
            dec_array.append(Coords.dec.deg)
            err_array.append(propmotionerror)
        self.catalog["RA_LoTSS_deg"]           = np.array(RA_array)
        self.catalog["DEC_LoTSS_deg"]          = np.array(dec_array)
        self.catalog["LoTSS_pos_error_arcsec"] = np.array(err_array)
        self.catalog.to_csv(outname)
                
    def download(self):
        # read surveys login details
        f=open("/home/ebonnassieux/.lotssrc")
        login=f.readline().strip("\n")
        passwd=f.readline().strip("\n")
        # start downloading
        for i in range(len(self.catalog)):    
            # correct CARMENES coords by average estimated proper motion
            Coords = SkyCoord(ra       = self.catalog[self.rakey].values[i]*u.deg,
                              dec      = self.catalog[self.deckey].values[i]*u.deg,
                              distance = Distance(self.catalog.iloc[i][self.distkey]*u.pc),
                              pm_ra_cosdec = self.catalog.iloc[i][self.ra_pm_key]*u.mas/u.yr,
                              pm_dec   = self.catalog.iloc[i][self.dec_pm_key]*u.mas/u.yr,
                              obstime  = self.j2000_reference ).apply_space_motion(self.time)
            propmotionerror = np.sqrt(self.catalog.iloc[i][self.ra_pm_key]**2 + self.catalog.iloc[i][self.dec_pm_key]**2) * self.lotss_yrs / 1000
            pos = Coords.to_string('hmsdms').replace("s","").replace("m",":").replace("d",":").replace("h",":") # stupid way to do it
            # make cutout of DR3 on position of carmenes; based on https://github.com/mhardcastle/lotss-cutout-api
            url='https://lofar-surveys.org/'
            base='dr3'
            page=base+'-cutout.fits'
            if self.verbose:
                print('%i of %i - trying'%(i,len(self.catalog)),url+page,'params=',{'pos':pos,'size':self.size})
            r=requests.get(url+page,params={'pos':pos,'size':self.size},auth=(login,passwd),stream=True)
            if r.headers['content-type']!='application/fits':
                if self.verbose:
                    print('LoTSS Server did not return FITS file, probably no coverage of this area')
            else:
                outfile = self.cutoutsdir+"/"+self.catalog.iloc[i]["Name"].strip().replace(" ","_")+".fits"
                with open(outfile,'wb') as o:
                    o.write(r.content)
                    r.close()
                    if self.verbose:
                        print("Written cutout to %s"%outfile)
                        
    def fix_headers(self):
    # header is busted: fix it
        for filename in os.listdir(self.cutoutsdir):
            if filename[-4:]=="fits":
                print(filename)
                with fits.open(self.cutoutsdir+"/"+filename, mode='update') as hdul:
                    try:
                        test=hdul[0].header["HISTORY"]
                    except:
                        hdul[0].header["HISTORY"] = "PLACEHOLDER"
                    try:
                        test=hdul[0].header["BPA"]
                    except:
                        hdul[0].header["BPA"] = 90
                    try:
                        test=hdul[0]/header["CUNIT4"]
                    except:
                        hdul[0].header["CUNIT4"] = 'Hz'                
                print(hdul[0].header["BMIN"],hdul[0].header["BMAJ"])
    

def readArguments():
    parser=argparse.ArgumentParser("Calculate visibility imagin weights based on calibration quality")
    parser.add_argument("-v","--verbose",          help="Be verbose, say everything program does.",\
                        required=False,action="store_true")
    parser.add_argument("--catalog",  type=str,   help="Name of the catalog to be read.", required=True)
    parser.add_argument("--cutoutdir", type=str, help="Name of the directory in which to write cutours",required=False,default="Cutouts")
    parser.add_argument("--size", type=float, help="Size of individual cutours in arcmin",required=False,default=30)
    parser.add_argument("--obstime", type=str, help="Reference time for proper motion calculations. Default is 2000-01-01 12:00:00",
                        default="2000-01-01 12:00:00",required=False)
    parser.add_argument("--decfilter", type=str,help="Declination filter to apply. All sources below this declination will be filtered out.",
                        default=-5,required=False)
    parser.add_argument("--namekey",type=str,help="Key for the column containing source names in your catalog.",default="Name",required=False)
    parser.add_argument("--RAkey",type=str,help="Key for the column containing RA in your catalog.",default="RA_deg",required=False)
    parser.add_argument("--Deckey",type=str,help="Key for the column containing Dec in your catalog.",default="DE_deg",required=False)
    parser.add_argument("--RA_pm_key",type=str,help="Key for the column containing RA proper motion in your catalog.",default="muRA_masa-1",required=False)
    parser.add_argument("--Dec_pm_key",type=str,help="Key for the column containing RA proper motion in your catalog.",default="muDE_masa-1",required=False)
    parser.add_argument("--distkey",type=str,help="Key for the column containing distances in pc in your catalog.",default="d_pc",required=False)
    args=parser.parse_args()
    return vars(args)


                
if __name__=="__main__":
    start_time     = time.time()
    # read arguments
    args           = readArguments()
    verb           = args["verbose"]
    catalog        = args["catalog"]
    cutoutsdir     = args["cutoutdir"]
    size           = args["size"]
    obstime        = args["obstime"]
    decfilter      = args["decfilter"]
    namekey        = args["namekey"]
    rakey          = args["RAkey"]
    deckey         = args["Deckey"]
    ra_pm_key      = args["RA_pm_key"]
    dec_pm_key     = args["Dec_pm_key"]
    dist_key       = args["distkey"]

    if os.path.isdir(cutoutsdir) == False:
        print(cutoutsdir,os.path.isdir(cutoutsdir))
        os.mkdir(cutoutsdir)
    cutouts = GetCutouts(catalog,cutoutsdir,size,obstime,decfilter,namekey,rakey,deckey,ra_pm_key,dec_pm_key,dist_key,verbose=verb)
    cutouts.SetLoTSSReferenceTime()
    #cutouts.PrintCatHeader() # debug option
    cutouts.download()
    cutouts.UpdatePositions()
    cutouts.fix_headers()
