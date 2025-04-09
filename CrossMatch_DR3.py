import numpy as np
import pandas as pd
from casacore.tables import table
from astropy.io import fits
from astropy.coordinates import SkyCoord, Distance
from astropy.time import Time
import astropy.units as u
import pylab
import os
import bdsf
from astroquery.simbad import Simbad
from aplpy import FITSFigure
from astropy import wcs
import time
import argparse

### this script looks for all the cutouts in the cutoutdir, runs pybdsf on each, and attempts to find a cross-match
# example invocation:
# time python3 CrossMatch_DR3.py -v --catalog carmencita.107.lotss_positions.csv --cutoutdir Cutouts_DR3 --catalogsdir FieldCatalogs_DR3 --diagdir Cutouts_DR3 --distance 6

class CrossMatch():
    """
    make docstrings
    """
    def __init__(self,
                 catalog,
                 xmatch_distance,
                 cutoutsdir,
                 catalogsdir,
                 diagdir,
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
        self.xmatch_dist     = xmatch_distance
        # only csv supported right now
        if catalog.split(".")[-1]=="csv":
            if self.verbose:
                print("Catalog read as csv")
            self.catalog     = pd.read_csv(catalog)
        else:
            if self.verbose:
                print("Catalog not in .csv format : currently not supported")
                exit()
        self.cutoutsdir      = cutoutsdir
        self.catsdir         = catalogsdir
        self.diagdir         = diagdir
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

    def UpdatePositions(self,outname="carmencita.107.lotss_positions.crossmatched.csv"):
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
        self.coords    = SkyCoord(ra=np.array(RA_array)*u.deg,dec=np.array(dec_array)*u.deg)
        self.pm_errors = err_array
        self.catalog.to_csv(outname)
         

        
    def generate_fieldcatalogs(self):
        for i in range(len(self.catalog)):
            CutoutFileName = self.cutoutsdir+"/"+self.catalog.iloc[i][self.namekey].strip().replace(" ","_")+".fits"
            if os.path.isfile(CutoutFileName):
                # bdsf expects more than 2 axes because radio astronomers are not cowards. DR3 cutouts are cowards though
                with fits.open(CutoutFileName, mode='update') as hdul:
                    frequency = hdul[0].header["RESTFRQ"]
                    # perform pybdsf analysis to get catalog for this cutout if not already done
                    proc=bdsf.process_image(CutoutFileName,frequency=frequency)
                    catname = self.catsdir+"/"+self.catalog.iloc[i][self.namekey].strip().replace(" ","_") + ".cat.fits"
                    print(catname)
                    proc.write_catalog(outfile=catname,clobber=True,format="fits", bbs_patches="source")

    def xmatch_fields(self,outname="./carmencita.107.LoTSS.crossmatched.results.csv"):
        # start cross-matching
        self.j               = 0
        self.matches         = 0
        self.inlotss         = 0
        self.catindex        = np.arange(len(self.catalog)) # to return from subset catalog to main catalog, if useful
        self.lotss_flux      = np.zeros_like(self.catalog[self.namekey].values)
        self.lotss_flux_err  = np.zeros_like(self.catalog[self.namekey].values)
        self.lotss_compacity = np.zeros_like(self.catalog[self.namekey].values)
        self.lotss_rms       = np.zeros_like(self.catalog[self.namekey].values)
        self.lotss_distance  = np.zeros_like(self.catalog[self.namekey].values)
        for i in range(len(self.catalog)):
            # check if cutout exists based on name
            CutoutFileName = self.cutoutsdir+"/"+self.catalog.iloc[i][self.namekey].strip().replace(" ","_")+".fits"
            if os.path.isfile(CutoutFileName):
                self.inlotss+=1
                #create and plot region of association
                regionfilename = "thisplanet.reg"
                f=open(regionfilename,"w")
                f.write("# Region file format: DS9 version 4.1\n")
                f.write("global color=red dashlist=8 3 width=1"+\
                        " font=\"helvetica 10 normal roman\" select=1"+\
                        " highlite=1 dash=0 fixed=0 edit=1 move=1"+\
                        " delete=1 include=1 source=1\n")
                f.write("fk5\n")
                xerr = str( np.sqrt( (self.catalog.iloc[i][self.ra_pm_key]/1000*self.lotss_yrs)**2  + self.xmatch_dist**2) / 3600 )
                yerr = str( np.sqrt( (self.catalog.iloc[i][self.dec_pm_key]/1000*self.lotss_yrs)**2 + self.xmatch_dist**2) / 3600 )
                writestr = "ellipse %s %s %s 90"%(self.coords[i].to_string("hmsdms"), xerr,yerr)
                textstr="%s"%self.catalog.iloc[i][self.namekey]
                writestr+=" # text={"+textstr+"}"
                writestr+="\n"
                f.write(writestr)
                f.close()
                image=fits.open(CutoutFileName)
                del image[0].header["CUNIT4"]
                temp = FITSFigure(image)
                temp.show_grayscale(vmin=-0.000183801,vmax=0.000858944,invert=False)
                try:
                    temp.recenter(self.coords[i].ra,self.coords[i].dec,width=0.1,height=0.1)
                except:
                    print("Problem with the image of %s"%self.catalog.iloc[i][self.namekey].strip())
                    pylab.clf()
                    temp.close()
                    continue                    
                temp.add_colorbar()
                temp.colorbar.set_width(0.2)
                temp.colorbar.set_location("right")
                temp.show_regions(regionfilename)
                # get lotss coordinates and separations
                catname = self.catsdir + "/" + self.catalog.iloc[i][self.namekey].strip().replace(" ","_") + ".cat.fits"
                #catname = VLOTSScatalog # do vlotss
                pybdsfhdul = fits.open(catname)
                pybdsfcat  = pybdsfhdul[1].data
                pybdsfhdul.close()
                lotss      = pd.DataFrame(pybdsfcat)
                LotssCoords = SkyCoord(lotss["RA"].values,lotss["DEC"].values,unit=u.deg)
                seps = self.coords[i].separation(LotssCoords).arcsec
                # calculate distance for this proper motion
                ThisDistance = np.sqrt(self.xmatch_dist**2 + self.pm_errors[i]**2)
                # plot nearby LoTSS source fluxes and distances.
                fieldregname = "field.reg"
                f=open(fieldregname,"w")
                f.write("# Region file format: DS9 version 4.1\n")
                f.write("global color=cyan dashlist=8 3 width=1"+\
                        " font=\"helvetica 10 normal roman\" select=1"+\
                        " highlite=1 dash=0 fixed=0 edit=1 move=1"+\
                        " delete=1 include=1 source=1\n")
                f.write("fk5\n")
                if len(seps[seps<120])>0:
                    for sep in seps[seps<120]:
                        writestr = "circle %s %s"%((LotssCoords[seps==sep].to_string("hmsdms"))[0], ThisDistance/3600)
                        textstr="%.2f +- %.2f mJy"%((lotss["Total_flux"])[seps==sep].values[0]*1000,(lotss["E_Total_flux"])[seps==sep].values[0]*1000)
                        writestr+=" # text={"+textstr+"}"
                        writestr+="\n"
                        f.write(writestr)
                    f.close()
                    temp.show_regions(fieldregname)                    
                # plot nearby SIMBAD sources
                simbad_field = Simbad.query_region(self.coords[i], radius=240 * u.arcsec)
                time.sleep(0.3) # to avoid overloading simbad queries and being Punished
                simbadregname = "simbad.reg"
                f=open(simbadregname,"w")
                f.write("# Region file format: DS9 version 4.1\n")
                f.write("global color=magenta dashlist=8 3 width=1"+\
                        " font=\"helvetica 10 normal roman\" select=1"+\
                        " highlite=1 dash=0 fixed=0 edit=1 move=1"+\
                        " delete=1 include=1 source=1\n")
                f.write("fk5\n")
                simbad_field = Simbad.query_region(self.coords[i], radius=240 * u.arcsec)
                for simbad_source in simbad_field:
                    writestr = "point %s %s"%(simbad_source["ra"],simbad_source["dec"])#,pos_err)
                    textstr  = simbad_source["main_id"]
                    writestr+=" # text={"+textstr+"}"
                    writestr+="\n"
                    f.write(writestr)
                f.close()
                temp.show_regions(simbadregname)
                # cross-match
                if np.sum(seps<=ThisDistance)>0:
                    self.matches+=1
                    currdist = np.min(seps)
                    print("n_matches: ",np.sum(seps<=ThisDistance))
                    print(self.catalog.iloc[i])
                    # add to catalog
                    print()
                    self.lotss_flux[i]      = lotss["Total_flux"][seps==currdist].values[0]
                    self.lotss_flux_err[i]  = lotss["E_Total_flux"][seps==currdist].values[0]
                    self.lotss_compacity[i] = lotss["Peak_flux"][seps==currdist].values[0]/(lotss["Total_flux"])[seps==currdist].values[0]
                    self.lotss_rms[i]       = lotss["Isl_rms"][seps==currdist].values[0]
                    self.lotss_distance[i]  = currdist
                    print("LoTSS flux            : ",self.lotss_flux[i]*1000,"+-",self.lotss_flux_err[i]*1000, "mJy")
                    print("Compactness           : ",self.lotss_compacity[i])
                    print("RMS                   : ",self.lotss_rms[i]*1000, "mJy/bm")
                    print("Total matches thus far: %4i / %i"%(self.matches,len(self.catalog)))
                    
                    # if cross-match successful, update diagnostic plot with the cross-match info
                    assocregname = "association.reg"
                    f=open(assocregname,"w")
                    f.write("# Region file format: DS9 version 4.1\n")
                    f.write("global color=green dashlist=8 3 width=1"+\
                            " font=\"helvetica 10 normal roman\" select=1"+\
                            " highlite=1 dash=0 fixed=0 edit=1 move=1"+\
                            " delete=1 include=1 source=1\n")
                    f.write("fk5\n")
                    writestr = "circle %s %s"%((LotssCoords[seps==currdist].to_string("hmsdms"))[0], ThisDistance/3600)
                    textstr="%.2f +- %.2f mJy"%((lotss["Total_flux"])[seps==currdist].values[0]*1000,(lotss["E_Total_flux"])[seps==currdist].values[0]*1000)
                    writestr+=" # text={"+textstr+"}"
                    writestr+="\n"
                    f.write(writestr)
                    f.close()
                    temp.show_regions(assocregname)
                    print()
                else:
                    print(i,"No match for %s"%self.catalog.iloc[i][self.namekey]) # todo
                fieldimgname = self.diagdir+"/"+self.catalog.iloc[i][self.namekey].strip().replace(" ","_")+".png"
                pylab.savefig(fieldimgname)
                pylab.clf()
                temp.close()
        # finalise diagnostics
        if self.verbose:
            print("Total matches, with a separation of %i arcsec: "%self.xmatch_dist, self.matches)
            print("Total CARMENES sources in LoTSS:",self.inlotss)
        self.catalog["LoTSS_flux_Jy"]   = self.lotss_flux
        self.catalog["e_LoTSS_flux_Jy"] = self.lotss_flux_err
        self.catalog["LoTSS_compacity"] = self.lotss_compacity
        self.catalog["LoTSS_distance"]  = self.lotss_distance
        self.catalog["LoTSS_RMS"]       = self.lotss_rms
        self.catalog.to_csv(outname)
        if self.verbose:
            print("Saved cross-match values to %s"%outname)

def readArguments():
    parser=argparse.ArgumentParser("Calculate visibility imagin weights based on calibration quality")
    parser.add_argument("-v","--verbose",          help="Be verbose, say everything program does.",\
                        required=False,action="store_true")
    parser.add_argument("--catalog",  type=str,   help="Name of the catalog to be read.", required=True)
    parser.add_argument("--distance",type=float,help="Cross-matching distance desired, in arcsec.",required=False, default=5)
    parser.add_argument("--cutoutdir", type=str, help="Name of the directory in which to write cutouts",required=False,default="Cutouts")
    parser.add_argument("--catalogsdir", type=str, help="Name of the directory in which to write pybdsf catalogs",required=False,default="FieldCatalogs")
    parser.add_argument("--diagdir", type=str, help="Name of the directory in which to write diagnostic pngs",required=False,default="Diagnostics")
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
    args                   = readArguments()
    verb                   = args["verbose"]
    catalog                = args["catalog"]
    xmatch_distance_arcsec = args["distance"]
    cutoutsdir             = args["cutoutdir"]
    catalogsdir            = args["catalogsdir"]
    diagsdir               = args["diagdir"]
    namekey                = args["namekey"]
    rakey                  = args["RAkey"]
    deckey                 = args["Deckey"]
    ra_pm_key              = args["RA_pm_key"]
    dec_pm_key             = args["Dec_pm_key"]
    distkey                = args["distkey"]
    
    crossmatch = CrossMatch(catalog,
                            xmatch_distance_arcsec,
                            cutoutsdir,
                            catalogsdir,
                            diagsdir,
                            namekey,
                            rakey,
                            deckey,
                            ra_pm_key,
                            dec_pm_key,
                            distkey,
                            verb)
    crossmatch.SetLoTSSReferenceTime()
    crossmatch.UpdatePositions()
    #crossmatch.generate_fieldcatalogs()
    crossmatch.xmatch_fields()
    


