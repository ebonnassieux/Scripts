import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord,  Angle, concatenate
from astropy.io import fits
from reproject import reproject_from_healpix, reproject_to_healpix
import pandas as pd
from mocpy import MOC, WCS
from astropy import units as u
from healpy.newvisufunc import projview, newprojplot
from regions import CircleSkyRegion

def MakeMOCFromSkyCoords(coords,radiusDeg=5,color='red',maxdepth=19):
    # script to generate a MOC of circles of
    # requested radius at each of the specified
    # coordinates.
    mocobj = MOC.from_astropy_regions(CircleSkyRegion(center=coords[0], radius=radiusDeg * u.deg),maxdepth)
    maxi=len(coords)
    i=0
    for coord in coords:
        thisregmoc = MOC.from_astropy_regions(CircleSkyRegion(center=coord, radius=radiusDeg * u.deg),maxdepth)
        mocobj=mocobj.union(thisregmoc)
        i=i+1
        print("done %4i of %4i"%(i,maxi),end="\r")
    print()
    return mocobj

def read_coords(filename):
    coords=[]
    f=open(filename)
    for line in f.readlines():
        coords.append(SkyCoord(float(line.split("\t")[0])*u.deg, float(line.split("\t")[1][:-2])*u.deg,frame="icrs"))
    outcoords = concatenate(coords)
    return outcoords
        

# read coords and MOCS
dr2moc = MOC.from_fits("dr2-moc.moc")
vlotss      = pd.DataFrame(fits.open("../vlotss_dr2_cat_accepted.fits")[1].data)
carmencita  = pd.read_csv("carmencita.107.lotss_positions.csv")
# define coords
carmencita_coords = SkyCoord(carmencita["RA_LoTSS_deg"].values*u.deg,carmencita["DEC_LoTSS_deg"].values*u.deg,frame="icrs")
vlotss_coords     = SkyCoord(vlotss["RA_i"].values*u.deg,vlotss["Dec_i"].values*u.deg,frame="icrs") 
# convert catalogues to MOCs
vlotss_moc     = MOC.from_skycoords(vlotss_coords,max_norder=29)
carmencita_moc = MOC.from_skycoords(carmencita_coords,max_norder=29)

# test makemoc
#test_coords=read_coords("dr3_pointings.txt")
#print("read coords")
#testmoc=MakeMOCFromSkyCoords(test_coords,radiusDeg=1.7,color='red')
#testmoc.save("dr3-moc.moc",format="fits",overwrite=True)

dr3moc = MOC.from_fits("dr3-moc.moc")

fig = plt.figure(figsize=(20, 10))

plt.rcParams.update({'font.size': 18})
fov = 170
# WCS setup
with WCS(
    fig,
    fov=fov * u.deg,
    center=SkyCoord(180, 0, unit="deg", frame="icrs"),
    coordsys="icrs",
    rotation=Angle(0, u.degree),
    projection="AIT",
) as wcs:
    ax = fig.add_subplot(1, 1, 1, projection=wcs)
    ax.set_frame_on(False)
# Fill and border for each MOC
dr2moc.fill(ax=ax, wcs=wcs, alpha=0.1, fill=True, color="green", label="LoTSS-DR2",lw=0)
dr2moc.border(ax=ax, wcs=wcs, alpha=0.8, color="black")
dr3moc.fill(ax=ax, wcs=wcs, alpha=0.2, fill=True, color="Green", label="LoTSS-DR3",lw=0)
dr3moc.border(ax=ax, wcs=wcs, alpha=0.5, color="black")
vlotss_moc.fill(ax=ax, wcs=wcs, alpha=1, fill=True, color="Red", label="V-LoTSS",lw=3)
vlotss_moc.border(ax=ax, wcs=wcs, alpha=1, color="black")
carmencita_moc.fill(ax=ax, wcs=wcs, alpha=1, fill=True, color="Blue", label="Carmencita")
carmencita_moc.border(ax=ax, wcs=wcs, alpha=1, color="black")


#testmoc.fill(ax=ax, wcs=wcs, alpha=0.1, fill=True, color="Red", label="testmoc",lw=0)
#testmoc.border(ax=ax, wcs=wcs, alpha=0.5, color="black")


# Plot details
plt.xlabel("RA")
plt.ylabel("DEC")
plt.title("Survey Coverages")
plt.legend()
plt.grid(color="black", linestyle="dotted")
plt.tight_layout()
plt.savefig("surveys_coverage.png")
plt.show()
