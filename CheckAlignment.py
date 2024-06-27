import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_from_healpix, reproject_to_healpix

# set below to false if we dont want to redo the planck maps
#reproducePlanckPaperMaps=True
reproducePlanckPaperMaps=False

# set below to false if we don't want to create reprojected healpix to fits file
ReprojectToFits=True
#ReprojectToFits=False


#basefitsfile='M31.NenuFAR.1GC.realigned.PURGED.fits'

basefitsfile='HiresM31_Oiii_Arc_Yann_Sainty_Marcel_Drechsler_wcs.fits'

def AnnotateHPmap(pltobj,hp,skycoords,annotation):
    color='black'
    width=0.1
    headwidth=0.1
    headlength=1
    proj = hp.projector.MollweideProj() # add rot=(M31coords.l.degree, M31coords.b.degree,0) to the argument if rotating maps
    pltobj.annotate(annotation,
                    xy=proj.ang2xy(skycoords.l.degree,skycoords.b.degree,lonlat=True),
                    xytext=(proj.ang2xy((skycoords.l.degree-15),(skycoords.b.degree+15),lonlat=True)),
                    arrowprops=dict(color=color,width=width,headwidth=headwidth,headlength=headlength)
                    )

def ReprojectToHDU(hdu,hpy_map,outname,hduiter=0):
#    hdu = fits.open(basefitsfile)
    target_header     = WCS(hdu[hduiter].header).sub([1,2])
    target_shape=target_header.array_shape
    target_data       = hdu[hduiter]#.data[1,:,:]
    array, footprint  = reproject_from_healpix((hpy_map,'galactic'), target_header, shape_out=target_shape, nested=False)
    hdu[hduiter].data = array
    hdu.writeto(outname,overwrite=True)

    
# define M31 skycoords
M31coords = SkyCoord('0h42m44s','+41d16m09s',frame='icrs')
# convert to galactic
M31coords = M31coords.galactic

#reproduce Planck cmap
from matplotlib.colors import ListedColormap
colombi1_cmap = ListedColormap(np.loadtxt("Planck_Parchment_RGB.txt")/255.)
colombi1_cmap.set_bad("gray") # color of missing pixels
colombi1_cmap.set_under("white") # color of background, necessary if you want to use
# this colormap directly with hp.mollview(m, cmap=colombi1_cmap)
cmap = colombi1_cmap

# set below to false if we dont want to redo the planck maps
reproducePlanckPaperMaps=True

#read file
filename='COM_CompMap_IQU-thermaldust-gnilc-unires_2048_R3.00.fits'
planck_map = hp.read_map(filename)*287.5
planck_map_i = hp.read_map(filename,field=0)*287.5
planck_map_q = hp.read_map(filename,field=1)*287.5
planck_map_u = hp.read_map(filename,field=2)*287.5



planck_map_pfrac = np.sqrt(planck_map_q**2 + planck_map_u**2)/planck_map_i
planck_map_p      = np.sqrt(planck_map_q**2 + planck_map_u**2)/planck_map_i
planck_map_polang = np.arctan2(-planck_map_u,planck_map_q)

# add M31 coordinates
proj = hp.projector.MollweideProj()

# planck paper; cf sec. 3 https://www.aanda.org/articles/aa/pdf/2020/09/aa33885-18.pdf
if reproducePlanckPaperMaps:
    hp.mollview(np.log10(planck_map_i),cmap=cmap,min=-1.49632,max=1.92835,unit=r'log(I [MJy sr$^-1$]',cbar=True)
#                rot=(M31coords.l.degree, M31coords.b.degree,0))
    AnnotateHPmap(plt,hp,M31coords,"M31")
    
    hp.graticule()
    plt.savefig('planck_map_I.png')
    plt.clf()
    print("Made I map")

    hp.mollview(planck_map_q,cmap=cmap,min=-0.2,max=0.2,unit=r'log(Q [MJy sr$^-1$]',cbar=True)
    AnnotateHPmap(plt,hp,M31coords,"M31")
    hp.graticule()
    plt.savefig('planck_map_Q.png')
    plt.clf()
    print("Made Q map")
    

    hp.mollview(planck_map_u,cmap=cmap,min=-0.2,max=0.2,unit=r'log(U [MJy sr$^-1$]',cbar=True)
    AnnotateHPmap(plt,hp,M31coords,"M31")
    hp.graticule()
    plt.savefig('planck_map_U.png')
    plt.clf()
    print("Made U map")

    # based on https://github.com/healpy/healpy/pull/617#issue-434041253
#    planck_LIC=hp.read_map("planck_LIC.fits") # made from comments below.
    Qsmooth = hp.smoothing(planck_map_q, np.deg2rad(1))
    Usmooth = hp.smoothing(planck_map_u, np.deg2rad(1))
    planck_LIC = hp.line_integral_convolution(Q=Qsmooth, U=Usmooth)
    planck_LIC = hp.smoothing(planck_LIC, np.deg2rad(1))
    hp.write_map("planck_LIC.fits",planck_LIC,overwrite=True)
    
    #hp.mollview(np.log(1 + np.sqrt(planck_map_q**2 + planck_map_u**2) * 100), cmap='inferno', cbar=False)
    #hp.mollview(planck_LIC, cmap=cmap, cbar=False, reuse_axes=True, title='WMAP K')
    hp.mollview(planck_LIC, cmap=cmap, cbar=False, title='WMAP K')
    AnnotateHPmap(plt,hp,M31coords,"M31")
    hp.graticule()
    plt.savefig('planck_LIC.png')
    plt.clf()

    #hp.write_map("planck_LIC.fits",planck_LIC)

#    planck_LIC=hp.read_map("planck_LIC.fits")
    
    print("Made LIC map")
    
    
    hp.mollview(np.log10(np.sqrt(planck_map_q**2 + planck_map_u**2)),cmap=cmap,min=-4,max=0.1,
                unit=r'log(P [MJy sr$^-1$]',cbar=True)
    AnnotateHPmap(plt,hp,M31coords,"M31")
    hp.graticule()
    plt.savefig('planck_map_P.png')
    plt.clf()
    print("Made P map")

if ReprojectToFits:
    hdu = fits.open(basefitsfile)
    target_header = WCS(hdu[0].header)

    ReprojectToHDU(hdu,planck_map_i,"planck_I.fits")
    ReprojectToHDU(hdu,planck_map_q,"planck_Q.fits")
    ReprojectToHDU(hdu,planck_map_u,"planck_U.fits")
    ReprojectToHDU(hdu,planck_map_p,"planck_P.fits")
    ReprojectToHDU(hdu,planck_map_pfrac,"planck_p.fits")
    ReprojectToHDU(hdu,planck_LIC,"planck_LIC.fits")
#    planck_LIC=hp.read_map("planck_LIC.fits")


    
#    array, footprint = reproject_from_healpix((planck_map_i,'galactic'), target_header, nested=False)
#    hdu[0].data = array
#    plt.imshow(array, interpolation='nearest')
#    hdu.writeto("planck_i.fits",overwrite=True)

    
    
