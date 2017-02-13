# This is a script that reads hexadecimal RA, Dec and a list of fits files
# and outputs one png cutout per fits centred at the coordinate. By default,
# these cutouts will be 1 degree by 1 degree.

# import libraries
import matplotlib
matplotlib.use("Agg")
import aplpy
import numpy as np
import sys
import glob

def MakeCutout(fitsfiles,RA,Dec,xDegrees=0.1,yDegrees=0.1,outname=None):
    RAdeg  = HHMMSStoDegrees(RA)*15. # this converts RA from hours to degrees
    Decdeg = HHMMSStoDegrees(Dec)
    # find vmin, vmax
    

    for ffile in fitsfiles:
        temp = aplpy.FITSFigure(ffile)
        temp.show_grayscale(vmin=-5.989e-02,vmax=5.991e-01)
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
    fitsfiles=sys.argv[1:-2]
    print "List of .fits: ",fitsfiles
    RA=sys.argv[-2]
    Dec=sys.argv[-1]
    print "RA  : %s"%RA
    print "Dec : %s"%Dec

    # make some control loop to say that RA, Dec need to be given in hexadecimal
    
    MakeCutout(fitsfiles,RA,Dec,xDegrees=0.1,yDegrees=0.1,outname=None)
