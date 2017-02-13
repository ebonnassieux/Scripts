import sys
import pyrap.images
import os

def go(imlist,outname):
    # Name of the output image
    outimg = outname
    # Read pixel data of first image.
    img = pyrap.images.image(imlist[0])
    pixels = img.getdata()
    print "Creating average image .",
    sys.stdout.flush()
    # Add pixel data of other images one by one.
    for name in imlist[1:]:
        tmp = pyrap.images.image(name)
        pixels += tmp.getdata()
        del tmp
        print ".",
        sys.stdout.flush()
    # Write averaged pixel data; have to create image file to write as fits
    img.saveas(outimg + ".img")
    img = pyrap.images.image(outimg + ".img")
    img.putdata(pixels / float(len(imlist)))
    img.tofits(outimg + ".fits")
    del img
    # get rid of the useless shite
    os.system("rm -rf %s.img"%outimg)
    print "done."

if __name__=="__main__":
    imlist=sorted(sys.argv[1:-1])
    outname=sys.argv[-1]
    print "Fits filename list: ", imlist
    print "Averaged fits filename: ",outname+".fits"
    go(imlist,outname)
