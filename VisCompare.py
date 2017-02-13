import glob
import os
import sys
import subprocess
import getpass
import numpy as np
from pyrap.tables import table
import pylab

def VisCompare(MSlistFilename,concatMS="/data/etienne.bonnassieux/GrothStrip/28_08_2014/DATA/concat.sb020.sb029.MS/"):
    # read concatMS data
    cMS=table(concatMS)
    concatdata=cMS.getcol("DATA")
    cMS.close()
    # read MSlist
    mslist=[]
    for line in open(MSlistFilename):
        mslist.append(line.rstrip())
    # read mslist data
    listdata=[]
    for ms in mslist:
        t=table(ms)
        listdata.append(t.getcol("CORRECTED_DATA"))
        t.close()
    print "all data read; calculating difference"
    # compare data
#    print np.sum(np.abs(concatdata-listdata))
    pylab.clf()
    pylab.plot((concatdata-listdata)[::100])
    pylab.show()
