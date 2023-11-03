### script to subtract one model data column
### from one corrected data column for a set
### of measurement sets.

from casacore.tables import table
import glob
import numpy as np


datadir  = '/path/to/data/directory'
mslist   = glob.glob(datadir+"*MS")
modelcol = 'WIDEFIELD_DATA'
datacol  = 'CORRECTED_DATA'
outcol   = 'TARGET_DATA'


for msname in mslist:
    print('Doing %s'%msname)
    ms = table(msname,readonly=False,ack=False)
    d  = ms.getcol(datacol)-ms.getcol(modelcol)
    if outcol not in ms.colnames():
        desc=ms.getcoldesc(datacol)
        desc["name"]=outcol
        desc['comment']=desc['comment'].replace(" ","_")
        ms.addcols(desc)
    ms.putcol(outcol,d)
    ms.close()
