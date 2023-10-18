import glob
import os
import numpy as np
from casacore.tables import table


# define working directory and mslist
pwd    = '/data/LOFAR/LBA/DATA/'
mslist = glob.glob(pwd+'*MS')

for ms in mslist:
    # read the target
    target=table(ms+'/OBSERVATION',ack=False).getcol('LOFAR_TARGET')['array'][0]
    print('Placing dataset %s in %s'%(ms,target))
    # create if not already there
    if not os.path.isdir(pwd+target):
        os.mkdir(pwd+target)
    # organise
    os.system('mv %s %s'%(ms,pwd+target))
