from casacore.tables import table
import numpy as np
import os
import argparse

def readArguments():
    parser=argparse.ArgumentParser("Fix the incorrect polarisation labeling of SPAM-calibrated GMRT datasets from RR to I")
    parser.add_argument("-v","--verbose",          help="Be verbose, say everything program does.",\
                        required=False,action="store_true")
    parser.add_argument("--filename",  type=str,   help="Name of the measurement set to "+\
                        "be corrected.", required=True,nargs="+")
    args=parser.parse_args()
    return vars(args)

def FixDataset(filename,verb):
    outfilename = filename.replace('RR','I')
    os.sys('cp -r %s %s'%(filename,outfilename))
    
    t=tables.table(outfilename+"/POLARIZATION", readonly=False,ack=verb)
    t.putcol("CORR_TYPE",np.array([[1]],dtype=np.int32))
    t.close()

if __name__=="__main__":
    # read arguments
    args           = readArguments()
    verb           = args["verbose"]
    mslist         = args["filename"]
    
    for ms in mslist:
        FixDataset(ms,v)

