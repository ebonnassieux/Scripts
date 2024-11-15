import os
from casacore.tables import table
import numpy as np
import pylab
import sys
import warnings
import time
import argparse
from enum import Enum

class PolEnum(Enum):
    """
    Returns the NRAO Stokes class enumeration, as defined here:
    https://casa.nrao.edu/active/docs/doxygen/html/classcasa_1_1Stokes.html
    """
    UNDEFINED = 0
    I         = 1
    Q         = 2
    U         = 3
    V         = 4
    RR        = 5
    RL        = 6
    LR        = 7
    LL        = 8
    XX        = 9
    XY        = 10
    YX        = 11
    YY        = 12
    RX        = 13
    RY        = 14
    LX        = 15
    LY        = 16
    XR        = 17
    XL        = 18
    YR        = 19
    YL        = 20
    PP        = 21
    PQ        = 22
    QP        = 23
    QQ        = 24
    RCircular = 25
    LCircular = 26
    Linear    = 27
    Ptotal    = 28
    Plinear   = 29
    PFtotal   = 30
    PFlinear  = 31
    Pangle    = 32    

def ReturnPolBases(inbasis,ncorrs):
    """
    Function which returns the labels for a given basis, for
    the appropriate number of correlations.
    Bases currently accepted are XY, RL, I.
    """
    if inbasis=="XY":
        if ncorrs==2:
            returnbases = ["XX","YY"]
        elif ncorrs==4:
            returnbases = ["XX","XY","YX","YY"]
        else:
            raise Exception("Basis not supported - XY basis requires 2 or 4 ncorrs.")
    elif inbasis=="RL":
        if ncorrs==2:
            returnbases = ["RR","LL"]
        elif ncorrs==4:
            returnbases = ["RR","RL","LR","LL"]
        else:
            raise Exception("Basis not supported - RL basis requires 2 or 4 ncorrs.")
    elif "I" in inbasis:
        if ncorrs==1:
            returnbases = ["I"]
        elif ncorrs==4:
            returnbases = ["I","Q","U","V"]
        else:
            raise Exception("Basis not supported - I basis requires either 1 or 4 ncorr (I, IQUV).")
    else:
        raise Exception("Basis not supported - please use XY, RL, or IQUV")
    return returnbases



class MSpol:
    """
    docstring goes here
    """

    # initialise the class
    def __init__(self, msname, inbasis=None, outbasis=None, verbose=False):
        self.verbose          = verbose
        cwd                   = os.getcwd()+"/"
        if msname[-1]=="/":
            self.msname       = msname[0:-1]
        else:
            self.msname       = msname
        if self.msname[0]!="/":
            self.msname       = cwd+msname
        self.inbasis          = inbasis
        self.outbasis         = outbasis
        if self.outbasis != None:
            self.readonly     = False
        else:
            self.readonly     = True
        # Stokes class enum from https://casa.nrao.edu/active/docs/doxygen/html/classcasa_1_1Stokes.html
        self.PolEnum          = PolEnum
        
    def FindPolBasis(self):
        """
        This function defines the polarisation basis to use as an input.
        If an inbasis is defined for the class, this overtakes the actual
        basis defined in the MS metatada.
        """
        if self.verbose:
            print("Running polarisation diagnostic on %s"%ms)
        # get pol info
        self.poltable = table(self.msname+"/POLARIZATION",readonly=self.readonly,ack=self.verbose)
        self.corrnums=np.concatenate(self.poltable.getcol("CORR_TYPE"))
        self.ncorrs = len(self.corrnums)
        self.corrtypes = []
        for corrnum in self.corrnums:
            self.corrtypes.append(self.PolEnum(corrnum).name)
        if self.inbasis!=None and self.inbasis in self.corrtypes and self.verbose==True:
                print("MS polarisation basis same as requested input:",self.corrtypes)
        elif self.inbasis!=None and self.inbasis not in self.corrtypes:
            if self.verbose==True:
                print("MS polarisation basis NOT the same as requested input.")
                print("Using requested input as the true value.")
            self.newpolbase = ReturnPolBases(self.inbasis,len(self.corrnums))
            newpolnums = []
            for polbase in self.newpolbase:
                newpolnums.append(self.PolEnum[polbase].value)
            self.corrnums = newpolnums
            newcorrtypes = []
            for corrnum in self.corrnums:
                newcorrtypes.append(self.PolEnum(corrnum).name)
            self.corrtypes = newcorrtypes
        if self.verbose==True:
            print("Input polarisation basis:", self.corrtypes)
        self.basis = self.corrtypes[0]

    def ChangeBasis(self):
        """
        Function to change the polarisation basis of the MS, from the input
        to the desired output. If input and output bases are the same, only
        the header will effectively be changed.
        """
        # build stokes from pol info
        if self.outbasis==None:
            if self.verbose:
                print("No output basis provided: no correction will be done.")
        else:
            # create output fle
            self.outfilename = self.msname.split(".MS")[0]+"."+str(self.outbasis)+".MS"
            os.system("cp -r %s %s"%(self.msname,self.outfilename))
            self.outms       = table(self.outfilename,readonly=False,ack=self.verbose)
            self.outpoltable = table(self.outfilename+"/POLARIZATION",readonly=False,ack=self.verbose)
            # define outpolnums
            self.outpolbase = ReturnPolBases(self.outbasis,len(self.corrnums))
            newpolnums = []
            for polbase in self.outpolbase:
                newpolnums.append(self.PolEnum[polbase].value)
            self.outpolnums = newpolnums
            # change header
            self.outcorrnums = np.array([self.outpolnums],dtype=np.int32)
            self.outpoltable.putcol("CORR_TYPE",self.outcorrnums)
            if self.verbose:
                print("Edited metadata: %s is now in basis %s"%(self.outfilename,self.outbasis))
            if self.ncorrs==1:
                if self.verbose:
                    print("Only 1 correlation recorded: there is no conversion to be applied to the data.")
            elif self.ncorrs==2:
                if self.verbose:
                    print("Only 2 correlations recorded: there is no conversion to be applied to the data.")
            elif self.ncorrs==4:
                if self.verbose:
                    print("4 correlations recorded: the data need to be reorganised.")
                self.outms = table(self.outfilename,readonly=False,ack=self.verbose)
                # calculate the new data for each data column
                # based on https://www.atnf.csiro.au/computing/software/atca_aips/node11.html
                for colname in self.outms.colnames():
                    if "DATA" in colname:
                        if colname!="DATA_DESC_ID":
                            d = self.outms.getcol(colname)
                            # create Stokes from correlations
                            if self.basis=="XX":
                                stokesI = 0.5 *    ( d[:,:,0] + d[:,:,3] )
                                stokesQ = 0.5 *    ( d[:,:,0] - d[:,:,3] )
                                stokesU = 0.5 *    ( d[:,:,1] + d[:,:,2] )
                                stokesV = -0.5*1j* ( d[:,:,1] - d[:,:,2] )
                            elif self.basis=="RR":
                                stokesI = 0.5 *    ( d[:,:,0] + d[:,:,3] )
                                stokesQ = 0.5 *    ( d[:,:,1] + d[:,:,2] )
                                stokesU = -0.5*1j* ( d[:,:,1] - d[:,:,2] )
                                stokesV = 0.5 *    ( d[:,:,0] - d[:,:,3] )
                            elif self.basis=="I":
                                stokesI = d[:,:,0]
                                stokesQ = d[:,:,1]
                                stokesU = d[:,:,2]
                                stokesV = d[:,:,3]
                            # create new correlations from Stokes
                            if self.outbasis=="XY":
                                d[:,:,0] = stokesI + stokesQ
                                d[:,:,1] = stokesU + stokesV*1j
                                d[:,:,2] = stokesU - stokesV*1j
                                d[:,:,3] = stokesI - stokesQ
                            elif self.outbasis=="RL":
                                d[:,:,0] = stokesI + stokesV
                                d[:,:,1] = stokesQ + stokesU*1j
                                d[:,:,2] = stokesQ - stokesU*1j
                                d[:,:,3] = stokesI - stokesV
                            elif self.outbasis=="IQUV":
                                test=1
                            self.outms.putcol(colname,d)
                            if self.verbose:
                                print("Reorganised column %s of MS %s"%(colname,self.outfilename))
                        
#        print(self.ms.colnames())


            
    def ReadMetadata(self):
        self.FindPolBasis()
        ### other things to read and save:
        # pointing and phase centres in hex
        # obs time, dt , nt
        # obs frequency, dnu, nchan
        # number of antennas

    def close(self):
        """
        This just closes all tables cleanly.
        """
        try:
            self.poltable.close()
            self.ms.close()
            self.outms.close()
            self.outpoltable.close()
        except:
            pass

def readArguments():
    parser=argparse.ArgumentParser("Read out metadata for a MS file and, if requested, change its basis")
    parser.add_argument("-v","--verbose",          help="Be verbose, say everything program does.",\
                        required=False,action="store_true")
    parser.add_argument("--filename",  type=str,   help="Name of the measurement set(s) to "+\
                        "be diagnosed", required=True,nargs="+")
    parser.add_argument("--inbasis",  type=str,   help="If you know the metadata for your MS file(s) is. "+\
                        "incorrect, set this to the true basis expected, e.g. IQUV. Default is None.",required=False,default=None)
    parser.add_argument("--outbasis",  type=str,   help="Basis which you want to convert your MS file(s) to. "+\
                        "Default is None.",required=False,default=None)
    args=parser.parse_args()
    return vars(args)

if __name__=="__main__":
    start_time     = time.time()
    # read arguments
    args           = readArguments()
    verb           = args["verbose"]
    mslist         = args["filename"]
    inbasis        = args["inbasis"]
    outbasis       = args["outbasis"]

    for ms in mslist:
        CheckMS = MSpol(msname=ms, inbasis=inbasis, outbasis=outbasis, verbose=verb)
        CheckMS.ReadMetadata()
        if outbasis!=None:
            CheckMS.ChangeBasis()
        CheckMS.close()
