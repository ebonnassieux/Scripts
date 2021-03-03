from pyrap.tables import table
from pyrap.tables import tablecopy
import numpy as np
import argparse
import os


def readArguments():
    # create parser
    parser=argparse.ArgumentParser(description="Rephase a set of measurement sets to the target coordinates.")
    parser.add_argument("-v","--verbose",help="Be verbose, say everything program does. Default is False",required=False,action="store_true")
    parser.add_argument("--msname",type=str,help="Name of the measurement sets you want to rephase.",required=True,nargs="+")
    parser.add_argument("--outputstr",type=str,help="String to add to the output measurement set name. Default is \"rephased\". "+\
                        "Setting to \"\" will overwrite input measurement set.",required=False,default="rephased")
    parser.add_argument("--RA",metavar="HH:MM:SS",type=str, help="Target phase centre Right Ascension in hexadecimal hour angle. "+\
                        "If not provided, defaults to phase centre RA of the first measurement set provided.",required=False,default="")
    parser.add_argument("--DEC",metavar="DD:MM:SS",type=str,help="Target phase centre Declination in hexadecimal degrees. "+\
                        "If not provided, defaults to phase centre RA of the first measurement set provided.",required=False,default="")
    parser.add_argument("--FieldID",type=int,help="FieldID index you want to apply the rephasing to. Default is 0.",required=False,default=0)
    parser.add_argument("--SPW",type=int,help="Spectral Window index you want to apply the rephasing to. Default is 0.",required=False,default=0)
    parser.add_argument("--rephase",type=bool,help="Rephase the visibilities. Default is True, but can be turned off to apply astrometric corrections.",
                        required=False, default=True)
    # parse
    args=parser.parse_args()
    return vars(args)

def HHMMSS2rad(HHMMSS):
    hhstr,mmstr,ssstr=HHMMSS.split(":")
    hang=float(hhstr)+float(mmstr)/60.+float(ssstr)/3600.
    # convert to rads
    #return hang*2.*np.pi/23.9345
    return hang/11.96725*np.pi

def DDMMSS2rad(DDMMSS):
    ddstr,mmstr,ssstr=DDMMSS.split(":")
    hang=int(ddstr)+int(mmstr)/60.+float(ssstr)/3600.
    # convert to rads
    return hang/180.*np.pi

    
def rephaseMS(msname, targetradRA, targetradDEC,outputstr="rephased",fieldid=0,spw=0,v=False,rephase=True):
    if v: print("Phase shifting %s"%msname)
    if outputstr!="":
        # make output measurement set
        outputmsname=msname[0:-3]+"."+outputstr+".ms"
        if v: print("Creating output measurement set %s"%outputmsname)
        tablecopy(msname,outputmsname,deep=True)
    # change phase centre
    t=table(outputmsname+"/FIELD",readonly=False,ack=False)
    phasecentr=t.getcol("PHASE_DIR")
    refradRA=phasecentr[fieldid,spw,0]
    refradDEC=phasecentr[fieldid,spw,1]
    phasecentr[fieldid,spw,0]=targetradRA
    phasecentr[fieldid,spw,1]=targetradDEC
    t.putcol("PHASE_DIR",phasecentr)
    t.close()
    if rephase==True:
        if v:print("Phase angle reference changed, phase shifting data")
        # get freq information
        t=table(msname+"/SPECTRAL_WINDOW")
        freq=t.getcol("REF_FREQUENCY")[0]
        OneOverchanwl=t.getcol("CHAN_FREQ")/299792458.
        t.close()
        # calculate phase shift to apply
        t=table(outputmsname,readonly=False,ack=False)
        uv, vv, wv = t.getcol("UVW").T
        deltal  = np.sin(targetradRA  - refradRA )
        deltam  = np.sin(targetradDEC - refradDEC)
        n1      = np.sqrt(1-deltal**2-deltam**2)-1
        expvall =(-2j*np.pi*(deltal*uv+deltam*vv+n1*wv)).reshape((len(uv),1))
        expvals =np.exp(np.dot(expvall,OneOverchanwl)).astype(np.clongdouble)
        # apply phase shift to visibilities
        listcols=t.colnames()
        for col in listcols:
            if "data" in col.lower():
                if col!="DATA_DESC_ID":
                    if v:print("Phase shifting %s"%col)
                    d=t.getcol(col)
                    for j in range(d.shape[2]):
                        d[:,:,j]=d[:,:,j]*expvals
                    t.putcol(col,d)
        t.close()
    if v:print("Finished phase shifting visibilities")
    if v: print("Rephased %s, output at %s"%(msname,outputmsname))

if __name__=="__main__":
    args=readArguments()
    mslist=args["msname"]
    appendstr=args["outputstr"]
    targetradRA=HHMMSS2rad(args["RA"])
    targetradDEC=DDMMSS2rad(args["DEC"])
    fieldid=args["FieldID"]
    spw=args["SPW"]
    rephase=args["rephase"]
    v=args["verbose"]

    # check wrap
    if targetradRA>np.pi:
        targetradRA-=2*np.pi
    if targetradRA<-np.pi:
        targetradRA+=2*np.pi

    print(targetradRA)
        
    for msname in mslist:
        rephaseMS(msname, targetradRA, targetradDEC,appendstr,fieldid,spw,v,rephase)
