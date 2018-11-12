import numpy as np
import pylab
import argparse
from astropy.io import fits

def readArguments():
    parser=argparse.ArgumentParser(description="Bootstrap an image based on 3C295 flux scale")
    parser.add_argument("-v","--verbose",help="Be verbose, say everything program does. Default is False",required=False,action="store_true")
    parser.add_argument("--model",type=str,help="Name of the .fits image containing your model of 3C295 at your bootstrap image's frequency",required=True)
    parser.add_argument("--image",type=str,help="Name of image you want to bootstrap",required=False,default="",nargs="+")
    parser.add_argument("--append",type=str,help="String to add to your bootstrapped image name",required=False,default="bootstrap")
    args=parser.parse_args()
    return vars(args)
    
def FluxScale3C295(freq):
    ### expect freq in MHz
    ### 3degree model is best
    a0=97.763
    delta0=2.787
    a1=-.582
    delta1=.045
    a2=-.298
    delta2=.085
    a3=.583
    delta3=.116
    a4=-0.363
    delta4=.137
    S=a0
    ### reminder: log in python is actually ln
    lognumber=np.log10(freq/150.)
    S=S*10**(a1*lognumber)
    S=S*10**(a2*lognumber**2)
    S=S*10**(a3*lognumber**3)
    S=S*10**(a4*lognumber**4)
    return S
#freqs=1.*(np.arange(2000)+10)
#flux=FluxScale3C295(freqs)
#pylab.loglog(freqs,flux)
#pylab.show()

if __name__=="__main__":
    args=readArguments()
    modelname=args["model"]
    if "model" not in modelname.split("."):
        print "Please ensure that your model image actually is a model image: includes \"model\" in its name"
#        exit()
    imagenames=args["image"]
    appendname=args["append"]
    # start bootstrapping
    hdul=fits.open(modelname)
    hdr=hdul[0].header
    reffreq=hdr["RESTFRQ"] # for ddf image
    print "Reference frequency for this model is: %f Hz"%reffreq
#    print hdr["CDELT4"]
    modeldata=hdul[0].data
    nfreqs=modeldata.shape[0]
    if nfreqs==1:
        print "Model is not an image cube; bootstrapping."
        intflux=np.sum(modeldata[0,0,:,:][modeldata[0,0,:,:]>0])
#        print intflux,np.sum(modeldata[0,0,:,:]),np.sum(np.abs(modeldata[0,0,:,:]))
        flux3c295=FluxScale3C295(reffreq/1e6) # convert from Hz to expected MHz
        print "Expected integrated flux, from Scaife & Healde: %3.3f"%flux3c295
        print "Model integrated flux: %3.3f"%intflux
        print "Ratio for %s: %f"%(modelname,flux3c295/intflux)        
        bootsdata=modeldata/intflux*flux3c295
        hdul[0].data=bootsdata
        hdul.writeto(modelname[:-4]+appendname+".fits",clobber=True)
        print "Saving bootstrapped model to %s"%modelname[:-4]+appendname+".fits"
        if imagenames!="":
            for imagename in imagenames:
                hdul=fits.open(imagename)
                hdul[0].data=hdul[0].data/intflux*flux3c295
                print "Saving bootstrapped image to %s"%imagename[:-4]+appendname+".fits"
                hdul.writeto(imagename[:-4]+appendname+".fits",clobber=True)
    else:
        print "Model is an image cube: assuming DDF output & bootstrapping accordingly."
        deltafreq=hdr["CDELT4"]
        if nfreqs%2:
            # in this case, odd number of frequencies: central channel is thus at ref freq
            freqs=(np.arange(nfreqs)-(nfreqs/2))*deltafreq+reffreq
        else:
            freqs=(np.arange(nfreqs)-(nfreqs/2-.5))*deltafreq+reffreq
        flux3c295=FluxScale3C295(freqs/1e6)
        intflux=np.sum(modeldata,axis=(2,3))[:,0] # 0 as we only keep I
        intflux=np.sum(modeldata,axis=(2,3))
        bootsdata=np.zeros_like(modeldata)
        for i in range(nfreqs):
            modeldata[i]=modeldata[i]/intflux[i]*flux3c295[i]
            print "Frequency: %3.3f MHz"%(freqs[i]/1e6)
            print "Expected integrated flux, from Scaife & Healde: %3.3f"%flux3c295[i]
            print "Model integrated flux: %3.3f"%intflux[i]                    
        hdul[0].data=modeldata
        hdul.writeto(modelname[:-4]+appendname+".fits",clobber=True)
        print "Saving bootstrapped model to %s"%modelname[:-4]+appendname+".fits"
        if imagenames!="":
            for imagenames in imagename:
                hdul=fits.open(imagename)             
                for i in range(nfreqs):
                    hdul[0].data[i]=hdul[0].data[i]/intflux[i]*flux3c295[i]
                print "Saving bootstrapped image to %s"%imagename[:-4]+appendname+".fits"
                hdul.writeto(imagename[:-4]+appendname+".fits",clobber=True)
                                                                     



    print "Thank you for using this script"

