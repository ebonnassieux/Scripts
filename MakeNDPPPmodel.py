import numpy as np
import pylab
import argparse
import warnings
# ignore matplotlib warnings
warnings.simplefilter("ignore")

### This program aims to take in a DDF MakeModel
### .npy model file, and bootstrap it against a
### given flux scale (in this case, 3C295). This
### bootstrapping is approximative, in anticipation
### of an analytical method for bootstrapping against
### (up to) fourth-order in log-space. 


### plot bit ###                                                                                                                                                                                           
cmfont = {'fontname':'Computer Modern'}
pylab.rcParams["font.family"] = "serif"
pylab.rcParams["font.serif"]  = "Computer Modern"
pylab.rcParams['text.usetex'] = True
pylab.rcParams["font.size"]   = 12
pylab.rcParams["axes.formatter.limits"] = -4,4

def readArguments():
    parser=argparse.ArgumentParser("Make bootstrapped NDPPP model from a DDF .npy sky model")
    parser.add_argument("-v","--verbose",help="Be verbose, say everything program does. Default is False",\
                        required=False,action="store_true")
    parser.add_argument("--filename",type=str,help="Name of the DDF .npy model from which"+\
                        " to create a bootstrapped NDPPP model",required=True)
    parser.add_argument("--source",type=str,help="Source from Scaife & Heald you're "+\
                        " bootstrapping. Default is 3C295. FOR NOW, ONLY WORKS FOR 3C295",\
                        required=False,default="3C295")
    parser.add_argument("--outname",type=str,help="Name of the output skymodel. Default is"+\
                        " 3C295.bootstrapped.skymodel",default="3C295.bootstrapped.skymodel",required=False)
    args=parser.parse_args()
    return vars(args)

def HHMMSS(angval):
    ### function to return HH:MM:SS.SS string
    ### from angular value in radians
    # convert angval to hh
    angval=angval/np.pi*12
    hh=int(angval)
    mm=int(60*(angval-hh))
    ss=3600*(angval-hh-mm/60.)
    hhmmss="%2i:%2i:%05.2f"%(hh,mm,ss)
    return hhmmss

def DDMMSS(angval):
    ### function to return DD:MM:SS.SS string
    ### from angular value in radians 
    angval=angval/np.pi*180
    hh=int(angval)
    mm=int(60*(angval-hh))
    ss=3600*(angval-hh-mm/60.)
    ddmmss="%+2i.%2i.%05.2f"%(hh,mm,ss)
    return ddmmss

def FluxScale3C295(freq):
    ### Encodes the flux scale of 3C295 as
    ### given by Scaife & Heald 2012
    ### (cf https://arxiv.org/abs/1203.0977 )
    ## expect freq in MHz, can be a scalar or array
    ## 3degree model is best but up to 4th order given
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
    ## reminder: log in python is actually ln
    lognumber=np.log10(freq/150.)
    S=S*10**(a1*lognumber)
    S=S*10**(a2*lognumber**2)
    S=S*10**(a3*lognumber**3)
    #S=S*10**(a4*lognumber**4)
    return S

def FluxFromComp(freqs,nu0,a0,a1,a2=0,a3=0,a4=0):
    ### Function that calculates the flux of a component
    ### at a set of frequencies freqs using its flux at
    ### reference frequency nu0 (i.e. a0) and its spectral
    ### indices (a1 1st-order, a2 2nd-order, etc)
    lognumber=np.log10(freqs/nu0)
    S=a0
    S=S*10**(a1*lognumber)
    S=S*10**(a2*lognumber**2)
    S=S*10**(a3*lognumber**3)
    S=S*10**(a4*lognumber**4)
    return S
    
def WriteNDPPPmodelFromNpyModel(modelfilename,outfilename,componentstring="3C295_comp",source="3C295",v=False):
    ### This function takes a npy model and writes
    ### it as a "bbs"-format model. It also bootstraps.
    # values for nu^2 and nu^3 coefficients respectively
    a1=-.582
    a2=-.298
    a3=.583
    # calculate bootstrap value
    model=np.load(modelfilename)
    flux=[]
    alpha=[]
    for i in range(len(model)):
        flux.append(model[i][4])
        alpha.append(model[i][9])
    flux=np.array(flux)
    alpha=np.array(alpha)
    nu0=model[0][8]*1e-6
    if source=="3C295":
        refflux=FluxScale3C295(nu0)
    scalingfactor=1.*refflux/np.sum(flux)
    # begin file interaction
    f=open(outfilename,"w+")
    # write header. Might need to add LogarithmicSI column to True for all,
    # as per https://sourceforge.net/p/wsclean/wiki/ComponentList/
    f.write("format = Name, Type, Ra, Dec, I, Q, U, V, MajorAxis, MinorAxis,"+\
            " Orientation, ReferenceFrequency='%f', SpectralIndex='[-.8,0,0]'\n"%(model[0][8]))
    # now fill the NDPPP model with the bootstrapped components
    il=len(str(len(model)))
    for i in range(len(model)):
        # do bootstrapping here
        ra=HHMMSS(model[i][1])
        dec=DDMMSS(model[i][2])
        f.write("\n%s%s, POINT, %s, %s, %+f, %+f, %+f, %+f, 0, 0, 0, %f, "%\
                (componentstring,str(i).zfill(il),ra,dec, model[i][4]*scalingfactor,model[i][5]*scalingfactor,\
                 model[i][6]*scalingfactor,model[i][7]*scalingfactor,model[i][8]))
        # now write the spectral indices
        f.write("[%f,%f,%f]"%(model[i][9],a2,a3))
    # exit gracefully
    f.close()
    if v: print "Saved bootstrapped BBS-type skymodel at: %s"%outfilename
        
def DiagPlotNDPPPmodel(filename,source="3C295",freqmin=110.,freqmax=240.,diagplotname="fluxscale",v=False):
    ### This function takes a BBS-style model and
    ### plots the associated diagnostic plot - it
    ### shows the integrated flux of the model as
    ### function of freq, plotted vs. the Scaife &
    ### Heald reference, with the reference frequency
    ### shown. Below are relative residuals, in %. This
    ### gives a reliability metric of the flux scale
    ### achieved using this model as a function of freq.
    ### Frequency in input is in units of MHz.
    # open file
    # read contents, header
    contents=open(filename,"r").read().replace("\n",",").split(",")
    nvals=contents.index("") # this is the empty line between header and model
    ncomps=len(contents)/nvals
    del contents[nvals]
    vals=np.array(contents).reshape((nvals,ncomps))
    freqs=1.*(np.arange(freqmax-freqmin)+freqmin)
    nu0=np.float64(vals[0][11].split("=")[1][1:-1])*1e-6
    nu0arr=np.zeros_like(freqs)
    modelflux=np.zeros_like(freqs)
    for i in range(1,ncomps):
        # read a0 & spectral indices from file
        a0=float(contents[i*nvals+4])
        a1=float(contents[i*nvals+12][2::])
        a2=float(contents[i*nvals+13])
        a3=0#float(contents[i*nvals+14][0:-1])
        modelflux+=FluxFromComp(freqs,nu0,a0,a1,a2,a3)
    # make plot
    mask=(freqs>(nu0-.5))*(freqs<(nu0+.5))
    if source=="3C295":
        nu0arr[mask]=FluxScale3C295(nu0)
        refflux=FluxScale3C295(freqs)
    fig1=pylab.figure(figsize=(8,8))
    frame1=fig1.add_axes((.13,.3,.85,.6))
    pylab.loglog(freqs,refflux,label=r"Scaife \& Heald")
    pylab.loglog(freqs,modelflux,label=r"Model")
    pylab.loglog(freqs,nu0arr,label=r"$\nu_0$")
    pylab.title(r"Integrated flux of %s model vs S\&H scale"%source)
    pylab.ylabel("Integrated flux [Jy]",**cmfont)
    pylab.xlabel(r"$\nu$ [MHz]",**cmfont)
    pylab.legend()
    frame1.set_xticklabels([])
    frame2=fig1.add_axes((.13,.13,.85,.2))
    pylab.loglog(freqs,np.abs(modelflux-refflux)/refflux*100)
    pylab.ylabel(r"Relative residual [\%]",**cmfont)
    pylab.xlabel(r"$\nu$ [MHz]",**cmfont)
    frame2.get_xaxis().get_major_formatter().labelOnlyBase = False
    pylab.tight_layout()
    pylab.savefig(source+diagplotname)
    if v: print "Saved diagnostic plot at: %s"%(source+diagplotname+".png")

if __name__=="__main__":
    args=readArguments()
    filename=args["filename"]
    outfilename=args["outname"]
    v=args["verbose"]
    WriteNDPPPmodelFromNpyModel(filename,outfilename,componentstring="3C295_comp",source="3C295",v=v)
    DiagPlotNDPPPmodel(outfilename,source="3C295",v=v)
