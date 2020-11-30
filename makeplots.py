from pyrap.tables import table
import numpy as np
import cmath
import sys
import pylab
import matplotlib.font_manager


### plot bit ###                                                                                                                                                                                           
cmfont = {'fontname':'Computer Modern'}
pylab.rcParams["font.family"] = "serif"
pylab.rcParams["font.serif"]  = "Computer Modern"
pylab.rcParams['text.usetex'] = True
pylab.rcParams["font.size"]   = 12
pylab.rcParams["axes.formatter.limits"] = -4,4

def MakePlotWithResiduals(values1,values2,residuals,xaxs="",title="",ylabel="",xlabel="",outfilename="",values1label="",values2label="",alpha=0.8,s=0.01):
    if xaxs=="":
        xaxs=np.arange(len(values1))
    fig1=pylab.figure(figsize=(8,4))
    frame1=fig1.add_axes((.13,.3,.85,.6))
    pylab.title(title,**cmfont)
    pylab.ylabel(ylabel,**cmfont)
    pylab.scatter(xaxs,values1,s=s,label=values1label,alpha=alpha,c="r")
    pylab.scatter(xaxs,values2,s=s,label=values2label,alpha=alpha,c="b")
    lgnd=pylab.legend()
    for i in range(len(lgnd.legendHandles)):
        lgnd.legendHandles[i]._sizes = [30]
    minlim=np.min(np.append(values1,values2))
    if minlim>0:
        minlim=0.6*minlim
    elif minlim==0.0:
        minlim=-0.1*np.max(np.append(values1,values2))
    else:
        minlim=1.1*minlim
    maxlim=np.max(np.append(values1,values2))
    if maxlim>0:
        maxlim=1.1*maxlim
    else:
        maxlim=0.9*maxlim
    pylab.ylim(minlim,maxlim)
    pylab.xlim(np.min(xaxs),np.max(xaxs))
    frame1.set_xticklabels([])
    frame2=fig1.add_axes((.13,.13,.85,.2))
    pylab.scatter(xaxs,residuals,s=s,alpha=alpha,c="k")
    minlim=np.min(residuals)
    if minlim>0:
        minlim=0.9*minlim
    else:
        minlim=1.1*minlim
    maxlim=np.max(residuals)
    if maxlim>0:
        maxlim=1.1*maxlim
    else:
        maxlim=0.9*maxlim
    pylab.ylim(minlim,maxlim)
    pylab.xlim(np.min(xaxs),np.max(xaxs))
    pylab.xlabel(xlabel,**cmfont)
    pylab.ylabel("Residuals",**cmfont)
    pylab.tight_layout()
    if outfilename!="":
        pylab.savefig(outfilename)
        print "Made plot, found at: %s.png"%(outfilename)
    else:
        pylab.show()
    pylab.clf()

print "Starting decorr as function of l,m plot"
decorrMeas=np.load("decorr.measured.npy")
decorrPred=np.load("decorr.predicted.npy")
lvals=decorrMeas[:,0]*(180./np.pi*60)
MakePlotWithResiduals(decorrMeas[:,2],decorrPred[:,2],-(decorrMeas[:,2]-decorrPred[:,2]),xaxs=lvals,title="",ylabel=r"$d_f$",xlabel=r"$l$ [']",outfilename="decorr.compare.png",\
                      values1label=r"Measured $d_f$",values2label=r"Predicted $d_f$",alpha=1.,s=1)


    
### make residuals bit ###

def PrintProgress(currentIter,maxIter,msg=""):
    sys.stdout.flush()
    if msg=="":
        msg="Progress:"
    sys.stdout.write("\r%s %5.1f %% "%(msg,100*(currentIter+1.)/maxIter))
    if currentIter==(maxIter-1):
        sys.stdout.write("\n")

msname="test.superterp.2source.ms"
ms=table(msname)
uvw=ms.getcol("UVW")
d=ms.getcol("DATA")
d1=ms.getcol("SIM_DATA")
flags=ms.getcol("FLAG")
d[flags]=0
d1[flags]=0
u,v,_=uvw.T
t1=table(msname+"/SPECTRAL_WINDOW")
OneOverchanwl=t1.getcol("CHAN_FREQ")/299792458.
i=0
corestatind=0
supstatind=0
formstatstr="CS"
superstatname="ST001"
antnames=table(msname+"/ANTENNA").getcol("NAME")
for antname in antnames:
    if antname[0:2]==formstatstr:
        corestatind=i
    if antname==superstatname:
        supstatind=i
    i=i+1
ants1=ms.getcol("ANTENNA1")
ants2=ms.getcol("ANTENNA2")
flags=ms.getcol("FLAG")
nosupstat=(ants2!=supstatind)*(ants1!=supstatind)
u=u[nosupstat]
v=v[nosupstat]



print "Starting decorr as function of l,m plot"
decorrMeas=np.load("decorr.measured.npy")
decorrPred=np.load("decorr.predicted.npy")
lvals=decorrMeas[:,0]
MakePlotWithResiduals(decorrMeas[:,2],decorrPred[:,2],decorrMeas[:,2]-decorrPred[:,2],xaxs=lvals,title="",ylabel=r"$d_f$",xlabel=r"$l$",outfilename="decorr.compare.png",\
                      values1label=r"Measured $d_f$",values2label=r"Predicted $d_f$")

# make uv plot
uvals=(np.reshape(u,(len(u),1))*OneOverchanwl).ravel()
vvals=(np.reshape(v,(len(v),1))*OneOverchanwl).ravel()
maxval=len(uvals)
pylab.figure(figsize=(8,8))
pylab.title(r"$uv$-coverage")
pylab.xlabel(r"$u$")
pylab.ylabel(r"$v$")
pylab.scatter(uvals[::10],vvals[::10],s=0.1,c="b")
pylab.scatter(-uvals[::10],-vvals[::10],s=0.1,c="b")
pylab.tight_layout()
pylab.savefig("uvwcoverage.png")
pylab.clf()


#pylab.figure(figsize=(12,4))
#pylab.title(r"Difference between simulated and beamformed data")
#pylab.xlabel(r"$Vis. value$")
#pylab.ylabel(r"$Vis. index$")
xvals=np.arange(d.shape[0])
#d=d.ravel()
#d1=d1.ravel()

print d.shape
print "starting vis plot"
xvals=xvals[::10]
d=np.abs(d[::10,:,0]).ravel()
d1=np.abs(d1[::10,:,0]).ravel()

print d.shape

xvals=np.arange(len(d))

print d.shape
print xvals.shape

pylab.figure(figsize=(8,4))
pylab.title("Superstation Visibility Amplitudes",**cmfont)
pylab.xlabel("Index",**cmfont)
pylab.ylabel("Visbility Amplitude",**cmfont)
pylab.scatter(xvals,np.abs(d), s=0.01,label="Beamformed Vis.",alpha=0.8,c="r")
pylab.scatter(xvals,np.abs(d1),s=0.01,label="Simulated Vis.",alpha=0.8,c="b")
lgnd=pylab.legend()
for i in range(len(lgnd.legendHandles)):
    lgnd.legendHandles[i]._sizes = [30]
pylab.ylim(0,1.1)
pylab.savefig("visdifferences.amp.png")
print "Saving file as: ","visdifferences.amp.png"
stop


#MakePlotWithResiduals(np.abs(d),np.abs(d1),np.abs(d1-d),title="",ylabel="Visiblity Amplitude",\
#                      xlabel="Index",outfilename="visdifferences.amp.png",values1label="Beamformed Vis.",values2label="Simulated Vis.")

MakePlotWithResiduals(np.angle(d),np.angle(d1),np.angle(d1-d),title="",ylabel="Visiblity Phase",\
                      xlabel="Index",outfilename="visdifferences.phase.png",values1label="Beamformed Vis.",values2label="Simulated Vis.")

xvals=np.arange(len(d))
pylab.scatter(xvals,d,s=0.1,c="b",alpha=0.999,label="Beamformed data Amplitude")
pylab.scatter(xvals,d1,s=0.1,c="r",alpha=0.999,label="Simulated data Amplitude")
#pylab.scatter(xvals,d-d1,s=0.1,c="k",alpha=0.6,label="Residuals")
pylab.tight_layout()
pylab.legend()
pylab.savefig("visdifference.png")
pylab.clf()



print "Starting decorr as function of l,m plot"
decorrMeas=np.load("decorr.measured.npy")
decorrPred=np.load("decorr.predicted.npy")
lvals=decorrMeas[:,0]
MakePlotWithResiduals(decorrMeas[:,2],decorrPred[:,2],decorrMeas[:,2]-decorrPred[:,2],xaxs=lvals,title="",ylabel=r"$d_f$",xlabel=r"$l$",outfilename="decorr.compare.png",\
                      values1label=r"Measured $d_f$",values2label=r"Predicted $d_f$")

