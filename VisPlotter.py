import numpy as np
from pyrap.tables import table
import pylab

def VisPlotter(MSname):
    ms=table(MSname)
    dcorr=ms.getcol("CORRECTED_DATA")
#    dcorr=ms.getcol("CORRECTED_DATA")
    flags=ms.getcol("FLAG")
    ant1=ms.getcol("ANTENNA1")
    ant2=ms.getcol("ANTENNA2")
    antlist=list(set(ant1))
    # apply flags
    dcorr[flags==1]=0
    # plot 1 antenna data
    problems=np.where(dcorr>0.99*np.max(dcorr))[1]
    pylab.plot(dcorr[ant1[ant2==ant2[problems][0]]==ant1[problems][0]][:,0,0])

    pylab.show()
    k=0
#    for i in antlist:
#        k=k+1
#        for j in antlist[k:]:
#            print i,j,dcorr[ant1[ant2==j]==i].shape
#            pylab.subplot(np.max(antlist)+1,np.max(antlist)+1,i+j)
#            print dcorr.shape
#            stop
#            pylab.plot(np.mean(np.mean(dcorr[ant1[ant2==j]==i],axis=1),axis=1)[::10])
            
#    stop
#    pylab.savefig("test.png")

    ms.close()
