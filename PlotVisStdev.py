from pyrap.tables import table
import numpy as np
import sys
import pylab

def PlotStdev(mslist,pngName="Variances.png",colname="DATA"):
#    print mslist
    varlist=[]
#    freqlist=[]
    for i in range(len(mslist)):
        #freqlist.append(table(mslist[i]+"/SPECTRAL_WINDOW").getcol("REF_FREQUENCY")[0])
        t=table(mslist[i])
        varlist.append(np.std(t.getcol(colname)[t.getcol("FLAG")==0]))
        t.close()
        np.save("varlist.npy",np.array(varlist))
    np.save("mslist.npy",np.array(mslist))

    pylab.plot(np.arange(len(mslist)),np.array(varlist))
    pylab.xlabel("Subband number among list")
    pylab.ylabel("stdev")# normalised on sb093")
    pylab.show()
    np.save("mslist.npy",np.array(mslist))
#    np.save("freqlist.npy",np.array(freqlist))
    np.save("varlist.npy",np.array(varlist))
#    stop



if __name__=="__main__":
    pngName=sys.argv[1]
    mslist=sys.argv[2:]
    print mslist
    PlotStdev(mslist,pngName="Variances.png")
