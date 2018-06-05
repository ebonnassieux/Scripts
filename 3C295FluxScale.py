import numpy
import pylab

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
freqs=1.*(np.arange(2000)+10)
flux=FluxScale3C295(freqs)
pylab.loglog(freqs,flux)
pylab.show()
