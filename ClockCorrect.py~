import sys
from pyrap.tables import table
import numpy as np
import matplotlib.pyplot as pylab
import lofar.parmdb

def ClockCorrect(MeasurementSet):
    # first, open measurement set on which you wish to perform clock correction
    t=table(MeasurementSet,readonly=False)
    # apply clock-correction to data that's already been amplitude-calibrated
    data=t.getcol("DATA")
    print "Measurement Set open: %s"%MeasurementSet
    # open the parmdb file
    db=lofar.parmdb.parmdb(MeasurementSet+"/instrument/")
    # get antenna names
    ants=table(MeasurementSet+'/ANTENNA')
    antname=ants.getcol("NAME")
    A0=t.getcol("ANTENNA1")
    A1=t.getcol("ANTENNA2")
    # get reference frequency
    sw=table(MeasurementSet+'/SPECTRAL_WINDOW/')
    ref_freq=sw.getcol('REF_FREQUENCY')[0]

    # now, start iterating over antennas.
    for i in range(0,len(antname)):
        for j in range(i,len(antname)):
            ant1=i
            ant2=j
            ind=np.where((A0==ant1)&(A1==ant2))[0]
            C1=db.getValuesGrid("Clock:%s"%antname[ant1])["Clock:%s"%antname[ant1]]["values"]
            C2=db.getValuesGrid("Clock:%s"%antname[ant2])["Clock:%s"%antname[ant2]]["values"]
            corr=np.exp(-2j*np.pi*ref_freq*(np.average(C1)-np.average(C2)))
            data[ind,:,:]=data[ind,:,:]*corr
        print "finished antennas paired with A"+str(i)

    #more pythonic method:
    # 


    # stick clock-corrected data back in
    t.putcol("CORRECTED_DATA",data)
    print "Clock successfully applied to %s"%MeasurementSet
    t.close()

if __name__=="__main__":
    ClockCorrect(sys.argv[1])
