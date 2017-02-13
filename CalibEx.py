import glob
import os
import sys
import subprocess
import getpass
import numpy as np

class CalibEx:
   def __init__(self,MSName,killmsParsetName,ddfParsetName,dtValsSec=np.array([2,4,8,16,30,60,120]),nChanSolVals=np.array([1,2,4,8]),CleanMemVal=True,AddColsVal=True):
      self.killmsParset=killmsParsetName
      self.ddfParset=ddfParsetName
      self.MS=MSName
      self.CleanMem=CleanMemVal
      self.AddCols=AddColsVal
      self.dt=dtValsSec/60.    # dt parameter is in minutes for killMS
      self.nchansols=nChanSolVals


   def go(self):
      # if specified, give cleanmem what it needs to avoid bus errors
      if self.CleanMem:
         pwd=getpass.getpass("Enter sudo password for freemem operation: ")
      # if specified, create CasaCols in MS
      if self.AddCols:
         os.system("MSTools.py --ms=%s > %s.mstools.log 2>&1"%(self.MS,self.MS))
      # begin calibration exploration
      for dtVal in self.dt:
         for nChanVal in self.nchansols:
            print "calibrate with dt=%3.3f, nchansols=%i"%(dtVal,nChanVal)
            calcommand="killMS.py %s --MSName=%s --dt=%f --NChanSols=%i > %s.killMS.%3.3fdt.%inchansols.log 2>$1"%(self.killmsParset,self.MS,dtVal,nChanVal,self.MS,dtVal,nChanVal)
            # calibrate with specified params
            print "executing command: %s"%calcommand
            #os.system(calcommand
            # print endstate
            print "calibration end||state:"
            os.system("tail -n 3 %s.killms.%3.3fdt.%inchansolslog"%(self.MS,dtVal,nChanVal))
            # flag bad calibration solutions
            clipcommand= "/home/cyril.tasse/PipelineCEP/PipelineCEP/clipcal.py --InCol=CORRECTED_DATA --ms=%s > %s.clipcal.%3.3fdt.%inchansols.log 2>&1"%(self.MS,self.MS,dtVal,nChanVal)
            print "executing commend: %s"%clipcommand
            #os.system(clipcommand)
            # image result
            imcommand="DDF.py %s --ImageName=test.%3.3fdt.%inchan> %s.DDF.%3.3fdt.%inchansols.log 2>$1"%(self.ddfParset,dtVal,nChanVal,self.MS,dtVal,nChanVal)
            print "executing command: %s"%imcommand
            #os.system(imcommand)
            # clean up if needed. TODO instead of freemem actually use the necessary bash code for platform indep.
            if self.CleanMem:
               os.system("echo %s | sudo -S /cep/lofar/bin/freemem 1; CleanSHM.py"%pwd)

if __name__=="__main__":
    # require following input format: MSName,killMSparset, DDFparset
    MSName=sys.argv[1]
    killMSparset=sys.argv[2]
    DDFparset=sys.argv[3]
    test=CalibEx(MSName,killMSparset,DDFparset)
    test.go()
