import numpy as np
import glob
import os
import sys
import subprocess
import bdsf

class process():
    def __init__(self,msfiles,run_name, initskymodel,njobs=4,nitersc = 1, start=0):
        self.mslist     = glob.glob(msfiles)
        self.run_name   = run_name
        self.initskymod = initskymodel
        self.njobs      = njobs
        self.pop        = []
        self.logfiles   = []
        self.niter      = nitersc
        self.thisiter   = start
        self.nMS        = len(self.mslist)
        self.jobbounds  = np.append(np.arange(0,self.nMS,self.njobs),self.nMS)
        # set up DP3 parallelisation
        os.system("export  OMP_NUM_THREADS=%i"%(int(64/self.njobs)))

    def InitialiseSC_Cal_Params(self):
        self.InitCalDicoParams={}
        self.InitCalDicoParams["gaincal.type"]          = "gaincal"
        self.InitCalDicoParams["gaincal.caltype"]       = 'diagonal'
        self.InitCalDicoParams["gaincal.solint"]        = "8"
        self.InitCalDicoParams["gaincal.nchan"]         = "8"
        self.InitCalDicoParams[" msin.datacolumn"]      = "DATA"
        self.InitCalDicoParams["msout.datacolumn"]      = "CORRECTED_DATA"
        self.InitCalDicoParams["gaincal.applysolution"] = "True"

    def InitialiseImParams(self):
        self.InitImDicoParams={}
        self.InitImDicoParams["-minuv-l"]             = '0.0'
#        self.InitImDicoParams["-maxuv-l"]             = '210000'
#        self.InitImDicoParams["-circular-beam"]       = ''
        self.InitImDicoParams["-size"]                = '1000 1000'
        self.InitImDicoParams["-reorder"]             = ''
        self.InitImDicoParams["-weight"]              = 'briggs 0'
        self.InitImDicoParams["-clean-border"]        = '1'
        self.InitImDicoParams["-parallel-reordering"] =	'4'
        self.InitImDicoParams["-mgain"]      	      =	'0.4'
        self.InitImDicoParams["-data-column"]         =	'CORRECTED_DATA'
        self.InitImDicoParams["-padding"]     	      =	'1.4'
        self.InitImDicoParams["-join-channels"]       =	''
        self.InitImDicoParams["-channels-out"]        =	'2'
        self.InitImDicoParams["-auto-mask"]           = '2.5'
        self.InitImDicoParams["-auto-threshold"]      = '0.5'
        self.InitImDicoParams["-pol"]                 = 'i'
        self.InitImDicoParams["-gridder"]     	      =	'wgridder'
        self.InitImDicoParams["-name"]        	      =	"IMAGES/"+self.run_name+'.pass%i'%self.thisiter
        self.InitImDicoParams["-scale"]       	      =	'0.1arcsec'
        self.InitImDicoParams["-nmiter"]      	      =	'50'
        self.InitImDicoParams["-niter"]               = '25000'
#        self.InitImDicoParams["-circular-beam"]       = ''
        self.InitImDicoParams["-beam-shape"]          = '1 1 0'
        self.InitImDicoParams["-multiscale"]          = ''
        self.InitImDicoParams["-fits-mask"]           = 'OJ285.lofar.mask.fits'
        
    def Calibration(self):
        if self.thisiter==0:
            print('Performing very first calibration')
            skymodel=initskymodel
        else:
            print("Using sky model %s"%self.skymodelname)
            skymodel=self.skymodelname
        print()
        for i in range(len(self.jobbounds)-1):
            jbegin=self.jobbounds[i]
            jend=self.jobbounds[i+1]
            for j in range(jbegin,jend):
                print("We are in loop %i - %i of %i"%(jbegin+1, jend, self.nMS))
                # make model
                os.system("rm -rf %s/sky"%self.mslist[j])
                sourcedbstr="makesourcedb in=%s out=%s/sky format=\'<\'"%(skymodel,self.mslist[j])
                print(sourcedbstr)
                os.system(sourcedbstr)
                self.logfiles.append(open("%s.%s.initcal.log"%(self.mslist[j],self.run_name), "w"))
                commandstr = 'DP3'+\
                    ' msin=%s'%self.mslist[j]+\
                    ' msout=%s'%self.mslist[j]+\
                    ' steps=[gaincal]'+\
                    ' gaincal.sourcedb=%s/sky'%self.mslist[j]+\
                    ' gaincal.parmdb=%s/instrument'%self.mslist[j]
                for key in  self.InitCalDicoParams.keys():
                    commandstr += " %s=%s"%(key, self.InitCalDicoParams[key])
                commandstr+= ' > %s.%s.initcal.log 2&>1 '%(self.mslist[j],self.run_name)
                print(commandstr)
                print()

                popenarr = ['DP3',
                            'msin=%s'%self.mslist[j],
                            'msout=%s'%self.mslist[j],
                            'steps=[gaincal]',
                            'gaincal.sourcedb=%s/sky'%self.mslist[j],
                            'gaincal.parmdb=%s/instrument'%self.mslist[j]]
                for key in  self.InitCalDicoParams.keys():
                    popenarr.append('%s=%s'%(key, self.InitCalDicoParams[key]))
                self.pop.append(subprocess.Popen(popenarr,
                                                 stdout=self.logfiles[-1],
                                                 stderr=subprocess.STDOUT
                                                 )
                                )
            for p in self.pop:
                p.wait()
            for f in self.logfiles:
                f.close()
                
    def InitialImaging(self):
        commandstr = 'wsclean'
        for key in  self.InitImDicoParams.keys():
            commandstr += " %s %s"%(key, self.InitImDicoParams[key])
        for ms in self.mslist:
            commandstr += " %s"%ms
        print(commandstr)
        print()
        popenarr = ['wsclean']
        for key in  self.InitImDicoParams.keys():
            popenarr.append('%s %s'%(key, self.InitImDicoParams[key]))
        for ms in self.mslist:
            popenarr.append(ms)
        print(popenarr)
        self.pop.append(subprocess.Popen(commandstr,shell=True))
        for p in self.pop:
            p.wait()
        for f in self.logfiles:
            f.close()

    def MakeNewSkyModel(self):
        lastimname="IMAGES/"+self.run_name+'.pass%i'%(self.thisiter-1)+"-MFS-image.fits"
        img=bdsf.process_image(lastimname,
                           thresh_isl=8,\
                           thresh_pix=10)
        self.skymodelname='%s.pass%i.skymodel'%(self.run_name,self.thisiter)
        img.write_catalog(outfile=self.skymodelname,
                          catalog_type='gaul',\
                          format='bbs',\
                          clobber=True)
        print("Created new skymodel %s"%self.skymodelname)


            
		

    def SelfCalibrate(self):
        while self.thisiter<=self.niter:
            if self.thisiter == 0:
                print("Starting self-calibration from scratch. ")
                self.InitialiseSC_Cal_Params()
                self.InitialiseImParams()
                self.Calibration()
                self.InitialImaging()
            else:
                print("Starting self-calibration pass %i"%self.thisiter)
                self.MakeNewSkyModel()
                self.InitialiseSC_Cal_Params()
                self.Calibration()
                self.InitialiseImParams()
                self.InitialImaging()
            self.thisiter+=1

			
							

if __name__=='__main__':
    msfiles      = 'DATA/oj287_spw?_field4.ms/'
    run_name     = 'OJ287.pipeline.selfcal.lofarmask.dt8.dnu8.nouvcut.briggs0'
    initskymodel = 'OJ287.pass2.skymodel'
    
    pipeline = process(msfiles,run_name,initskymodel,njobs=4,nitersc = 9, start=0)
    pipeline.SelfCalibrate()
