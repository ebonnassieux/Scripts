import numpy as np
import os
import glob
#import bdsf
import subprocess



mslist=np.sort(glob.glob("/data/etienne.bonnassieux/NenuFAR/2021/L1/SB404.MS"))

dpname="DP3"

calmode=""
#calmode="DPPP"
calmode="kMS"
#imgmode="wsclean"
imgmode="DDF"
passmin=0
passmax=3
#skymodel="virgoA.skymodel"
skymodel="virgoA.skymodel"#"apparent.skymodel"#"coma.LSM.skymodel"
imname="image.sb404.nenubeam.nobeam"

# parallelisation
nMS=len(mslist)
#njobs=10
#jobbounds=range(0,nMS,njobs)
#if jobbounds[-1]!=(nMS): jobbounds.append(nMS)

msliststr=""
for msn in mslist:
    print(msn)
    msliststr=msliststr+msn+" "
if imgmode=="DDF":
    mslistname="mslist."+imname+".txt"
    a=open(mslistname,"w")
    for ms in mslist:
        a.write(ms+"\n")
    a.close()

pop=[]
    
for i in range(passmin,passmax):
    ### Calibrate per MS. Parallelise this later
    if i==0:
        # make model if first iter. Otherwise use predicted vis
        if calmode=="kMS":
            ##  Make DI sky model
            modstr="python /home/etienne.bonnassieux/DDFpy3.10/sources/DDFacet/MyDDF/bin/MakeModel.py"+\
                    "--SkyModel %s --NCluster 1"%skymodel
            print(modstr)
            os.system(modstr)
    for ms in mslist:
        njobs=1
        os.system("export  OMP_NUM_THREADS=%i"%(int(52/njobs)))
        if calmode=="DPPP":
            if i==0:
                ### use DPPP
                dpname="DP3"
                ##  make model
                os.system("rm -rf %s/sky"%ms)
                sourcedbstr="makesourcedb in=%s out=%s/sky format=\'<\'"%(skymodel,ms)
                print(sourcedbstr)
                os.system(sourcedbstr)
                calstr="%s steps=[gaincal] gaincal.type=gaincal"%dpname+\
                        " gaincal.sourcedb=%s/sky gaincal.parmdb=%s/instrument"%(ms,ms)+\
                        " msin=%s msout=%s gaincal.caltype=diagonal gaincal.solint=8"%(ms,ms)+\
                        " gaincal.nchan=4 msin.datacolumn=DATA"+\
                        " gaincal.applysolution=True"+\
                        " msout.datacolumn=CORRECTED_DATA"
                
            else:
                calstr="%s steps=[gaincal] gaincal.type=gaincal gaincal.usemodelcolum=True"%dpname+\
                        " gaincal.modelcol=%s gaincal.parmdb=%s/instrument"%("MODEL_DATA",ms)+\
                        " msin=%s msout=%s gaincal.caltype=diagonal gaincal.solint=8"%(ms,ms)+\
                        " gaincal.applysolution=True gaincal.nchan=0 msin.datacolumn=DATA"+\
                        " msout.datacolumn=CORRECTED_DATA"
                
            print(calstr)
            #os.system(calstr)
            pop.append(subprocess.Popen(calstr,shell=True))
            
            for p in pop:
                p.wait()
#            ##  apply calibration solutions
#            applystr="DP3 steps=[applycal]  .parmdb=%s/instrument"%ms+\
#                        " msin=%s msout=%s  msin.datacolumn=DATA"%(ms,ms)+\
#                        " msout.datacolumn=CORRECTED_DATA"
#            print(applystr)
#            #os.system(applystr)
#            pop.append(subprocess.Popen(applystr,shell=True))
#            for p in pop:
#                p.wait()

        elif calmode=="kMS":
            ##  Calibrate
            beammodel="None"
            if i>0:
                calstr= "python /data/etienne.bonnassieux/DDFpy3.10/sources/killMS/MykMS/bin/kMS.py"+\
                        " --MSName %s --InCol DATA"%ms+" --OutCol CORRECTED_DATA --Beam-Model %s"%beammodel+\
                        " --SkyModel %s --SolverType CohJones --dt .25 --Decorrelation FT "%skymodel +\
                        " --NChanSols 1 --NIterLM 50 --PolMode  IDiag --NCPU 32 --DebugPdb 0"+\
                        " --ApplyToDir 0 --ApplyMode AP"
            else:
                calstr= "python /data/etienne.bonnassieux/DDFpy3.10/sources/killMS/MykMS/bin/kMS.py"+\
                        " --MSName %s --InCol DATA"%ms+" --OutCol CORRECTED_DATA"+\
                        " --SkyModelCol %s --SolverType CohJones --dt .25 --Decorrelation FT "%skymodel+\
                        " --NChanSols 1 --NIterLM 50 --PolMode  IDiag --NCPU 32 --DebugPdb 0"+\
                        " --ApplyToDir 0 --ApplyMode AP"
            print(calstr)
            pop.append(subprocess.Popen(calstr,shell=True))
            for p in pop:
                p.wait()

        stop


    ### Image all data at once.
    # wsclean imaging
    imstr=""
    if imgmode=="wsclean":
        imstr="wsclean   -update-model-required -size 900 900 -scale 4arcmin "+\
               "-weighting-rank-filter 3  -data-column CORRECTED_DATA"+\
               " -local-rms -pol i -name IMAGES/%s.pass%i -weight briggs -0.5"+\
               " -mem 95 -niter 15000 -multiscale -mem 50 -j 26 "%(imname,i)+\
               "  %s"%msliststr
    # ddf imaging
    if imgmode=="DDF":
        beammodel="None"#"NENUFAR" # "None"
        imstr= "python /data/etienne.bonnassieux/DDFpy3.10/sources/DDFacet/MyDDF/bin/DDF.py"+\
               " --Data-MS %s --Parallel-NCPU 48 --Beam-Model %s "%(mslistname,beammodel)+\
               " --Output-Name IMAGES/%s.pass%i --Deconv-Mode HMP --Deconv-MaxMajorIter 20"%(imname,i)+\
               " --Cache-Reset 1 --Output-Also \"all\"  --Data-ColName CORRECTED_DATA "+\
               " --Image-NPix 1200 --Image-Cell 240 --Debug-Pdb 0  --Cache-Reset 1 --Beam-CenterNorm 1"+\
               " --Predict-ColName MODEL_DATA  --Facets-NFacets 13 --Output-Mode Clean --RIME-DecorrMode FT "+\
               " --Mask-Auto 1 --Mask-SigTh 5 --Freq-NDegridBand %i --Weight-Mode Briggs --Weight-Robust 0 "%(len(mslist))
    if imstr!="":
        os.system(imstr)
        print(imstr)
        print("deleting DDF cache")
        os.system("rm -rf 2021/L1/*ddfcache")
        print("rm -rf 2021/L1/*ddfcache")
        os.system("rm -rf *ddfcache")
        print("rm -rf *ddfcache")

# " --Image-PhaseCenterRADEC=[\"13:00:00.0\",\"50:00:00.0\"] "%(imname,i)+\
        
    ### Make new model
#    if imgmode=="wsclean":
#        imgrestored="IMAGES/"+imname+".pass%i"%i+"-image.fits"
#    elif imgmode=="DDF":
#        imgrestored="IMAGES/%s.pass%i"%(imname,i)+".app.restored.fits"
#    if i>0:
#        bdsfimg=bdsf.process_image(imgrestored,thresh_isl=5,thresh_pix=10)
#        skymodel="IMAGES/"+imname+".pass%i"%i+".skymodel"
#        bdsfimg.write_catalog(outfile=skymodel,bbs_patches="single",catalog_type="gaul",clobber=True)



    
