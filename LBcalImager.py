import glob
import os
import sys
import subprocess
import numpy as np

class LBcalImager:
   def __init__(self,ms,paramset,nImagesAtOnce=1,path2lbcal="/home/etienne.bonnassieux/Software/LBcals/LBcals",path2images="./LBcalImagerResults"):
      self.PathToLBcalData=path2lbcal
      self.PathToImageDir=path2images
      self.MSName=ms
      self.parset=paramset
      self.njobs=nImagesAtOnce

   def ReadLBcalDat(self):
      # initialise the arrays where we'll put contents of .dat files
      NorthStarDat=[]
      TGSSdat=[]
      VLSSdat=[]
      WENSSdat=[]
      ## read northstar file
      for line in open("%s/LBcals__Northstar.cat"%self.PathToLBcalData,"r"):
         NorthStarDat.append(line.rstrip().split())
      # header: object name, RA, Dec
      NorthStar=np.array(NorthStarDat)[:,0:3]
      for i in range(NorthStar.shape[0]):
         NorthStar[i,1]=HHMMSStoHourAng(NorthStar[i,1])*15
         NorthStar[i,2]=HHMMSStoHourAng(NorthStar[i,2])
      ## read TGSS file
      for line in open("%s/LBcals__TGSSmatches.dat"%self.PathToLBcalData,"r"):
         TGSSdat.append(line.rstrip().split())
      # header: RA, Dec, S peak, S integrated, distance
      TGSS=np.array(TGSSdat[1:-1]).astype(float)
      # convert from deg to angular hour
      TGSS[:,0]=TGSS[:,0]/15
      ## read VLSS file
      for line in open("%s/LBcals__VLSSmatches.dat"%self.PathToLBcalData,"r"):
         VLSSdat.append(line.rstrip().split())
      # drop header. Contains: RA, Dec, S integrated, distance
      VLSStemp=np.array(VLSSdat[1:-1]).astype(float)
      VLSS=np.zeros((VLSStemp.shape[0],4))
      VLSS[:,0]=VLSStemp[:,0]+VLSStemp[:,1]/60.+VLSStemp[:,2]/3600.
      VLSS[:,1]=VLSStemp[:,3]+VLSStemp[:,4]/60.+VLSStemp[:,5]/3600.
      VLSS[:,2]=VLSStemp[:,6]
      VLSS[:,3]=VLSStemp[:,7]
      ## read WENSS file
      for line in open("%s/LBcals__WENSSmatches.dat"%self.PathToLBcalData,"r"):
         WENSSdat.append(line.rstrip().split())
      # header: RA, Dec, S peak, S integrated, distance
      WENSStemp=np.array(WENSSdat[1:-1]).astype(float)
      WENSS=np.zeros((WENSStemp.shape[0],5))
      WENSS[:,0]=WENSStemp[:,0]+WENSStemp[:,1]/60.+WENSStemp[:,2]/3600.
      WENSS[:,1]=WENSStemp[:,3]+WENSStemp[:,4]/60.+WENSStemp[:,5]/3600.
      WENSS[:,2]=WENSStemp[:,6]
      WENSS[:,3]=WENSStemp[:,7]
      WENSS[:,4]=WENSStemp[:,8]
      return NorthStar,TGSS,VLSS,WENSS

   def FindNbrightSourcesInRange(self,nsources,distminDeg,distmaxDeg):
      # first, load data
      NorthStar,TGSS,VLSS,WENSS=self.ReadLBcalDat()
      # perform distance selection
      TGSSselec=(TGSS[:,-1]>distminDeg)*(TGSS[:,-1]<distmaxDeg) # booleans; * operation is "and"
      VLSSselec=(VLSS[:,-1]>distminDeg)*(VLSS[:,-1]<distmaxDeg)
      WENSSselec=(WENSS[:,-1]>distminDeg)*(WENSS[:,-1]<distmaxDeg)
      # find nsources brightest object within selection, per survey
      tempTGSS=TGSS[TGSSselec]
      tempVLSS=VLSS[VLSSselec]
      tempWENSS=WENSS[WENSSselec]
      TGSSbrightest=tempTGSS[np.argsort(tempTGSS)[-2][-nsources:],:]#tempTGSS[-2].argsort()[-nsources:]
      VLSSbrightest=tempVLSS[np.argsort(tempVLSS)[-2][-nsources:],:]#tempVLSS[-2].argsort()[-nsources:]
      WENSSbrightest=tempWENSS[np.argsort(tempWENSS)[-2][-nsources:],:]#tempWENSS[-2].argsort()[-nsources:]
      # find corresponding RA, dec

      ThreeCeecoords=np.array([HHMMSStoHourAng("14:11:49.87")*15,HHMMSStoHourAng("52:48:57.60")])
#      for i in range(nsources):
#          print np.sqrt( (ThreeCeecoords[0]-TGSSbrightest[i,0])**2 + (ThreeCeecoords[1]-TGSSbrightest[i,1])**2 ),TGSSbrightest[i,-1]
#          print np.sqrt( (ThreeCeecoords[0]-VLSSbrightest[i,0])**2 + (ThreeCeecoords[1]-VLSSbrightest[i,1])**2 ),VLSSbrightest[i,-1]
#          print np.sqrt( (ThreeCeecoords[0]-WENSSbrightest[i,0])**2 +( ThreeCeecoords[1]-WENSSbrightest[i,1])**2 ),WENSSbrightest[i,-1]

#      print TGSSbrightest
#      print VLSSbrightest
#      print WENSSbrightest

      Coords=TGSSbrightest[:,0:2]#.ravel()
      Coords=np.append(Coords,VLSSbrightest[:,0:2]).reshape((Coords.shape[0]+VLSSbrightest[:,0:2].shape[0],2))
      Coords=np.append(Coords,WENSSbrightest[:,0:2]).reshape((Coords.shape[0]+WENSSbrightest[:,0:2].shape[0],2))
      # create string array and convert to HHMMSS
      outcoords=np.empty(Coords.shape,dtype="a11")
      for i in range(Coords.shape[0]):
         outcoords[i,0]=HourAngtoHHMMSS(Coords[i,0])*15
         outcoords[i,1]=HourAngtoHHMMSS(Coords[i,1])
      # remove duplicates
      u,ind=np.unique(outcoords,return_index=True)
      sortcoords=u[np.argsort(ind)].reshape(outcoords.shape)
      return sortcoords

   def go(self,sourceRA,sourceDec,MaxAngDist,nPerCatalogPerCircle=4,nCircles=5):
      # convert input RA, Dec to angular values
      RAdeg=HHMMSStoHourAng(sourceRA)
      Decdeg=HHMMSStoHourAng(sourceDec)
      # launch LBcal catalog to find sources up to MaxAngDist
      launchdir=os.getcwd()
      os.chdir(self.PathToLBcalData)
      LBcalCommand="Rscript LBcals.r --RA_deg %f -D %f -r %f"%(RAdeg,Decdeg,MaxAngDist)
      print "Executing command: %s"%LBcalCommand
      #os.system(LBcalCommand)
      # return to cwd
      os.chdir(launchdir)
      # make array of min, max distances from central object
      bounds=np.arange(nCircles+1)*np.float(MaxAngDist)/nCircles
      # go to image directory, do images
      for i in range(nCircles):
         # go to subdirectory in which to keep images
         currentImageDir=self.PathToImageDir+".%3.1f-%3.1f.DegAway"%(bounds[i],bounds[i+1])#,sourceRA,sourceDec)
         os.system("mkdir %s"%currentImageDir)
         coords=self.FindNbrightSourcesInRange(nPerCatalogPerCircle,bounds[i],bounds[i+1])
         print CalcAngDist(reference=["14:11:49.87","52:48:57.60"],Coords=coords)
         # make images
         jobbounds=range(0,coords.shape[0],self.njobs)
         if jobbounds[-1]!=(coords.shape[0]): jobbounds.append(coords.shape[0])
         for k in range(len(jobbounds)-1):
            pop=[]
            jbegin=jobbounds[k]
            jend=jobbounds[k+1]
            print "we're in batch %i - %i of %i"%(jbegin+1,jend,coords.shape[0])
            for j in range(jbegin,jend):
               ImageCommand="DDF.py %s --MSName %s --ImageName %s/LBcal.PSF.%s-%s --Mode=PSF --PhaseCenterRADEC=[%s,%s] > %s/LBcal.%s-%s.log 2>&1"\
               %(self.parset,self.MSName,currentImageDir,coords[j,0],coords[j,1],coords[j,0],coords[j,1],currentImageDir,coords[j,0],coords[j,1])
               print "Executing command: %s"%ImageCommand
               pop.append(subprocess.Popen(ImageCommand,shell=True))
            for p in pop:
               p.wait()
         # return to launch directory, images done. Placed here in case of relative path given for PathToImageDir
         os.chdir(launchdir)

def CalcAngDist(ref,Coords):
    reference=np.array([HHMMSStoHourAng(ref[0])*np.pi/12.,HHMMSStoHourAng(ref[1])*np.pi/180.])
    dists=np.array([])
    halfpi=np.pi/2.
    for i in range(Coords.shape[0]):
        coords=np.array([HHMMSStoHourAng(Coords[i,0])*np.pi/12.,HHMMSStoHourAng(Coords[i,1])*np.pi/180.])
        distance=np.arccos( np.cos(halfpi-reference[1])*np.cos(halfpi-coords[1]) + np.cos(reference[0]-coords[0])*np.sin(halfpi-reference[1])*np.sin(halfpi-coords[1]) )
        dists=np.append(dists,distance)
    # convert back to degrees
    dists=dists*180/np.pi
    return dists


def HHMMSStoHourAng(HHMMSS):
   # convert HHMMSS string to angular value float
   HH,MM,SS=np.array(HHMMSS.split(":")).astype(float)
   hourval=HH+MM/60.+SS/3600
   return hourval

def HourAngtoHHMMSS(HourVal):
   HH=int(HourVal)
   MM=int((HourVal-HH)*60)
   SS=((HourVal-HH-MM/60.)*3600)
   hhmmss="%02i:%02i:%05.2f"%(HH,MM,SS)
   return hhmmss

if __name__=="__main__":
    # example of calling the object as a function: python LBcalImager.py 14:26:30.53 52:19:00.60 5 msname ddfparset 
    RA=sys.argv[1]
    Dec=sys.argv[2]
    MaxAngDist=float(sys.argv[3])
#    if sys.argv[4][0]=="/":
#       # if absolute path is given
    MSName=sys.argv[4]
#    else:
#       # if relative path is given
#       MSName=os.getcwd()+"/"+sys.argv[4]
#    if sys.argv[5][0]=="/":
    DDFparset=sys.argv[5]
#    else:
#       DDFparset=os.getcwd()+"/"+sys.argv[5]
    test=LBcalImager(MSName,DDFparset)
    test.go(sourceRA=RA,sourceDec=Dec,nPerCatalogPerCircle=5,MaxAngDist=MaxAngDist,nCircles=3)
