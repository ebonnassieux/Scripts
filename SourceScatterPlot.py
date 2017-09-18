import numpy as np
import pylab
from astropy.io import fits
from matplotlib.patches import Ellipse

def WendyCatalogSourceReadRADEC(filename="/data/etienne.bonnassieux/Bootes/Wendy_catalogues/Bootes_merged_Icorr_2014a_all_ap4_mags_hmap.anom.fits",npoints=100000):
    hdulist=fits.open(filename)
    datapoints=datapoints=len(hdulist[1].data)
    HowManyPointsToPlot=min(datapoints,npoints)
    indices=(np.random.rand(HowManyPointsToPlot)*datapoints).astype(np.int)
    RAlist=np.zeros(HowManyPointsToPlot)
    Declist=np.zeros(HowManyPointsToPlot)
    for i in range(HowManyPointsToPlot):
        RAlist[i]=hdulist[1].data[indices[i]][1]
        Declist[i]=hdulist[1].data[indices[i]][2]
    hdulist.close()
    return RAlist,Declist

def ReadSpitzerRADEC2010(filename="/data/etienne.bonnassieux/Bootes/Spitzer_catalog/Spitzer2010/table1.dat",npoints=100000):
    f=open(filename,"r")
    ralist=[]
    declist=[]
    for line in f:
        columns=line.split()
        ralist.append(columns[2])
        declist.append(columns[3])
    f.close()
    if npoints<len(declist):
        indices=(np.random.rand(npoints)*len(declist)).astype(np.int)
        return np.array(ralist)[indices],np.array(declist)[indices]
    else:
        return np.array(ralist),np.array(declist)

def ReadSpitzerhilambda(filename="/data/etienne.bonnassieux/Bootes/Spitzer_catalog/table2.dat",npoints=100000):
    f=open(filename,"r")
    ralist=[]
    declist=[]
    for line in f:
        columns=line.split()
        ralist.append(columns[0])
        declist.append(columns[1])
    f.close()
    if npoints<len(declist):
        indices=(np.random.rand(npoints)*len(declist)).astype(np.int)
        return np.array(ralist)[indices],np.array(declist)[indices]
    else:
        return np.array(ralist),np.array(declist)


def ReadSpitzerlolambda(filename="/data/etienne.bonnassieux/Bootes/Spitzer_catalog/table1.dat",npoints=100000):
    f=open(filename,"r")
    ralist=[]
    declist=[]
    for line in f:
        columns=line.split()
        ralist.append(columns[0])
        declist.append(columns[1])
    f.close()
    if npoints<len(declist):
        indices=(np.random.rand(npoints)*len(declist)).astype(np.int)
        return np.array(ralist)[indices],np.array(declist)[indices]
    else:
        return np.array(ralist),np.array(declist)


def ReadXBootesRADEC(filename="/data/etienne.bonnassieux/Bootes/Chandra_XBootes_catalog/table3.dat",npoints=100000):
    f=open(filename,"r")
    ralist=[]
    declist=[]
    for line in f:
        columns=line.split()
        ra=float(columns[2])+float(columns[3])/60.+float(columns[4])/3600.
        dec=float(columns[5])+float(columns[6])/60.+float(columns[7])/3600.
        ralist.append(ra*15)
        declist.append(dec)
    f.close()
    if npoints<len(declist):
        indices=(np.random.rand(npoints)*len(declist)).astype(np.int)
        return np.array(ralist)[indices],np.array(declist)[indices]
    else:
        return np.array(ralist),np.array(declist)

def ReadMidIrRADEC(filename="/data/etienne.bonnassieux/Bootes/16microns/table1.dat",npoints=10000):
    f=open(filename,"r")
    ralist=[]
    declist=[]
    for line in f:
        columns=line.split()
        temp1,temp2=columns[2].split("+")
        ra=float(temp1[1:3])+float(temp1[3:5])/60.+float(temp1[5:10])/3600.
        dec=float(temp2[0:2])+float(temp2[2:4])/60.+float(temp2[4:8])/3600.
        ralist.append(ra*15)
        declist.append(dec)
    f.close()
    if npoints<len(declist):
        indices=(np.random.rand(npoints)*len(declist)).astype(np.int)
        return np.array(ralist)[indices],np.array(declist)[indices]
    else:
        return np.array(ralist),np.array(declist)

def ReadOpticalRADEC(filename="/data/etienne.bonnassieux/Bootes/XBootes_Optical_Counterparts_catalog/table2.dat",npoints=100000):
    f=open(filename,"r")
    ralist=[]
    declist=[]
    for line in f:
        columns=line.split()
        ralist.append(columns[8])
        declist.append(columns[9])
    f.close()
    if npoints<len(declist):
        indices=(np.random.rand(npoints)*len(declist)).astype(np.int)
        return np.array(ralist)[indices],np.array(declist)[indices]
    else:
        return np.array(ralist),np.array(declist)

def ReadWiseRADEC(filenames=["/data/etienne.bonnassieux/Bootes/WISE_catalog/table5.dat","/data/etienne.bonnassieux/Bootes/WISE_catalog/table6.dat"],npoints=100000):
    f=open(filenames[0],"r")
    ralist=[]
    declist=[]
    for line in f:
        columns=line.split("|")
        radecstring=columns[1].split()
        ra=float(radecstring[0])+float(radecstring[1])/60.+float(radecstring[2])/3600.
        dec=float(radecstring[3])+float(radecstring[4])/60.+float(radecstring[5])/3600.
        ralist.append(ra*15)
        declist.append(dec)
    f.close()
    f=open(filenames[1],"r")
    for line in f:
        columns=line.split()
        if columns[1]=="14":
            starti=1
        else:
            starti=2
        ra=float(columns[starti])+float(columns[starti+1])/60.+float(columns[starti+2])/3600.
        dec=float(columns[starti+3])+float(columns[starti+4])/60.+float(columns[starti+5])/3600.
        ralist.append(ra*15)
        declist.append(dec)
    if npoints<len(declist):
        indices=(np.random.rand(npoints)*len(declist)).astype(np.int)
        return np.array(ralist)[indices],np.array(declist)[indices]
    else:
        return np.array(ralist),np.array(declist)

def ReadGMRTRADEC(filename="/data/etienne.bonnassieux/Bootes/GMRT_153MHz_catalog/table3.dat",npoints=100000):
    f=open(filename,"r")
    ralist=[]
    declist=[]
    for line in f:
        columns=line.split()
        ralist.append(columns[1])
        declist.append(columns[2])
    f.close()
    if npoints<len(declist):
        indices=(np.random.rand(npoints)*len(declist)).astype(np.int)
        return np.array(ralist)[indices],np.array(declist)[indices]
    else:
        return np.array(ralist),np.array(declist)

def ReadVLARADEC(filename="/data/etienne.bonnassieux/Bootes/VLA_325MHz_catalog/table1.dat",npoints=100000):
    f=open(filename,"r")
    ralist=[]
    declist=[]
    for line in f:
        columns=line.split()
        ralist.append(columns[1])
        declist.append(columns[3])
    f.close()
    if npoints<len(declist):
        indices=(np.random.rand(npoints)*len(declist)).astype(np.int)
        return np.array(ralist)[indices],np.array(declist)[indices]
    else:
        return np.array(ralist),np.array(declist)

def ReadLALARADEC(filename="/data/etienne.bonnassieux/Bootes/Chandra_LALA_catalog/table1.dat",npoints=100000):
    f=open(filename,"r")
    ralist=[]
    declist=[]
    for line in f:
        columns=line.split()
        ra=float(columns[3])+float(columns[4])/60.+float(columns[5])/3600.
        dec=float(columns[6])+float(columns[7])/60.+float(columns[8])/3600.
        ralist.append(ra*15)
        declist.append(dec)
    f.close()
    if npoints<len(declist):
        indices=(np.random.rand(npoints)*len(declist)).astype(np.int)
        return np.array(ralist)[indices],np.array(declist)[indices]
    else:
        return np.array(ralist),np.array(declist)

def ReadLOFARlbaRaDec(filename="/data/etienne.bonnassieux/Bootes/LOFAR_LBA_catalogues/table3.dat",npoints=100000):
    f=open(filename,"r")
    ralist=[]
    declist=[]
    for line in f:
        columns=line.split()
        ralist.append(columns[3])
        declist.append(columns[5])
    f.close()
    if npoints<len(declist):
        indices=(np.random.rand(npoints)*len(declist)).astype(np.int)
        return np.array(ralist)[indices],np.array(declist)[indices]
    else:
        return np.array(ralist),np.array(declist)

def ReadWesterborkRADEC(filename="/data/etienne.bonnassieux/Bootes/WSRT_1.4GHz_catalog/catalog.dat",npoints=1000000):
    f=open(filename,"r")
    ralist=[]
    declist=[]
    for line in f:
        columns=line.split()
        ralist.append(columns[8])
        declist.append(columns[9])
    f.close()
    if npoints<len(declist):
        indices=(np.random.rand(npoints)*len(declist)).astype(np.int)
        return np.array(ralist)[indices],np.array(declist)[indices]
    else:
        return np.array(ralist),np.array(declist)

def ReadBBSradec(filename="/data/etienne.bonnassieux/Bootes/LOFAR_data/cutout.catalog.bbs",npoints=100000):
    f=open(filename,"r")
    ralist=[]
    declist=[]
    next(f);next(f)
    for line in f:
        columns=line.split(",")
        rastring=columns[2]
        decstring=columns[3]
        temp=rastring.split(":")
        ra=float(temp[0])+float(temp[1])/60.+float(temp[2])/3600.
        temp=decstring.split(".")
        dec=float(temp[0])+float(temp[1])/60.+float(float(temp[2])+float(temp[3])/10**len(temp[3]))/3600. 
        ralist.append(ra*15)
        declist.append(dec)
    f.close()
    if npoints<len(declist):
        indices=(np.random.rand(npoints)*len(declist)).astype(np.int)
        return np.array(ralist)[indices],np.array(declist)[indices]
    else:
        return np.array(ralist),np.array(declist)



if __name__=="__main__":
#    WendyRA,WendyDec=WendyCatalogSourceReadRADEC()
    #print "Read Wendy catalogues"
    #SpitzerRA,SpitzerDec=ReadSpitzer2010RADEC()
    XBootesRA,XBootesDec=ReadXBootesRADEC()
    #print "Read XBootes catalogues"
    #WiseRA,WiseDec=ReadWiseRADEC()
    #print "Read WISE catalogues"
    #SpitzerloRA,SpitzerloDec=ReadSpitzerlolambda()
    #SpitzerhiRA,SpitzerhiDec=ReadSpitzerhilambda()
    #print "Read Spitzer catalogues"
    #gmrtRA,gmrtDec=ReadGMRTRADEC()
    #print "Read GMRT catalogues"
    #vlaRA,vlaDec=ReadVLARADEC()
    #print "Read VLA catalogues"
    #lalaRA,lalaDec=ReadLALARADEC()
    #print "Read Chandra LALA field catalogue"
    #lbaRA,lbaDec=ReadLOFARlbaRaDec()    #print "Read LOFAR LBA catalogue"
    #wsrtRA,wsrtDec=ReadWesterborkRADEC()
    #print "Read WSRT data"
    #lofarRA,lofarDec=ReadBBSradec()
    #print "Read Cyril LOFAR sources"
    #midirRA,midirDec    =ReadMidIrRADEC()
    #opticRA,opticDec    =ReadOpticalRADEC()
    #pylab.scatter(SpitzerRA,SpitzerDec,s=0.1,color="b",alpha=0.8,label="Spitzer")
    pylab.figure(figsize=(12,12))
    # make polygons
    lwfactor=3
    pylab.gca().add_patch(Ellipse([218.0,34.5],4.6,4.6,fill=None,linestyle="-",linewidth=2*lwfactor,label="LOFAR HBA"))
    pylab.gca().add_patch(pylab.Polygon([[216.131,35.8359],[219.763,35.841],[219.69,33.465],[218.968,33.4704],[218.963,32.8756],\
                                         [218.275,32.8716],[218.275,32.3053],[216.139,32.288]],fill=None,linestyle="--",linewidth=1*lwfactor,label="NOAO-Deep"))
    #pylab.gca().add_patch(pylab.Polygon([[217,33.51],[217,35.5],[219.45,35.5],[219.45,33.51]],fill=None,linestyle="--",linewidth=2*lwfactor,label="LOFAR HBA"))
    pylab.gca().add_patch(pylab.Polygon([[216.73,35.586],[219.294,35.586],[219.294,35.0132],[219.687,35.0132],[219.687,33.39],[219.294,33.39],[219.294,32.875],[216.73,32.875],\
                                         [216.73,33.39],[216.38,33.39],[216.38,35.0132],[216.73,35.0132]],fill=None,linestyle=":",linewidth=1*lwfactor,label="WSRT 1.4GHz"))
    pylab.gca().add_patch(pylab.Polygon([[216.192,35.6757],[219.597,35.6757],[219.548,33.8594],[217.831,32.4453],[216.258,32.4531]],fill=None,\
                                        linewidth=1*lwfactor,linestyle="-.",label="Spitzer"))
    # make ellipses
    pylab.gca().add_patch(Ellipse([218.03607,34.33062],4.8,3.8,fill=None,label="GMRT 153MHz",linestyle="-",linewidth=0.5*lwfactor))
    pylab.gca().add_patch(Ellipse([217.990991788,34.3211253693],5.1,4.15,fill=None,label="VLA 324.5 MHz",linestyle="--",linewidth=0.5*lwfactor))
    pylab.gca().add_patch(Ellipse([217.778608838,34.270842462],5.4,4.5,fill=None,linestyle=":",linewidth=0.5*lwfactor,label="LOFAR LBA",angle=-40))
    # make scatterplots
    #pylab.scatter(WiseRA,WiseDec,s=1,label="WISE")
#    pylab.gca().add_patch(pylab.Polygon([[213.5,30.1],[224.8,30.1],[224.8,38.9],[213.5,38.9]],label="WISE All-Sky Survey",alpha=0.1,color="r"))
    pylab.scatter(XBootesRA,XBootesDec,s=0.4,color="g",alpha=1,label="XBootes")

    #pylab.scatter(SpitzerloRA,SpitzerloDec,s=1,color="royalblue",alpha=0.1,label=r'Spitzer')# $4.5\mu$m')
    #pylab.gca().add_patch(pylab.Polygon([[216.192,35.6757],[219.597,35.6757],[219.548,33.8594],[217.831,32.4453],[216.258,32.4531]],alpha=0.1,color="royalblue"))
#    pylab.gca().add_patch(pylab.Polygon([[216.192,35.6757],[219.597,35.6757],[219.548,33.8594],[217.831,32.4453],[216.258,32.4531]],fill=None,linestyle="--",label="Spitzer"))
#    pylab.scatter(SpitzerhiRA,SpitzerhiDec,s=0.1,color="cadetblue",alpha=0.8,label=r'Spitzer $3.6\mu$m')
#    pylab.scatter(gmrtRA,gmrtDec,s=0.5,color="fuchsia",alpha=1,label="GMRT 153MHz")
#    pylab.gca().add_patch(Ellipse([218.03607,34.33062],4.8,3.8,fill=None,label="GMRT 153MHz",linestyle="--",linewidth=0.5))
#    pylab.gca().add_patch(Ellipse([218.03607,34.33062],4.8,3.8,alpha=0.1,color="fuchsia"))
    #pylab.scatter(vlaRA,vlaDec,s=0.5,color="g",alpha=0.1,label="VLA 324.5MHz")
#    pylab.gca().add_patch(Ellipse([217.990991788,34.3211253693],5.1,4.15,fill=None,label="VLA 324.5 MHz",linestyle="-",linewidth=0.5))
    #pylab.scatter(lbaRA,lbaDec,s=1,color="blueviolet",alpha=0.8,label="LOFAR LBA")
#    pylab.gca().add_patch(Ellipse([217.778608838,34.270842462],5.4,4.5,fill=None,linestyle=":",linewidth=0.5,label="LOFAR LBA",angle=-40))
    #pylab.scatter(lalaRA,lalaDec,s=0.1,color="forestgreen",alpha=0.8,label="Chandra LALA sources")
#    pylab.scatter(WendyRA,WendyDec,s=0.1,color="r",alpha=0.8,label="Optical")#Wendy")
#    pylab.gca().add_patch(pylab.Polygon([[216.131,35.8359],[219.763,35.841],[219.69,33.465],[218.968,33.4704],[218.963,32.8756],\
#                                         [218.275,32.8716],[218.275,32.3053],[216.139,32.288]],fill=None,linestyle="-",linewidth=2,label="Optical"))
#    pylab.gca().add_patch(pylab.Polygon([[216.131,35.8359],[219.763,35.841],[219.69,33.465],[218.968,33.4704],[218.963,32.8756],\
#                                         [218.275,32.8716],[218.275,32.3053],[216.139,32.288]],alpha=0.1,color="r"))
#    pylab.scatter(XBootesRA,XBootesDec,s=0.4,color="g",alpha=1,label="XBootes")

    # keep wise as-is
#    pylab.scatter(WiseRA,WiseDec,s=1,label="WISE")
    #pylab.scatter(wsrtRA,wsrtDec,s=1,color="chartreuse",alpha=1,label="WSRT 1.4GHz")
#    pylab.gca().add_patch(pylab.Polygon([[216.73,35.586],[219.294,35.586],[219.294,35.0132],[219.687,35.0132],[219.687,33.39],[219.294,33.39],[219.294,32.875],[216.73,32.875],\
#                                         [216.73,33.39],[216.38,33.39],[216.38,35.0132],[216.73,35.0132]],fill=None,linestyle="-",label="WSRT 1.4GHz"))
#    pylab.gca().add_patch(pylab.Polygon([[216.73,35.586],[219.294,35.586],[219.294,35.0132],[219.687,35.0132],[219.687,33.39],[219.294,33.39],[219.294,32.875],[216.73,32.875],\
#                                         [216.73,33.39],[216.38,33.39],[216.38,35.0132],[216.73,35.0132]],alpha=0.1,edgecolor="green"))
     #pylab.scatter(lofarRA,lofarDec,s=0.2,color="indigo",label="LOFAR HBA")
#    pylab.gca().add_patch(pylab.Polygon([[217,33.51],[217,35.5],[219.45,35.5],[219.45,33.51]],fill=None,linestyle="--",linewidth=2,label="LOFAR HBA"))
#    pylab.gca().add_patch(pylab.Polygon([[217,33.51],[217,35.5],[219.45,35.5],[219.45,33.51]],alpha=0.1,edgecolor="indigo"))
    #pylab.scatter(opticRA,opticDec,s=0.1,alpha=0.8,label="Optical")
    #pylab.scatter(midirRA,midirDec,s=1.,alpha=0.8,label=r'16$\mu$m')
    lgnd=pylab.legend(scatterpoints=1,fontsize=19)
    ftsize=20
    pylab.tick_params(axis="x",labelsize=ftsize)
    pylab.tick_params(axis="y",labelsize=ftsize)
    pylab.xlim([215,221])
    pylab.ylim([31.5,37.5])
    tesst=1
    for i in range(len(lgnd.legendHandles)):
        lgnd.legendHandles[i]._sizes = [60]
        if i > 3:
            tesst=2
        lgnd.legendHandles[i].set_linewidth(lwfactor/tesst)
    lgnd.get_frame().set_alpha(0.8)
    pylab.xlabel("RA [deg]",fontsize=ftsize)
    pylab.xticks()
    pylab.ylabel("Dec [deg]",fontsize=ftsize)
    pylab.tight_layout()
    pylab.savefig("SkyCoverageMap.png")
    pylab.show()
