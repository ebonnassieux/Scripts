import numpy as np
from casacore.tables import table
import astropy.units as u
from astropy.coordinates import SkyCoord
import pandas as pd
import os

# for debug
import pylab

from astropy.constants import c


def get_lmn(phasecentre:SkyCoord,
            sourcecoords:SkyCoord):
    refra  = phasecentre.ra.rad
    refdec = phasecentre.dec.rad
    ras    = sourcecoords.ra.rad
    decs   = sourcecoords.dec.rad
    l = np.cos(decs) * np.sin(ras - refra)
    m = np.sin(decs) * np.cos(refdec) - np.cos(decs) * np.sin(refdec) * np.cos(ras - refra)
    n = np.sqrt(1 - l**2 - m**2)
    return l,m,n

def get_phasecentre(msname:str,
                    fieldid=0,
                    spw=0) -> SkyCoord:
    # read phase centre from the MS
    phase_dir_tab = table(msname+"/FIELD",ack=False)
    phasecentr    = phase_dir_tab.getcol("PHASE_DIR")
    refradRA      = phasecentr[fieldid,spw,0] # fieldid=spw=0 for LOFAR
    refradDEC     = phasecentr[fieldid,spw,1] # fieldid=spw=0 for LOFAR
    phasecentre   = SkyCoord(refradRA*u.rad,refradDEC*u.rad,frame="fk5")
    phase_dir_tab.close()
    return phasecentre

def AddDataColumn(msname  : str,
                  colname : str):
        ms = table(msname,readonly=False)
        desc=ms.getcoldesc("DATA")
        desc["name"]=colname
        desc['comment']=desc['comment'].replace(" ","_")
        ms.addcols(desc)
        ms.putcol(colname,np.zeros_like(ms.getcol("DATA")))
        ms.close()

def FixBBScsv(filename,
              overwrite=True):
#    print("BBS format cannot be fixed. Writing a proper csv instead.")
    # open file
    with open(filename) as f:
        lines = f.readlines()
        # check header length
        headlen = len(lines[0].split(","))
#        print("header length:", headlen)
        # remove extra \n in header
        lines[0]=lines[0].replace("\n","").replace(" ","")
        maxlinelen=0
        avglinelen=[]
        for line in lines[1:]:
            # skip comments, skip empty lines
            if line[0]!="#" and "," in line:
                linelen = len(line.split(","))
                maxlinelen = max(maxlinelen,linelen)
        popinds=[]
        for i,line in enumerate(lines[1:]):
            # purge empty lines, comments and patch descriptions.
            if "," not in line or line[0]=='#' or len(line.split(","))<headlen:
                popinds.append(i+1)
            # fix SPI by replacing , in CSV with ;
            else:
                comps = line.split("[")
                prefix = comps[0]+"["
                comps = comps[1].split("]")
                suffix = "]"+comps[1].replace("\n","")
                spivals = comps[0].replace(",",";")
                nuline = prefix+spivals+suffix
                lines[i+1]=nuline
        for popind in sorted(popinds,reverse=True):
            lines.pop(popind)
    if os.path.isfile(filename)==False:
        with open(filename+".csv","w") as f:
            for line in lines:
                f.write(line+"\n")
    elif os.path.isfile(filename) and overwrite==True:
        with open(filename+".csv","w") as f:
            for line in lines:
                f.write(line+"\n")
    return filename+".csv"


def ReadBBSmodel(filename):
    # clean up spectral index stuff so it can be read as CSV
    skymodel = pd.read_csv(FixBBScsv(filename))
    # clean up Dec column for SkyCoord readability
    for i in range(len(skymodel)):
        temp = skymodel.loc[i,"Dec"].split(".")
        skymodel.loc[i,"Dec"] = temp[0]+":"+temp[1]+":"+temp[2]+"."+temp[3]
    # more cleanup as needed
    todo=1
    # return the final pandas object
    return skymodel

def generate_model_data(ms,
                        skymodel:pd.DataFrame,
                        outcol:str,
                        gains=None|np.ndarray
                        ):
    # get source coords and lmn
    source_coords = SkyCoord(skymodel["Ra"].values,
                            skymodel["Dec"].values,
                            unit=(u.hourangle, u.deg),
                            frame="fk5")
    l,m,n    = get_lmn(phasecentre,source_coords)
    fluxes  = skymodel["I"].values
    u_,v_,w_=ms.getcol("UVW").T*u.m
    freqs = table(msname+"/SPECTRAL_WINDOW/",ack=False)
    chanfreqs = freqs.getcol("CHAN_FREQ")*u.Hz
    freqs.close()
    wavelengths = c / chanfreqs
    one_over_lambda = 1/wavelengths
    u_ = u_.reshape(len(u_),1)
    v_ = v_.reshape(len(v_),1)
    w_ = w_.reshape(len(w_),1)
    uvals = np.dot(u_,one_over_lambda)
    vvals = np.dot(v_,one_over_lambda)
    wvals = np.dot(w_,one_over_lambda)
    # simulate visibilities
    model_data = np.zeros_like(ms.getcol("DATA"))
    nsources = len(l)
    for i in range(nsources):
        print("    Making predicted vis for source %i/%i"%(i+1,nsources))
        # skip empty flux sources
        if skymodel["I"].values[i] !=0:
            print("     -> do stokes I predict")
            # generate model_data from https://arxiv.org/pdf/1101.1764 sec1.7 eq10
            ### DEBUG: THERE SHOULD BE A - IN FRONT OF THIS, BUT COORDINATES THEN BREAK. FIGURE OUT WHAT WENT WRONG.
            modelI = fluxes[i]*np.exp(2 * np.pi * 1j *( uvals*l[i] + vvals*m[i] + wvals*(n[i]-1)))
            # add Stokes-I flux to autocorrelations
            model_data[:,:,0]+= modelI
            model_data[:,:,3]+= modelI
    if type(gains) is np.ndarray:
        model_data = apply_gains(model_data,
                                 gains,
                                 ms.getcol("ANTENNA1"),
                                 ms.getcol("ANTENNA2"),
                                 ms.getcol("TIME"),
                                 short_timescale=120,
                                 long_timescale=1800
                                 )

    if outcol not in ms.colnames():
        AddDataColumn(msname,outcol)
    ms.putcol(outcol,model_data)


def generate_gains(ms,
                   amptscale:float=120,
                   phasetscale:float=1.,
                   dphase=0.1,
                   ampchanscale:float=4.,
                   mode:str="IFull",
                   ampscale=10,
                   phaselim=2*np.pi
                   ) -> np.ndarray:
    d=ms.getcol("DATA")
    nants=len(np.unique(ms.getcol("ANTENNA1")))+1
    nt=len(list(set(ms.getcol("TIME"))))
    nch=d.shape[1]
    npol=d.shape[2]
    if mode=="iFull":
        ncorrs=npol
    elif mode=="iDiag":
        ncorrs=2
    elif mode=="Scalar":
        ncorrs=1
    gainshape = (nants,nt,nch,npol)
    amps   = np.random.rand(nants,npol)*ampscale
    phases = np.random.rand(nants,npol)*phaselim#/1e15 # debug
    dphases = np.tile(np.dot(np.reshape(np.random.rand(nants)*dphase,(nants,1)),
                     np.reshape(np.arange(nt),(1,nt))),npol).reshape(nants,nt,npol)
    amptvar = 0.2*np.sin(np.arange(nt)/amptscale-np.random.uniform(amptscale))+1
    amptemps = np.zeros((nants,nt,npol))
    for i in range(npol):
        dphases[:,:,i] = dphases[:,:,i] + np.tile(phases[:,i],nt).reshape(nants,nt)
        amptemps[:,:,i] = np.dot(np.reshape(amps[:,i],(nants,1)),np.reshape(amptvar,(1,nt)))
    phases=dphases
    amps = amptemps
    # add amplitude chan variation later.
    gains = amps * np.exp(1j*phases)
    # filter for ifull, idiag, scalar.
    if mode=="scalar":
        gains[:,:,3]=gains[:,:,0]
        gains[:,:,1]=np.zeros_like(gains[:,:,0])
        gains[:,:,2]=np.zeros_like(gains[:,:,0])
    if mode=="iDiag":
        gains[:,:,1]=np.zeros_like(gains[:,:,0])
        gains[:,:,2]=np.zeros_like(gains[:,:,0])
    return gains

def apply_gains(ms,
                gains:np.ndarray,
                colname:str) -> None:
    data = ms.getcol(colname)
    ant1 = ms.getcol("ANTENNA1")
    ant2 = ms.getcol("ANTENNA2")
    nant = len(np.unique(ant1))+1
    nbl  = int( (nant * (nant - 1))/2) 
    nt   = len(list(set(ms.getcol("TIME"))))
    nch  = data.shape[1]
    npol = data.shape[2]
    # reshape data for gain application operations
    data = data.reshape(nt,nbl,nch,npol)
    ant1 = ant1.reshape(nt,nbl)
    ant2 = ant2.reshape(nt,nbl)
    for iant in range(nant):
        maskshape = (nant-1-iant)*nt
        # apply G_p
        g = gains[iant,:,:]
        # do Jones transpose per-freq instead of naïve g_H = g.conj().T
        g_H = np.copy(g).conj()
        g_H[:,1] = g[:,2].conj()
        g_H[:,2] = g[:,1].conj()
        mask1 = (ant1==iant)
        mask2 = (ant2==iant)
        tempdata_g = data[mask1].reshape(nant-1-iant,nt,nch,npol)
        tempdata_gh = data[mask2].reshape(iant,nt,nch,npol)
        for j in range(nch):
            ### apply dot products explicitly
            ###iant=nant will always be missing from antenna1
            if iant<nant-1:
                # G dot V
                tempdata_g[:,:,j,0] = g[:,0]*tempdata_g[:,:,j,0] + g[:,1]*tempdata_g[:,:,j,2]
                tempdata_g[:,:,j,1] = g[:,0]*tempdata_g[:,:,j,1] + g[:,1]*tempdata_g[:,:,j,3]
                tempdata_g[:,:,j,2] = g[:,2]*tempdata_g[:,:,j,0] + g[:,3]*tempdata_g[:,:,j,2]
                tempdata_g[:,:,j,3] = g[:,2]*tempdata_g[:,:,j,1] + g[:,3]*tempdata_g[:,:,j,3]
            ### iant=0 will always be missing from antenna2
            if iant>0:
                # V dot G_H
                tempdata_gh[:,:,j,0] = tempdata_gh[:,:,j,0]*g_H[:,0] + tempdata_gh[:,:,j,1]*g_H[:,2]
                tempdata_gh[:,:,j,1] = tempdata_gh[:,:,j,0]*g_H[:,1] + tempdata_gh[:,:,j,1]*g_H[:,3]
                tempdata_gh[:,:,j,2] = tempdata_gh[:,:,j,2]*g_H[:,0] + tempdata_gh[:,:,j,3]*g_H[:,2]
                tempdata_gh[:,:,j,3] = tempdata_gh[:,:,j,2]*g_H[:,1] + tempdata_gh[:,:,j,3]*g_H[:,3]
        # save updated gains
        data[mask1] = tempdata_g.reshape((nant-1-iant)*nt,nch,npol)
        data[mask2] = tempdata_gh.reshape(iant*nt,nch,npol)
    # write to MS
    data = data.reshape(nt*nbl,nch,npol)
    ms.putcol(colname,data)

def add_vis_noise_to_col(ms,
                     colname:str,
                     image_rms:float):
    # load data to know how many vis we have
    data = ms.getcol(colname)
    # image_rms = sigma_vis / sqrt(N_vis) -- inverted
    sigma = image_rms *np.sqrt(data.shape[0]*data.shape[1])
    noisevals1 = np.random.normal(loc=0,
                                  scale=sigma,
                                  size=data.shape)
    noisevals2 = 1j*np.random.normal(loc=0,
                                     scale=sigma,
                                     size=data.shape)
    data=data+noisevals1+noisevals2
    ms.putcol(colname,data)


def filter_skymodel(skymodel,
                    threshold_Jy=0.1):
    skymodel=skymodel[skymodel["I"]>threshold_Jy]
    
    return skymodel
    


if __name__=="__main__":
    ########### input values - TODO get an optparse for these.
    ### this script takes in an mslist, provided below.
    mslist = ["calib.2gc.ms"]
    ### this script takes in gain and skymodel files, provided below
    skymodel_list  = ["calib.2gc.skymodel0",
                      "calib.2gc.skymodel1",
                      "calib.2gc.skymodel2",
                      "calib.2gc.skymodel3"]
#    skymodel_list  = ["calib.debug.skymodel",
#                      "calib.debug.copy.skymodel"]
    ### this script generates per-visibility noise appropriate to your target image rms, provided below
    image_rms = 1.e-3
    ### this script treats all sources below a certain flux level as unknown by the MODEL_DATA column. This level is given below
    model_flux_threshold=0

    ### begin with measurement overview
    print()
    print("Starting simple observation simulation script.")
    for msname in mslist:
        print("  Measurement set file containing new simulation :",msname)
        ms=table(msname,readonly=False,ack=False)
        # reinitialise DATA column to be empty
        d = np.zeros_like(ms.getcol("DATA"))
        ms.putcol("DATA",d)
        # calculate noise once, we want the same one for all dirs
        add_vis_noise_to_col(ms,"DATA",image_rms)
        noisevals=ms.getcol("DATA")
        d = d + noisevals
        # read phase centre from the MS
        phasecentre   = get_phasecentre(msname)
        print("  Simulation phase centre                        :",phasecentre.to_string("hmsdms"))
        # start DDcal
        for idir,skymodel_name in enumerate(skymodel_list):
            print(f"  Doing direction {idir}")
            # read coordinates from the SM
            skymodel = ReadBBSmodel(skymodel_name)
            print(skymodel)
            # add skymodel to MS
            skymodel.to_csv(f"{msname}/complete_skymodel_{idir}.csv")
            print("    True number of sources in the sky model        :",len(skymodel))
            # generate true sky for this dir
            generate_model_data(ms, skymodel, outcol=f"DATA_{idir}")
            # generate gains for this dir
            tscale = np.random.uniform(0.1,idir+1)*np.random.uniform(0.1,10)*1800
            ascale = np.random.uniform(0.1,idir+1)*np.random.uniform(0.5,120)
            gains = generate_gains(ms,
                                   phasetscale=tscale,
                                   amptscale=ascale,
                                   mode="iDiag")
            print(f"    Saving DD gains to {msname}/gains_dir{idir}.npy")
            np.save(f"{msname}/gains_dir{idir}",gains)
            # apply gains for this dir
            apply_gains(ms,gains,colname=f"DATA_{idir}")
            # preserve non-noisy measurement in data
            di = ms.getcol(f"DATA_{idir}")
            d = d + di
            # add noise to measurements
            ms.putcol(f"DATA_{idir}",di+noisevals)
            print("    Wrote noisy, corrupted vis to                  :",f"DATA_{idir}")
            # filter sky for this dir
            skymodel1 = filter_skymodel(skymodel,threshold_Jy=model_flux_threshold)
            # add incomplete skymodel to MS
            skymodel1.to_csv(f"{msname}/incomplete_skymodel_{idir}.csv")
            # write filtered MODEL_DATA column for this dir
            generate_model_data(ms, skymodel1, outcol=f"MODEL_DATA_{idir}")
            print("    Wrote incomplete sky model visibilities to     :",f"MODEL_DATA_{idir}")
        # fill DI data column
        ms.putcol("DATA",d)
        print("Wrote full DD observation to DATA.")
        # exit gracefully
        ms.close()
