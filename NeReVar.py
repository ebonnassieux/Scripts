import os
from casacore.tables import table
import numpy as np
import pylab
import sys
import warnings
import time
import argparse
from astropy.time import Time

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.patheffects as pe


class CovWeights:
    """
    A class used to generate interferometric quality-based weighting scheme values.
    Standard use is to instantiate the class, then FindWeights, SaveWeights, close,
    and CreateDiagnostics.

    Attributes:
    --------------
    MSName           : str
       a string containing the path to the Measurement Set for which to 
       generate weighting scheme
    dt               : float
       timescale [min] over which the gain variances should be calculated. Default 0
    nt               : int
       number of timesteps over which gain variances should be calculated. Used
       as alternative to dt. Default 0
    dfreq            : float
       bandwidth [Mhz] over which the gain variances should be calculated. Default 0
    nchan            : int
       number of channels over which gain variances should be calculated. Used
       as alternative to dfreq. Default 0
    SaveDataProducts : bool
       flag to save data products and diagnostics in designated directory, in
       addition to the MS weights column. Default True
    uvcut            : array of floats
       values [km] for inner and outer uvcuts to apply while calculating the gain
       variance values. default [0,20000]
    gainfile         : str
       not used yet. Will eventually allow for specific gain file to be read to
       extract relevant quantities. Default None
    phaseonly        : bool
       flag to avoid using residual amplitude values directly. Can only be set to
       False if amplitude gains are provided through a gainfile. Default True
    antnorm          : bool
       flag to normalise antenna coefficient array per antenna rather than globally.
       functionality not yet implemented. Default False.
    modelcolname     : str
       name of the gain-corrupted model data column to be read in order to generate
       the residuals. Default MODEL_DATA.
    datacolname      : str
       name of the raw data column from which to subtract modelcolname visibilities.
       Default DATA.
    weightscolname   : str
       name of the MS column in which to save the weighting scheme generated. Will be
       created if not already there. Default IMAGING_WEIGHT.
    verbose          : bool
       flag to print out stdout information. Default: True
    diagdir          : str
       Name of the directory in which to place data products and 
       diagnostic plots. Will be created if not existing. Default is NeReVar.
    """

    ### initialise the instance.
    def __init__(self, MSName, dt=0, nt=0, dfreq=0, nchan=0, SaveDataProducts=True, \
                 uvcut=[0,2000], gainfile=None, phaseonly=True,antnorm=False, \
                 modelcolname="MODEL_DATA", datacolname="DATA", \
                 weightscolname="IMAGING_WEIGHT", verbose=True,diagdir="NeReVar"):
        """
        A class used to generate interferometric quality-based weighting scheme values.
        Standard use is to instantiate the class, then FindWeights, SaveWeights, close,
        and CreateDiagnostics.

        Attributes:
        --------------
        MSName           : str
            a string containing the path to the Measurement Set for which to
            generate weighting scheme
        dt               : float
            timescale [min] over which the gain variances should be calculated. Default 0
        nt               : int
            number of timesteps over which gain variances should be calculated. Used
            as alternative to dt. Default 0
        dfreq            : float
            bandwidth [Mhz] over which the gain variances should be calculated. Default 0
        nchan            : int
            number of channels over which gain variances should be calculated. Used
            as alternative to dfreq. Default 0
        SaveDataProducts : bool
            flag to save data products and diagnostics in designated directory, in
            addition to the MS weights column. Default True
        uvcut            : [float, float]
            values [km] for inner and outer uvcuts to apply while calculating the gain
            variance values. default [0,20000]
        gainfile         : str
            functionality not yet implemented. Do not use.
        phaseonly        : bool
            flag to avoid using residual amplitude values directly. Can only be set to
            False if amplitude gains are provided through a gainfile. Default True
        antnorm          : bool
            flag to normalise antenna coefficient array per antenna rather than globally.
            functionality not yet implemented. Default False.
        modelcolname     : str
            name of the gain-corrupted model data column to be read in order to generate
            the residuals. Default MODEL_DATA.
        datacolname      : str
            name of the raw data column from which to subtract modelcolname visibilities.
            Default DATA.
        weightscolname   : str
            name of the MS column in which to save the weighting scheme generated. Will be
            created if not already there. Default IMAGING_WEIGHT.
        verbose          : bool
            flag to print out stdout information. Default: True
        diagdir          : str
            Name of the directory in which to place data products and
            diagnostic plots. Will be created if not existing. Default is NeReVar.
        """
        # define verbosity
        self.verbose          = verbose
        if MSName[-1]=="/":
            self.MSName       = MSName[0:-1]
        else:
            self.MSName       = MSName
        self.SaveDataProducts = SaveDataProducts
        if self.SaveDataProducts:
            # define directory in which to save stuff + diagnostics
            if diagdir[0]=="/":
                self.DiagDir  = diagdir
            else:
                self.DiagDir  = self.MSName+"/"+diagdir
            if self.DiagDir[-1]!="/":
                self.DiagDir=self.DiagDir+"/"
        self.dfreq            = int(dfreq / 2)
        self.uvcut            = uvcut
        self.gainfile         = gainfile
        self.phaseonly        = phaseonly
        self.antnorm          = antnorm
        self.ms               = table(self.MSName,readonly=False,ack=self.verbose)
        # open antennas
        self.ants             = table(self.ms.getkeyword("ANTENNA"),ack=self.verbose)
        # open antenna tables
        self.antnames         = self.ants.getcol("NAME")
        self.nAnt             = len(self.antnames)
        # open uvw to perform uvcut
        self.u,self.v,_       = self.ms.getcol("UVW").T
        # load ant indices
        self.A0               = self.ms.getcol("ANTENNA1")
        self.A1               = self.ms.getcol("ANTENNA2")
        # create time array
        self.tarray           = self.ms.getcol("TIME")
        self.t0               = np.min(self.tarray)
        self.tvals            = np.sort(list(set(self.tarray)))
        self.t0               = np.min(self.tvals)
        self.dumptime         = self.tvals[1] - self.tvals[0]
        # set initial time to 0; if MJD needed add self.t0 again
        self.tarray           = self.tarray - self.t0
        self.tvals            = self.tvals  - self.t0
        # create time intervals
        if nt!=0: 
            if dt!=0:
                if self.verbose:
                    print("dt and nt are incompatible; nt is retained.")
            self.nt           = nt
            self.dt           = nt*self.dumptime
        else:
            self.nt           = dt / self.dumptime
            self.dt           = dt
        # load frequency information
        self.freqs            = table(self.ms.getkeyword("SPECTRAL_WINDOW"),ack=self.verbose)
        self.chanfreqs        = self.freqs.getcol("CHAN_FREQ")
        self.chanwidth        = np.mean(self.freqs.getcol("CHAN_WIDTH"))
        # create freq intervals
        if nchan !=0:
            if self.verbose:
                print("dnu and nchan are incompatible; nchan is retained.")
            self.nchan        = nchan
            self.dfreq        = nchan * self.chanwidth
        else:
            self.nchan        = dfreq / self.chanwidth
            self.dfreq        = dfreq
        # since we are looking forward and backward, divide intervals by 2
        self.dt               =     self.dt     / 2.
        self.nt               = int(int(self.nt   ) / 2. + (self.nt    % 2 > 0))
        self.dfreq            =     self.dfreq  / 2.
        self.nchan            = int(int(self.nchan) / 2. + (self.nchan % 2 > 0))
        self.nbl              = np.where(self.tarray==self.tarray[0])[0].size
        self.colnames         = self.ms.colnames()
        self.modelcolname     = modelcolname
        self.datacolname      = datacolname
        self.weightscolname   = weightscolname
        if self.modelcolname in self.colnames and self.datacolname in self.colnames:
            if self.verbose:
                print("Creating RESIDUAL_DATA equivalent from %s - %s"%(self.datacolname,self.modelcolname))
            self.residualdata = ms.getcol(self.datacolname)-ms.getcol(self.datacolname)
        elif "RESIDUAL_DATA" in self.colnames:
            if self.verbose:
                print("Reading RESIDUAL_DATA directly from MS")
            self.residualdata = self.ms.getcol("RESIDUAL_DATA")
        else:
            if self.verbose:
                print("Model, data colnames not present; RESIDUAL_DATA column not in measurement set: reading CORRECTED_DATA")
            self.residualdata = self.ms.getcol("CORRECTED_DATA")
        self.nChan            = self.residualdata.shape[1]
        self.nPola            = self.residualdata.shape[2]
        self.nt               = int(self.residualdata.shape[0]/self.nbl)
        self.flags            = self.ms.getcol("FLAG")
        # apply uvcut
        self.uvlen            = np.sqrt(self.u**2+self.v**2)
        self.flags[self.uvlen>self.uvcut[1]]=1
        self.flags[self.uvlen<self.uvcut[0]]=1
        # apply flags to data
        self.residualdata[self.flags==1]=0
        # reshape antennas and data columns for convenience [LOFAR only]
        self.residualdata     = self.residualdata.reshape((self.nt,self.nbl,self.nChan,self.nPola))
        self.flags            = self.flags.reshape((self.nt,self.nbl,self.nChan,self.nPola))
        self.tarray           = self.tarray.reshape((self.nt,self.nbl))
        self.A0               = self.A0.reshape((self.nt,self.nbl))
        self.A1               = self.A1.reshape((self.nt,self.nbl))
        self.ant1             = np.arange(self.nAnt)
        self.residuals             = np.zeros_like(self.residualdata,dtype=np.complex64)
        # remove crosspols
        self.residuals[:,:,:,0]    = self.residualdata[:,:,:,0]
        self.residuals[:,:,:,1]    = self.residualdata[:,:,:,3]
        # build antenna coefficient array
        self.CoeffArray       = np.zeros((self.nAnt, self.nt, self.nChan))

    ### calculate the weights
    def FindWeights(self):
        """
        Function to calculate the antenna-based variance estimates
        used as the baseline for NeReVar's quality-based weighting
        scheme. This creates self.CoeffArray and writes it to the
        diagnostic directory if the latter is requested.
        """
        if self.verbose:
            print("Begin calculating antenna-based coefficients")
        mask   = np.zeros_like(self.residuals).astype(bool)
        for t_i,t_val in enumerate(self.tvals):
            # mask for relevant times within dt
            tmask  = ( (t_val+self.dt  >= self.tvals) * (t_val-self.dt  <= self.tvals))
            # build weights for each antenna at time t_i
            for ant in self.ant1:
                Resids = self.residuals[tmask]
                # build mask for set of vis w/ ant-ant_i and ant_i-ant bls
                antmask    = (self.A0[tmask]==ant) + (self.A1[tmask]==ant)
                AntResids  = Resids[antmask]
                AbsResids  = np.abs(AntResids)
                # before averaging operation, check if the data is not flagged to save time
                for chan_i in range(self.nChan):
                    chanmin    = max(0,chan_i-self.nchan)
                    vals       = AntResids[:,chanmin:(chan_i+self.nchan),:]
                    weights    = AbsResids[:,chanmin:(chan_i+self.nchan),:]                    
                    if np.sum(weights) > 0:
                        self.CoeffArray[ant, t_i, chan_i] = np.average( np.real( vals * vals.conj() ), \
                                                                        weights = weights.astype(bool) )
                    else:
                        # if flagged, set var estimates to 0
                        self.CoeffArray[ant, t_i, chan_i] = 0
            if self.verbose:
                PrintProgress(t_i,self.nt)
        for i in range(self.nAnt):
            # flag any potential NaNs
            self.CoeffArray[np.isnan(self.CoeffArray)]=np.inf
            # if requested, flag per antenna
            if self.antnorm:
                for i in range(self.nAnt):
                    print(self.CoeffArray.shape)
                    antcoeffs = self.CoeffArray[i,:,:]
                    # check that the full antenna is not flagged
                    if np.sum(antcoeffs)!=0:
                        self.CoeffArray[i,:,:] = np.average(antcoeffs, weights=antcoeffs.astype(bool))
                        
            else:
                self.CoeffArray = self.CoeffArray /    \
                    np.average( self.CoeffArray, weights = self.CoeffArray.astype(bool))
            
            # create diagnostic directory if not yet created
            if self.SaveDataProducts:
                if not os.path.exists(self.DiagDir):
                    os.makedirs(self.DiagDir)
                coeffFilename = self.DiagDir+"CoeffArray.npy"
                if self.verbose:
                    print("Save coefficient array as %s."%coeffFilename)
                np.save(coeffFilename,self.CoeffArray)

    ### save the weights in the designated measurement set column
    def SaveWeights(self):
        if self.verbose:
            print("Begin saving the data")
        if self.weightscolname in self.ms.colnames():
            if self.verbose:
                print("%s column already present; will overwrite"%self.weightscolname)
        else:
            W=np.ones((self.nt*self.nbl,self.nChan,self.nPola))
            desc=self.ms.getcoldesc("WEIGHT_SPECTRUM")
            desc["name"]=self.weightscolname
            desc['comment']=desc['comment'].replace(" ","_")
            self.ms.addcols(desc)
            self.ms.putcol(self.weightscolname,W)
        # create weight array
        w=np.zeros((self.nt,self.nbl,self.nChan,self.nPola))
        ant1=np.arange(self.nAnt)
        if self.verbose:
            print("Fill weights array")
        for i in range(self.nt):
            A0ind = self.A0[i,:]
            A1ind = self.A1[i,:]
            for j in range(self.nbl):
                for k in range(self.nChan):
                    var1=self.CoeffArray[A0ind[j],i,k]
                    var2=self.CoeffArray[A1ind[j],i,k]
                    if var1 and var2 != 0:
                        weights = 1. / ( var1 + var2 )
                    else:
                        weights=0
                    for k1 in range(self.nPola):
                        w[i,j,k,k1] = weights
            if self.verbose:
                PrintProgress(i,self.nt)
        w=w.reshape(self.nt*self.nbl,self.nChan,self.nPola)
        w = w / np.average(w,weights=w.astype(bool))
        if self.weightscolname!=None:
            self.ms.putcol(self.weightscolname,w)
        else:
            if self.verbose:
                print("No colname given, so weights not saved in MS.")
        self.weights = w

    ### exit gracefully
    def close(self):
        """
        Function to close open read-write files. Should be called after
        SaveWeights is run, and will not affect CreateDiagnosticPlots.
        """
        self.ants.close()
        self.ms.close()

    ### create diagnostic plots if requested
    def CreateDiagnosticPlots(self):
        # create the output directory
        if not os.path.exists(self.DiagDir):
            os.makedirs(self.DiagDir)
        # create the coefficient directory
        self.CoeffDict = {"Times"    : self.tvals,
                          "ObsStart" : Time(self.t0/3600./24,format="mjd").iso,
                          "ObsEnd"   : Time((self.t0+np.max(self.tvals))/3600./24,format="mjd").iso,
                          "Freqs"    : self.chanfreqs,
                          "Antennas" : self.antnames,
                          "CoeffArr" : {},
                          }
        for idx, ant in enumerate(self.antnames):
            self.CoeffDict["CoeffArr"][ant] = self.CoeffArray[idx, : :]
        # apply flags for the diagnostics
        flags    = (self.flags.reshape((self.nbl*self.nt,self.nChan,self.nPola)).astype(bool) == False)
        uvflags  = (np.sum(flags,axis=(1,2)).astype(bool))
        uvdist   = np.sqrt(self.u[uvflags]**2 + self.v[uvflags]**2)/1000.
        for i in range(self.nChan):
            pylab.scatter(uvdist,self.weights[uvflags,i,0],s=0.1)
        pylab.xlabel(r"$uv$-distance [km]")
        pylab.ylabel(r"Weight value")
        pylab.savefig(self.DiagDir+"WeightsUVWave")

        nAnts = len(self.CoeffDict['Antennas'])
        nColumns = int(np.floor(np.sqrt(nAnts)))
        nRows = int(np.ceil(nAnts/nColumns))
        
        t_max = (np.max(self.CoeffDict['Times'])-np.min(self.CoeffDict['Times']))/3600
        chan_max = len(self.CoeffDict['Freqs'])
        
        #Start plot
        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(nRows, nColumns)
        gs.update(wspace=0, hspace=0, right=0.945, left=0.03, top=0.99, bottom=0.025)
        
        for idx_y in range(nRows):
            for idx_x in range(nColumns):
                ant_idx = nColumns*idx_y + idx_x                
                ax = fig.add_subplot(gs[idx_y,idx_x])
                
                ax.set_xlim((0, t_max))
                ax.set_ylim((0, chan_max))
                
                if idx_x > 0:
                    ax.yaxis.set_ticklabels([])
                else:
                    ticklabels=ax.yaxis.get_ticklabels()
                    if ticklabels[-1]._y > 0.9*chan_max and idx_y != 0:
                        ax.yaxis.set_ticklabels(ticklabels[:-1])
                if idx_y < nRows-1:
                    ax.xaxis.set_ticklabels([])
                else:
                    ticklabels=ax.xaxis.get_ticklabels()
                    if ticklabels[-1]._x > 0.9*t_max and idx_x != nColumns-1:
                        ax.xaxis.set_ticklabels(ticklabels[:-1])
                ax.tick_params(axis='y',which='major',direction='in',left='on',right='on')
                ax.tick_params(axis='x',which='major',direction='in',bottom='on',top='on')

                
                #Check if we're not overshooting
                if ant_idx >= nAnts:
                    break
                
                ant_name = self.CoeffDict['Antennas'][ant_idx]
                
                coeffs = self.CoeffDict['CoeffArr'][ant_name].T
                coeffs[coeffs==0] = np.nan
                im = ax.imshow(coeffs, origin='lower', aspect='auto', interpolation='none', extent=(0, t_max, 0, chan_max), vmin=0.24, vmax=2.30)
                ax.set_title(ant_name, c='w', fontsize=8, y=1.0, pad=-14, path_effects=[pe.withStroke(linewidth=1, foreground="black")])

        cbar_ax = fig.add_axes([0.95, 0.025, 0.02, 0.965])
        fig.colorbar(im, cax=cbar_ax, aspect=40, extend='both', pad=0.02, fraction=0.047)
        plt.savefig(self.DiagDir+"coeffs.png", dpi=300)

        
### auxiliary functions ###
### printer for when needed
def PrintProgress(currentIter,maxIter,msg=""):
    sys.stdout.flush()
    if msg=="":
        msg="Progress:"
    sys.stdout.write("\r%s %5.1f %% "%(msg,100*(currentIter+1.)/maxIter))
    if currentIter==(maxIter-1):
        sys.stdout.write("\n")

### parser
def readArguments():
    parser=argparse.ArgumentParser("Calculate visibility imagin weights based on calibration quality")
    parser.add_argument("-v","--verbose",          help="Be verbose, say everything program does. Default is False",required=False,action="store_true")
    parser.add_argument("--filename",  type=str,   help="Name of the measurement set for which weights want to be calculated",required=True,nargs="+")
    parser.add_argument("--dt",        type=float, help="Time interval, in minutes, for variance estimation. "+\
                        "Default of 0 means an estimate is made for every measurement.",required=False, default=0)
    parser.add_argument("--nt",        type=int,   help="Time interval, in timesteps, for variance estimation. "+\
                        "Default of 0 means an estimate is made for every measurement. If both dt and nt provided, nt prevails."\
                        ,required=False, default=0)
    parser.add_argument("--dnu",       type=float, help="Frequency interval, in MHz, for variance estimation. Default of 0, "+\
                        "which solves across all frequency in the dataset.",required=False,default=0)
    parser.add_argument("--nchan",     type=int,   help="Frequency interval, in channels, for variance estimation. Default of 0, "+\
                        "which solves across all frequency in the dataset.",required=False,default=0)
    parser.add_argument("--weightcol", type=str,   help="Name of the weights column name you want to save the weights to. "+\
                        "Default is QUAL_WEIGHT.",required=False,default="QUAL_WEIGHT")
    parser.add_argument("--datacol",   type=str,   help="Name of the data column name you want to read to build residual visibilities. "+\
                        "Default is DATA.",required=False,default="DATA")
    parser.add_argument("--modelcol",  type=str,   help="Name of the weights column name you want to save the weights to. "+\
                        "Default is MODEL_DATA_CORR.",required=False,default="MODEL_DATA_CORR")
    parser.add_argument("--gainfile",  type=str,   help="Name of the gain file you want to read to rebuild the calibration quality weights."+\
                        " If no file is given, equivalent to rebuilding weights for phase-only calibration.",required=False,default="")
    parser.add_argument("--uvcutkm",   type=float, nargs=2,default=[0,3000],required=False,help="uvcut used during calibration, in km.")
    parser.add_argument("--phaseonly",             help="Use if calibration was phase-only; "+\
                        "this means that gain information doesn't need to be read.",required=False,action="store_true")
    parser.add_argument("--diagnostics",type=str, default="NeReVar_Diagnostics",required=False,\
                        help="Full path and name of folder in which to save diagnostic plots. By default, will save in MS/NeReVar_Diagnostics")
    parser.add_argument("--NormPerAnt",            help="Normalise gains per antenna to avoid suppressing long baselines", \
                        required=False,action="store_true")
    args=parser.parse_args()
    return vars(args)



### if program is called as main ###
if __name__=="__main__":
    start_time     = time.time()
    # read arguments
    args           = readArguments()
    verb           = args["verbose"]
    mslist         = args["filename"]
    dt             = args["dt"]*60.
    nt             = args["nt"]
    nchan          = args["nchan"]
    dfreq          = args["dnu"]*1.e6
    weightscolname = args["weightcol"]
    modelcolname   = args["modelcol"]
    datacolname    = args["datacol"]
    gainfile       = args["gainfile"]
    uvcut          = args["uvcutkm"]*1000
    phaseonly      = args["phaseonly"]
    NormPerAnt     = args["NormPerAnt"]
    diagdir        = args["diagnostics"]
    # calculate weights for each measurement set
    for msname in mslist:
        if verb:
            print("Finding time-covariance weights for: %s"%msname)
        covweights=CovWeights(MSName=msname,dt=dt,nt=nt, nchan=nchan, dfreq=dfreq, gainfile=gainfile,uvcut=uvcut,phaseonly=phaseonly, \
                              antnorm=NormPerAnt, modelcolname=modelcolname, datacolname=datacolname, weightscolname=weightscolname,verbose=verb, \
                              diagdir=diagdir)
        coefficients=covweights.FindWeights()
        covweights.SaveWeights()
        covweights.CreateDiagnosticPlots()
        covweights.close()
        if verb:
            print("Total runtime: %f min"%((time.time()-start_time)/60.))
