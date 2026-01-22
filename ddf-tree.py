
import numpy as np

### build up branch dict
ddf_dict={}
ddf_dict["Codebase"] = "DDFacet"
ddf_dict["Branch"]   = "MassiveMerge_PR_SSD3_FullParallel_OverlapIslands_ModelImage"

ddf_dict["DDFacet"]={"comment":"This is the main DDFacet repository."}

ddf_dict["DDFacet"]["Array"]={"comment":"Contains array/matrix manipulation functions"}
ddf_dict["DDFacet"]["Array"]["ModLinAlg.py"]={"comment":"Matrix inversion and manipulation functions."}
ddf_dict["DDFacet"]["Array"]["ModSharedArray.py"]={"comment":"Numpy to sharedarray conversion functions."}
ddf_dict["DDFacet"]["Array"]["NpParallel.py"]={"comment":"Functions to add matrixes of different size to each other, and to find max in a matrix"}
ddf_dict["DDFacet"]["Array"]["NpShared.py"]={"comment":"Shared dictionary functions"}
ddf_dict["DDFacet"]["Array"]["PrintRecArray.py"]={"comment":"Prettytable print for recarrays"}
ddf_dict["DDFacet"]["Array"][" __init__.py"]={"comment":"Contains DDF disclaimer"}
ddf_dict["DDFacet"]["Array"]["lsqnonneg.py"]={"comment":"Linear least squares with nonnegativity constraints."}
ddf_dict["DDFacet"]["Array"]["shared_dict.py"]={"comment":"Defines shared-memory dictionary object."}

ddf_dict["DDFacet"]["Data"]={"comment":"Contains functions and classes to read raw and intermediate interferometric data."}
ddf_dict["DDFacet"]["Data"]["ClassATCABeam.py"]={"comment":"Returns ATCA primary beam Jones matrices at requested coordinates."}
ddf_dict["DDFacet"]["Data"]["ClassBeamMean.py"]={"comment":"Smoothes and averages beam values between facets."}
ddf_dict["DDFacet"]["Data"]["ClassDaskMS.py"]={"comment":"Defines the DaskMS object for internal use."}
ddf_dict["DDFacet"]["Data"]["ClassData.py"]={"comment":"Defines data properties: pointing, BLmapping, current data/timebin"}
ddf_dict["DDFacet"]["Data"]["ClassEveryBeam.py"]={"comment":"Returns Everybeam skala40_wave primary beam J-mats at req. coords."}
ddf_dict["DDFacet"]["Data"]["ClassFITSBeam.py"]={"comment":"Interpreter for MeerKAT .fits beam files. Check if up to date..."}
ddf_dict["DDFacet"]["Data"]["ClassGMRTBeam.py"]={"comment":"Returns GMRT primary beam Jones matrices at requested coordinates."}
ddf_dict["DDFacet"]["Data"]["ClassJones.py"]={"comment":"!CRITICAL! Defines DD-Jones matrix format for DDF, incl. read/write."}
ddf_dict["DDFacet"]["Data"]["ClassLOFARBeam.py"]={"comment":"Returns LOFAR primary beam Jones matrices at requested coordinates."}
ddf_dict["DDFacet"]["Data"]["ClassMS.py"]={"comment":"!CRITICAL! Defines the DDF internal MS object."}
ddf_dict["DDFacet"]["Data"]["ClassNenuBeam.py"]={"comment":"Returns NenUFAR primary beam Jones matrices at requested coordinates."}
ddf_dict["DDFacet"]["Data"]["ClassSmearMapping.py"]={"comment":"Computes visibility smearing mapping for given BL t/nu averaging."}
ddf_dict["DDFacet"]["Data"]["ClassSmoothJones.py"]={"comment":"Applies Jones-matrix smoothing by some alpha parameter"}
ddf_dict["DDFacet"]["Data"]["ClassStokes.py"]={"comment":"Functions to go from Stokes to corrs, following NRAO definitions."}
ddf_dict["DDFacet"]["Data"]["ClassVisServer.py"]={"comment":"!CRITICAL! Interface between MS and vis chunks, applying BDA etc."}
ddf_dict["DDFacet"]["Data"]["ClassWeightMachine.py"]={"comment":"Handles imaging weights (Briggs, WEIGHT column etc)"}
ddf_dict["DDFacet"]["Data"]["PointingProvider.py"]={"comment":"Reads external pointing solutions, interpolates in time per antenna."}
ddf_dict["DDFacet"]["Data"]["__init__.py"]={"comment":"Contains DDF disclaimer"}
ddf_dict["DDFacet"]["Data"]["sidereal.py"]={"comment":"External module. \"A Python module for astronomical calculations\". "}

ddf_dict["DDFacet"]["Gridder"]={"comment":"Contains the gridding/degridding functionalities."}
ddf_dict["DDFacet"]["Gridder"]["old_c_gridder"]={"comment":"I assume this is deprecated."}
ddf_dict["DDFacet"]["Gridder"]["old_c_gridder"]["CMakeLists.txt"]={"comment":"Bld and install instructions."}
ddf_dict["DDFacet"]["Gridder"]["old_c_gridder"]["Constants.h"]={"comment":"Defines c and pi."}
ddf_dict["DDFacet"]["Gridder"]["old_c_gridder"]["Gridder.c"]={"comment":"!CRITICAL! Gridder code."}
ddf_dict["DDFacet"]["Gridder"]["old_c_gridder"]["Gridder.h"]={"comment":"!CRITICAL! Gridder header."}
ddf_dict["DDFacet"]["Gridder"]["old_c_gridder"]["GridderSmearPols.c"]={"comment":"Huge file. To investigate."}
ddf_dict["DDFacet"]["Gridder"]["old_c_gridder"]["GridderSmearPols.h"]={"comment":"Smeared gridder header."}
ddf_dict["DDFacet"]["Gridder"]["old_c_gridder"]["JonesServer.c"]={"comment":"Jones, NormJones operations+print."}
ddf_dict["DDFacet"]["Gridder"]["old_c_gridder"]["Matrix.c"]={"comment":"Matrix operations."}
ddf_dict["DDFacet"]["Gridder"]["old_c_gridder"]["Semaphores.h"]={"comment":"Give, set, delete semaphores."}
ddf_dict["DDFacet"]["Gridder"]["old_c_gridder"]["Stokes.h"]={"comment":"IQUV-XYRL conversion functions."}
ddf_dict["DDFacet"]["Gridder"]["old_c_gridder"]["Tools.h"]={"comment":"GiveFreqStep, FATAL+errprint."}
ddf_dict["DDFacet"]["Gridder"]["Arrays.cc"]={"comment":"OpenMP array operations."}
ddf_dict["DDFacet"]["Gridder"]["CMakeLists.txt"]={"comment":"cmake for _pyArrays, _pyGridderSmearPols."}
ddf_dict["DDFacet"]["Gridder"]["CorrelationCalculator.h"]={"comment":"Estimates BDA correction."}
ddf_dict["DDFacet"]["Gridder"]["DecorrelationHelper.cc"]={"comment":"Calculates DecorrFactor."}
ddf_dict["DDFacet"]["Gridder"]["DecorrelationHelper.h"]={"comment":"Header for above."}
ddf_dict["DDFacet"]["Gridder"]["GridderSmearPols.cc"]={"comment":"Stokes management for smearing."}
ddf_dict["DDFacet"]["Gridder"]["JonesServer.cc"]={"comment":"Jones operation functions in C."}
ddf_dict["DDFacet"]["Gridder"]["JonesServer.h"]={"comment":"Header for above."}
ddf_dict["DDFacet"]["Gridder"]["Semaphores.cc"]={"comment":"Give, set, and delete semaphores."}
ddf_dict["DDFacet"]["Gridder"]["Semaphores.h"]={"comment":"Header for above."}
ddf_dict["DDFacet"]["Gridder"]["Stokes.h"]={"comment":"Header for IQUV-XYRL Stokes grid/degrid."}
ddf_dict["DDFacet"]["Gridder"]["__init__.py"]={"comment":"Empty"}
ddf_dict["DDFacet"]["Gridder"]["common.h"]={"comment":"Namespace definition? I don't understand this one."}
ddf_dict["DDFacet"]["Gridder"]["degridder.h"]={"comment":"!CRITTICAL! Header for degridder"}
ddf_dict["DDFacet"]["Gridder"]["gridder.h"]={"comment":"!CRITICAL! Header for gridder"}

ddf_dict["DDFacet"]["Imager"]={"comment":"Contains all imager functionalities, including deconvolution algorithms."}

ddf_dict["DDFacet"]["Imager"]["GA"]={"comment":"Genetic Algorithm - DEAP-based"}
ddf_dict["DDFacet"]["Imager"]["GA"]["ClassArrayMethodGA.py"]={"comment":"Quality metrics, mutation functions, etc"}
ddf_dict["DDFacet"]["Imager"]["GA"]["ClassEvolveGA.py"]={"comment":"Defines the per-island GA evolution of the clean model."}
ddf_dict["DDFacet"]["Imager"]["GA"]["__init__.py"]={"comment":"Contains DDFacet license."}

ddf_dict["DDFacet"]["Imager"]["HOGBOM"]={"comment":"Hogbom clean. Minimial implementation to serve as reference for dev work."}
ddf_dict["DDFacet"]["Imager"]["HOGBOM"]["ClassImageDeconvMachineHogbom.py"]={"comment":"Minimal implementation."}
ddf_dict["DDFacet"]["Imager"]["HOGBOM"]["ClassModelMachineHogbom.py"]={"comment":"Model manipulation functionalities."}
ddf_dict["DDFacet"]["Imager"]["HOGBOM"]["__init__.py"]={"comment":"Contains DDFacet license."}

ddf_dict["DDFacet"]["Imager"]["MSMF"]={"comment":"MultiScaleMultiFrequency clean."}
ddf_dict["DDFacet"]["Imager"]["MSMF"]["ClassImageDeconvMachineMSMF.py"]={"comment":"MSMF implementation. Documented!"}
ddf_dict["DDFacet"]["Imager"]["MSMF"]["ClassModelMachineMSMF.py"]={"comment":"Model manipulation functionalities"}
ddf_dict["DDFacet"]["Imager"]["MSMF"]["ClassMultiScaleMachine.py"]={"comment":"Manages the scales part of MSMF."}
ddf_dict["DDFacet"]["Imager"]["MSMF"]["__init__.py"]={"comment":"Contains DDFacet license."}

ddf_dict["DDFacet"]["Imager"]["MultiFields"]={"comment":"Multifield clean functionalities. Good for LOFAR-VLBI"}      
ddf_dict["DDFacet"]["Imager"]["MultiFields"]["AppendSubFieldInfo.py"]={"comment":"Descriptive."}
ddf_dict["DDFacet"]["Imager"]["MultiFields"]["ClassDeconvMachineMultiField.py"]={"comment":"Multifield deconv. functions."}
ddf_dict["DDFacet"]["Imager"]["MultiFields"]["ClassFacetMachineMultiFields.py"]={"comment":"Coordinates facets."}
ddf_dict["DDFacet"]["Imager"]["MultiFields"]["ClassImageDeconvMachineMultiFields.py"]={"comment":"Handles the deconvolution."}
ddf_dict["DDFacet"]["Imager"]["MultiFields"]["ClassImageNoiseMachine.py"]={"comment":"Gets noise stats, gives brutalrestoreds."}
ddf_dict["DDFacet"]["Imager"]["MultiFields"]["ClassImageNoiseMachineMultiField.py"]={"comment":"Iterates over the fields."}
ddf_dict["DDFacet"]["Imager"]["MultiFields"]["ClassMaskMachineMultiFields.py"]={"comment":"Iterates masks over the fields."}
ddf_dict["DDFacet"]["Imager"]["MultiFields"]["ClassModelMachineMultiField.py"]={"comment":"Iterates models over the fields."}

ddf_dict["DDFacet"]["Imager"]["MultiSliceDeconv"]={"comment":"Not sure what this is, or if it is still used."}
ddf_dict["DDFacet"]["Imager"]["MultiSliceDeconv"]["MORESANE"]={"comment":"Baffling - this looks like dev that was not finalised."}
ddf_dict["DDFacet"]["Imager"]["MultiSliceDeconv"]["MORESANE"]["ClassMoresane.py"]={"comment":"Inherits from FitsImage"}
ddf_dict["DDFacet"]["Imager"]["MultiSliceDeconv"]["MORESANE"]["ClassMoresaneSingleSlice.py"]={"comment":"????"}
ddf_dict["DDFacet"]["Imager"]["MultiSliceDeconv"]["MORESANE"]["TryMORESANEDeconv.py"]={"comment":"Test file for MORESANE. Contains all actual functions."}
ddf_dict["DDFacet"]["Imager"]["MultiSliceDeconv"]["MORESANE"]["__init__.py"]={"comment":"Empty."}
ddf_dict["DDFacet"]["Imager"]["MultiSliceDeconv"]["Orieux"]={"comment":"Orieux deconvolution functions."}
ddf_dict["DDFacet"]["Imager"]["MultiSliceDeconv"]["Orieux"]["Edwin"]={"comment":"plein de trucs là-dedans. Librairie d\'Orieux."}
ddf_dict["DDFacet"]["Imager"]["MultiSliceDeconv"]["Orieux"]["ClassOrieux.py"]={"comment":"Wrapper for Orieux library."}
ddf_dict["DDFacet"]["Imager"]["MultiSliceDeconv"]["Orieux"]["__init__.py"]={"comment":"Empty."}
ddf_dict["DDFacet"]["Imager"]["MultiSliceDeconv"]["ClassImageDeconvMachineMultiSlice.py"]={"comment":"DeconvMachine for multislice."}
ddf_dict["DDFacet"]["Imager"]["MultiSliceDeconv"]["ClassModelMachineMultiSlice.py"]={"comment":"ModelMachine for multislice."}
ddf_dict["DDFacet"]["Imager"]["MultiSliceDeconv"]["__init__.py"]={"comment":"Contains DDFacet license."}
      
ddf_dict["DDFacet"]["Imager"]["SASIR"]={"comment":"JL Starck algo, https://www.cosmostat.org/software/sasir"}
ddf_dict["DDFacet"]["Imager"]["SASIR"]["ClassSasir.py"]={"comment":"DDF class for Sasir"}
ddf_dict["DDFacet"]["Imager"]["SASIR"]["TrySasirDeconv.py"]={"comment":"Test file for Sasir."}
ddf_dict["DDFacet"]["Imager"]["SASIR"]["__init__.py"]={"comment":"Containts DDFacet license."}
      
ddf_dict["DDFacet"]["Imager"]["SSD"]={"comment":"describe"}      
ddf_dict["DDFacet"]["Imager"]["SSD"]["GA"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["SSD"]["GA"]["ClassEvolveGA.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["SSD"]["GA"]["ClassSmearSM.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["SSD"]["GA"]["TryGADeconv.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["SSD"]["GA"]["__init__.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["SSD"]["GA"]["algorithms.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["SSD"]["MCMC"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["SSD"]["MCMC"]["ClassMetropolis.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["SSD"]["MCMC"]["ClassPDFMachine.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["SSD"]["MCMC"]["__init__.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["SSD"]["ClassArrayMethodSSD.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["SSD"]["ClassConvMachine.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["SSD"]["ClassImageDeconvMachineSSD.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["SSD"]["ClassInitSSDModelHMP.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["SSD"]["ClassInitSSDModelMoresane.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["SSD"]["ClassIslandDistanceMachine.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["SSD"]["ClassModelMachineSSD.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["SSD"]["ClassMutate.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["SSD"]["ClassParamMachine.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["SSD"]["__init__.py"]={"comment":"describe"}
      
ddf_dict["DDFacet"]["Imager"]["SSD2"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["SSD2"]["GA"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["SSD2"]["GA"]["ClassEvolveGA.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["SSD2"]["GA"]["ClassSmearSM.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["SSD2"]["GA"]["TryGADeconv.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["SSD2"]["GA"]["__init__.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["SSD2"]["GA"]["algorithms.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["SSD2"]["MCMC"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["SSD2"]["MCMC"]["ClassMetropolis.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["SSD2"]["MCMC"]["ClassPDFMachine.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["SSD2"]["MCMC"]["__init__.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["SSD2"]["ClassArrayMethodSSD.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["SSD2"]["ClassConvMachine.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["SSD2"]["ClassImageDeconvMachineSSD.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["SSD2"]["ClassInitSSDModelHMP.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["SSD2"]["ClassInitSSDModelMoresane.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["SSD2"]["ClassInitSSDModelMultiSlice.py"]={"comment":"describe. One of the SSD2 functions not in SSD."}
ddf_dict["DDFacet"]["Imager"]["SSD2"]["ClassIslandDistanceMachine.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["SSD2"]["ClassModelMachineSSD.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["SSD2"]["ClassMutate.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["SSD2"]["ClassParamMachine.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["SSD2"]["ClassTaylorToPower.py"]={"comment":"describe. One of the SSD2 functions not in SSD."}
ddf_dict["DDFacet"]["Imager"]["SSD2"]["__init__.py"]={"comment":"describe"}
      
ddf_dict["DDFacet"]["Imager"]["SSD3"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["SSD3"]["GA"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["SSD3"]["GA"]["ClassEvolveGA.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["SSD3"]["GA"]["ClassSmearSM.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["SSD3"]["GA"]["TryGADeconv.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["SSD3"]["GA"]["__init__.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["SSD3"]["GA"]["algorithms.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["SSD3"]["MCMC"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["SSD3"]["MCMC"]["ClassMetropolis.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["SSD3"]["MCMC"]["ClassPDFMachine.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["SSD3"]["MCMC"]["__init__.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["SSD3"]["MultiNest"]={"comment":"describe. In SSD3 and not SSD2."}
ddf_dict["DDFacet"]["Imager"]["SSD3"]["MultiNest"]["ClassMultiNest.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["SSD3"]["MultiNest"]["__init__.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["SSD3"]["MultiNest"]["svgd.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["SSD3"]["ClassArrayMethodSSD.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["SSD3"]["ClassBreakIslands.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["SSD3"]["ClassConvMachine.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["SSD3"]["ClassImageDeconvMachineSSD.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["SSD3"]["ClassInitSSDModelHMP.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["SSD3"]["ClassInitSSDModelMoresane.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["SSD3"]["ClassInitSSDModelMultiSlice.py"]={"comment":"describe. One of the SSD2 functions not in SSD."}
ddf_dict["DDFacet"]["Imager"]["SSD3"]["ClassIslandDistanceMachine.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["SSD3"]["ClassModelMachineSSD.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["SSD3"]["ClassMutate.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["SSD3"]["ClassParamMachine.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["SSD3"]["ClassTaylorToPower.py"]={"comment":"describe. One of the SSD2 functions not in SSD."}
ddf_dict["DDFacet"]["Imager"]["SSD3"]["__init__.py"]={"comment":"describe"}
      
ddf_dict["DDFacet"]["Imager"]["WSCMS"]={"comment":"Implementation of the multi-scale algorithm implemented in wsclean."}
ddf_dict["DDFacet"]["Imager"]["WSCMS"]["ClassImageDeconvMachineWSCMS.py"]={"comment":"Implementation of wsclean multiscale clean."}
ddf_dict["DDFacet"]["Imager"]["WSCMS"]["ClassModelMachineWSCMS.py"]={"comment":"Defines WSCMS model."}
ddf_dict["DDFacet"]["Imager"]["WSCMS"]["ClassScaleMachine.py"]={"comment":"Handles the multiscale aspect."}
ddf_dict["DDFacet"]["Imager"]["WSCMS"]["__init__.py"]={"comment":"Contains DDFacet license."}

ddf_dict["DDFacet"]["Imager"]["WSCMS2"]={"comment":"Update on wsclean multiscale clean algo."}
ddf_dict["DDFacet"]["Imager"]["WSCMS2"]["ClassImageDeconvMachineWSCMS.py"]={"comment":"Updated implementation of WSCMS."}
ddf_dict["DDFacet"]["Imager"]["WSCMS2"]["ClassModelMachineWSCMS.py"]={"comment":"Updated."}
ddf_dict["DDFacet"]["Imager"]["WSCMS2"]["ClassScaleMachine.py"]={"comment":"Updated."}
ddf_dict["DDFacet"]["Imager"]["WSCMS2"]["ClassWSCMS_MinorLoop.py"]={"comment":"Absent from initial WSCMS implementation."}
ddf_dict["DDFacet"]["Imager"]["WSCMS2"]["__init__.py"]={"comment":"Contains DDFacet license."}

ddf_dict["DDFacet"]["Imager"]["ClassCasaImage.py"]={"comment":"Follow CASA standards to create a fits image."}
ddf_dict["DDFacet"]["Imager"]["ClassDDEGridMachine.py"]={"comment":"!CRITICAL! This is the DDE gridder code."}
ddf_dict["DDFacet"]["Imager"]["ClassDeconvMachine.py"]={"comment":"!CRITICAL! This is the deconvolver code."}
ddf_dict["DDFacet"]["Imager"]["ClassFacetMachine.py"]={"comment":"Does tesselation, gridding/degridding, projection to image, unprojection to facets."}
ddf_dict["DDFacet"]["Imager"]["ClassFacetMachineTessel.py"]={"comment":"Extends ClassFacetMachine to do Voronoi split of sky."}
ddf_dict["DDFacet"]["Imager"]["ClassFrequencyMachine.py"]={"comment":"Interface to fit frequency axis in model image."}
ddf_dict["DDFacet"]["Imager"]["ClassGainMachine.py"]={"comment":"Defines and applies Clean-gain parameter (NOT antenna gains)"}
ddf_dict["DDFacet"]["Imager"]["ClassImToGrid.py"]={"comment":"Gives gridded visibilities for a given image, I think."}
ddf_dict["DDFacet"]["Imager"]["ClassImageDeconvMachine.py"]={"comment":"Handles image-side functions. Unclear what diff with ClassDeconvMachine."}
ddf_dict["DDFacet"]["Imager"]["ClassImageNoiseMachine.py"]={"comment":"Calculates noise map and gives brutalRestoreds."}
ddf_dict["DDFacet"]["Imager"]["ClassMaskMachine.py"]={"comment":"Mask management code."}
ddf_dict["DDFacet"]["Imager"]["ClassModelMachine.py"]={"comment":"Read/write internal model format to and from image/dico"}
ddf_dict["DDFacet"]["Imager"]["ClassMontblancMachine.py"]={"comment":"Montblanc management code."}
ddf_dict["DDFacet"]["Imager"]["ClassPSFServer.py"]={"comment":"PSF management code, including crop and frequency normalisation handling"}
ddf_dict["DDFacet"]["Imager"]["ClassWeighting.py"]={"comment":"Calculates visibility weights."}
ddf_dict["DDFacet"]["Imager"]["ModCF.py"]={"comment":"Looks like W-term correction code"}
ddf_dict["DDFacet"]["Imager"]["ModModelMachine.py"]={"comment":"Input model image dictionary, instantiates and returns copy of correct ModelMachine"}
ddf_dict["DDFacet"]["Imager"]["__init__.py"]={"comment":"Includes DDFacet license."}

ddf_dict["DDFacet"]["Other"]={"comment":"Contains logger, debug and terminal functionalities."}
ddf_dict["DDFacet"]["Other"]["AsciiReader.py"]={"comment":"Readers for MultiFieldFile, txt, csv, Ascii."}
ddf_dict["DDFacet"]["Other"]["AsyncProcessPool.py"]={"comment":"AsyncProcessPool management functions."}
ddf_dict["DDFacet"]["Other"]["CacheManager.py"]={"comment":"!CRITICAL! Manages DDF caching. Has docs <3"}
ddf_dict["DDFacet"]["Other"]["ClassGiveSolsFile.py"]={"comment":"Handles kMS filenames and solsdir."}
ddf_dict["DDFacet"]["Other"]["ClassJonesDomains.py"]={"comment":"Defines Jones domains + MergeJones."}
ddf_dict["DDFacet"]["Other"]["ClassPrint.py"]={"comment":"Defines DDF output print parameters."}
ddf_dict["DDFacet"]["Other"]["ClassTimeIt.py"]={"comment":"Timer class for logging+monitoring."}
ddf_dict["DDFacet"]["Other"]["Exceptions.py"]={"comment":"Exceptions handler. Also enables pbd."}
ddf_dict["DDFacet"]["Other"]["MPIManager.py"]={"comment":"Handles MPI management. I dont get it."}
ddf_dict["DDFacet"]["Other"]["ModColor.py"]={"comment":"Defines custom DDF output colours"}
ddf_dict["DDFacet"]["Other"]["ModProbeCPU.py"]={"comment":"TrackCPU, gets CPU load and times."}
ddf_dict["DDFacet"]["Other"]["Multiprocessing.py"]={"comment":"Handles shared memory management."}
ddf_dict["DDFacet"]["Other"]["MyImshow.py"]={"comment":"GiveVal, imshow. Looks like convenience function."}
ddf_dict["DDFacet"]["Other"]["MyLogger.py"]={"comment":"Defines and sets logger parameters."}
ddf_dict["DDFacet"]["Other"]["MyPickle.py"]={"comment":"Defines pickle save, load, convert"}
ddf_dict["DDFacet"]["Other"]["PrintList.py"]={"comment":"Defines ListToStr."}
ddf_dict["DDFacet"]["Other"]["PrintOptParse.py"]={"comment":"Defines PrintOptParse."}
ddf_dict["DDFacet"]["Other"]["README-APP.md"]={"comment":"Multiprocessing w/ AsyncProcessPool + SharedDict"}
ddf_dict["DDFacet"]["Other"]["__init__.py"]={"comment":"_handle_exception, {enable/disable}_pdb_on_exception"}
ddf_dict["DDFacet"]["Other"]["grepall.py"]={"comment":"'grep -r \"name\" --include=*.typein .'%(name,typein)"}
ddf_dict["DDFacet"]["Other"]["logger.py"]={"comment":"!CRITICAL! Detailed logger params and behaviour."}
ddf_dict["DDFacet"]["Other"]["logo.py"]={"comment":"ASCII art."}
ddf_dict["DDFacet"]["Other"]["progressbar.py"]={"comment":"Nadia Alramli progress bar."}
ddf_dict["DDFacet"]["Other"]["reformat.py"]={"comment":"Reformats / characters in a string"}
ddf_dict["DDFacet"]["Other"]["terminal.py"]={"comment":"Nadia Alramli terminal variables and functions."}

ddf_dict["DDFacet"]["Parset"]={"comment":"DDF parser and initialiser functionalities."}
ddf_dict["DDFacet"]["Parset"]["DefaultParset.cfg"]={"comment":"Input for ReadCfg.py, functions as documentation."}
ddf_dict["DDFacet"]["Parset"]["MyOptParse.py"]={"comment":"DDF command-line parser."}
ddf_dict["DDFacet"]["Parset"]["ParsetChanges"]={"comment":"Changelog, but from when to when? Mystery."}
ddf_dict["DDFacet"]["Parset"]["ReadCFG.py"]={"comment":"Builds the parset object from the command line."}
ddf_dict["DDFacet"]["Parset"]["__init__.py"]={"comment":"Contains DDFacet license."}
ddf_dict["DDFacet"]["Parset"]["ddfacet_stimela_inputs_schema.yaml"]={"comment":"Stimela yaml for DDF."}
ddf_dict["DDFacet"]["Parset"]["ddfacet_stimela_inputs_tweaks.yaml"]={"comment":"Hacky bugfix for above."}
ddf_dict["DDFacet"]["Parset"]["generate_stimela_schema.py"]={"comment":"Stimela schema regenerator."}
ddf_dict["DDFacet"]["Parset"]["test_recipe.yaml"]={"comment":"Recipe to test Stimela functionalities."}

ddf_dict["DDFacet"]["Tests"]={"comment":"Various unit, functionality and production tests."}
ddf_dict["DDFacet"]["Tests"]["DebugParsets"]={"comment":"Please. I beg you. A crumb of documentation."}
ddf_dict["DDFacet"]["Tests"]["DebugParsets"]["ParsetDDFacet.Imager.txt"]={"comment":"I think this manually sets object values?"}
ddf_dict["DDFacet"]["Tests"]["DebugParsets"]["ParsetDDFacet.JonesDefs.txt"]={"comment":"Same as above for Jones matrices?"}
ddf_dict["DDFacet"]["Tests"]["DebugParsets"]["ParsetDDFacet.txt"]={"comment":"I think this tests facet interpolation?"}
ddf_dict["DDFacet"]["Tests"]["DebugParsets"]["simms.sh"]={"comment":"I think this generates a simulated VLA dataset for tests."}
ddf_dict["DDFacet"]["Tests"]["DebugParsets"]["tdlconf.profiles"]={"comment":"Looks like Cattery params for 4 test simulations."}
ddf_dict["DDFacet"]["Tests"]["DebugParsets"]["testxcen-f9-ddenorm.parset"]={"comment":"Looks like an actual test DDF parset!"}

ddf_dict["DDFacet"]["Tests"]["FastUnitTests"]={"comment":"One functionality test, one unit test, one empty file."}
ddf_dict["DDFacet"]["Tests"]["FastUnitTests"]["TestFitter.py"]={"comment":"Tests deconvolution"}
ddf_dict["DDFacet"]["Tests"]["FastUnitTests"]["TestLibraries.py"]={"comment":"\"No unit tests since gridder was removed\""}
ddf_dict["DDFacet"]["Tests"]["FastUnitTests"]["TestStokesConverter.py"]={"comment":"Actual unit test of Stokes convertor."}
ddf_dict["DDFacet"]["Tests"]["FastUnitTests"]["__init__.py"]={"comment":"Contains DDFacet license."}

ddf_dict["DDFacet"]["Tests"]["ShortAcceptanceTests"]={"comment":"Compares reference images to current outputs made with same params."}
ddf_dict["DDFacet"]["Tests"]["ShortAcceptanceTests"]["ClassCompareFITSImage.py"]={"comment":"Abstract class for testing purposes."}
ddf_dict["DDFacet"]["Tests"]["ShortAcceptanceTests"]["TestClean.py"]={"comment":"Tests various algos results within 1e-5 to 1e-6"}
ddf_dict["DDFacet"]["Tests"]["ShortAcceptanceTests"]["TestFacetPredict.py"]={"comment":"Checks per-facet predict w/wo MeqTr beam."}
ddf_dict["DDFacet"]["Tests"]["ShortAcceptanceTests"]["TestHogbomClean.py"]={"comment":"Compares Hogbom clean image w/ reference."}
ddf_dict["DDFacet"]["Tests"]["ShortAcceptanceTests"]["TestLOFAR_J1329_p4729.py"]={"comment":"Looks like first test on real data?"}
ddf_dict["DDFacet"]["Tests"]["ShortAcceptanceTests"]["TestMontblancPredict.py"]={"comment":"Compares DDF and Montblanc predicts."}
ddf_dict["DDFacet"]["Tests"]["ShortAcceptanceTests"]["TestOneMinorCycleSubtract.py"]={"comment":"Commented out. Come on man"}
ddf_dict["DDFacet"]["Tests"]["ShortAcceptanceTests"]["TestSupernovaStokesV.py"]={"comment":"Compares Stokes-V clean to reference."}
ddf_dict["DDFacet"]["Tests"]["ShortAcceptanceTests"]["TestUltimateDeconvRealSolsSSD.py"]={"comment":"Descriptive name, I assume."}
ddf_dict["DDFacet"]["Tests"]["ShortAcceptanceTests"]["TestWSCMS.py"]={"comment":"Compares WSCMS restored to reference image."}
ddf_dict["DDFacet"]["Tests"]["ShortAcceptanceTests"]["TestWeighting.py"]={"comment":"Tests PSFs for various imaging weights."}
ddf_dict["DDFacet"]["Tests"]["ShortAcceptanceTests"]["TestWidefieldDirty.py"]={"comment":"Tests correctness of big dirty images."}
ddf_dict["DDFacet"]["Tests"]["ShortAcceptanceTests"]["__init__.py"]={"comment":"Empty"}

ddf_dict["DDFacet"]["Tests"]["VeryLongAcceptanceTests"]={"comment":"Standardised calls, testing different fields."}
ddf_dict["DDFacet"]["Tests"]["VeryLongAcceptanceTests"]["Test3C147.py"]={"comment":"Tests CSS field, w/wo beam."}
ddf_dict["DDFacet"]["Tests"]["VeryLongAcceptanceTests"]["TestDEEP2.py"]={"comment":"Tests Hubble DEEP2 field."}
ddf_dict["DDFacet"]["Tests"]["VeryLongAcceptanceTests"]["TestDEEP2Montblanc.py"]={"comment":"As above, w/ Montblanc."}
ddf_dict["DDFacet"]["Tests"]["VeryLongAcceptanceTests"]["TestDeepClean.py"]={"comment":"Dynamic range test."}
ddf_dict["DDFacet"]["Tests"]["VeryLongAcceptanceTests"]["TestHogbomPolClean.py"]={"comment":"Tests full-Stokes clean."}
ddf_dict["DDFacet"]["Tests"]["VeryLongAcceptanceTests"]["TestSupernova.py"]={"comment":"Tests complex polarised field clean."}
ddf_dict["DDFacet"]["Tests"]["VeryLongAcceptanceTests"]["__init__.py"]={"comment":"Contains DDFacet license."}
ddf_dict["DDFacet"]["Tests"]["__init.py__"]={"comment":"Contains DDF license, sets matplotlib backend to Agg."}

ddf_dict["DDFacet"]["ToolsDir"]={"comment":"Contains variety of DDFacet tools."}
ddf_dict["DDFacet"]["ToolsDir"]["CatToFreqs.py"]={"comment":"Sets frequency values for input catalog"}
ddf_dict["DDFacet"]["ToolsDir"]["ClassAdaptShape.py"]={"comment":"Make image cutouts or padding"}
ddf_dict["DDFacet"]["ToolsDir"]["ClassMovieMachine.py"]={"comment":"Makes movies out of png files"}
ddf_dict["DDFacet"]["ToolsDir"]["ClassSpectralFunctions.py"]={"comment":"Includes beam and spi stuff."}
ddf_dict["DDFacet"]["ToolsDir"]["Gaussian.py"]={"comment":"Variety of gaussian functions."}
ddf_dict["DDFacet"]["ToolsDir"]["GeneDist.py"]={"comment":"Generates distributions and samples."}
ddf_dict["DDFacet"]["ToolsDir"]["GiveEdges.py"]={"comment":"Finds edges of arrays."}
ddf_dict["DDFacet"]["ToolsDir"]["GiveMDC.py"]={"comment":"Multi Direction Calib (?)"}
ddf_dict["DDFacet"]["ToolsDir"]["ModCoord.py"]={"comment":"lm-radec coord convertors."}
ddf_dict["DDFacet"]["ToolsDir"]["ModFFTW.py"]={"comment":"Attempt at parallelising FFTW with APP."}
ddf_dict["DDFacet"]["ToolsDir"]["ModFitPSF.py"]={"comment":"Finds restoring beam. THIS ONE SUCKS"}
ddf_dict["DDFacet"]["ToolsDir"]["ModFitPoly2D.py"]={"comment":"Two-dimensional polynomial fit."}
ddf_dict["DDFacet"]["ToolsDir"]["ModMosaic.py"]={"comment":"I think this is a mosaicker."}
ddf_dict["DDFacet"]["ToolsDir"]["ModParset.py"]={"comment":"Parset-objects functions."}
ddf_dict["DDFacet"]["ToolsDir"]["ModRotate.py"]={"comment":"Rotate phase centre."}
ddf_dict["DDFacet"]["ToolsDir"]["ModTaper.py"]={"comment":"Gaussian tapering function."}
ddf_dict["DDFacet"]["ToolsDir"]["ModToolBox.py"]={"comment":"Variety of FFTs and functions."}
ddf_dict["DDFacet"]["ToolsDir"]["__init__.py"]={"comment":"Contains DDFacet license"}
ddf_dict["DDFacet"]["ToolsDir"]["casapy2bbs.py"]={"comment":"Casacore image to BBS convertor."}
ddf_dict["DDFacet"]["ToolsDir"]["fft_comparison.py"]={"comment":"Test file for pyFFTW3 and PyFFTW."}
ddf_dict["DDFacet"]["ToolsDir"]["findrms.py"]={"comment":"RMS finder. Different from SkyModel function?"}
ddf_dict["DDFacet"]["ToolsDir"]["gaussfitter2.py"]={"comment":"2D Gaussian fitter w/ various functions."}
ddf_dict["DDFacet"]["ToolsDir"]["rad2hmsdms.py"]={"comment":"rad-HHMMSS convertor."}

ddf_dict["DDFacet"]["cmake"]={"comment":"Contains cmake build functions to find critical dependencies."}
ddf_dict["DDFacet"]["cmake"]["FindCasaCore.cmake"]={"comment":"Try to find Casacore include dirs and libraries."}
ddf_dict["DDFacet"]["cmake"]["FindCfitsIO.cmake"]={"comment":"Try to find CFITSIO."}
ddf_dict["DDFacet"]["cmake"]["FindNumPy.cmake"]={"comment":"Find the Python NumPy package."}
ddf_dict["DDFacet"]["cmake"]["FindRT.cmake"]={"comment":"Check for the presence of RunTime compiler (RT)."}
ddf_dict["DDFacet"]["cmake"]["FindWcsLib.cmake"]={"comment":"Try to find WCSLIB."}
ddf_dict["DDFacet"]["cmake"]["Findpybind11.cmake"]={"comment":"Find the pybind11 package headers as installed with pip <= 9.0.3"}
ddf_dict["DDFacet"]["CMakeLists.txt"]={"comment":"Build and install instructions for DDFacet. Deprecated? IDK"}
ddf_dict["DDFacet"]["CleanSHM"]={"comment":"Intended to clean all semaphores and shared memory. Often only partly successfuly..."}
ddf_dict["DDFacet"]["CompareImages.py"]={"comment":"Takes IM1.fits, IM2.fits and creates {IM1-IM2}.fits residual."}
ddf_dict["DDFacet"]["DDF.py"]={"comment":"Main DDFacet executable."}
ddf_dict["DDFacet"]["DDF_parallel.py"]={"comment":"As above, but MPI, I think."}
ddf_dict["DDFacet"]["FindDiffsCache.py"]={"comment":"Checks if a cache is the same as a reference cache."}
ddf_dict["DDFacet"]["MakeMovie.py"]={"comment":"Makes N snapshot DDf images."}
ddf_dict["DDFacet"]["MemMonitor.py"]={"comment":"Plots memory use."}
ddf_dict["DDFacet"]["Restore.Py"]={"comment":"Makes restored image from clean model + residual image."}
ddf_dict["DDFacet"]["SelfCal.py"]={"comment":"KAFCA self-calibration script."}
ddf_dict["DDFacet"]["SplitMS.py"]={"comment":"Splits MS file from specified t0 to t1."}
ddf_dict["DDFacet"]["TensorFlowServerFork.py"]={"comment":"I don't know what TensorFlow does."}
ddf_dict["DDFacet"]["__init__.py"]={"comment":"Contains DDFacet license. import pkg_resources, sets version number."}
ddf_dict["DDFacet"]["__main__.py"]={"comment":"Calls driver functions."}
ddf_dict["DDFacet"]["compatibility.py"]={"comment":"Defines some python2 vs python3 stuff."}
ddf_dict["DDFacet"]["fits2png.py"]={"comment":"Descriptive filename."}
ddf_dict["DDFacet"]["plot_clean_logs.py"]={"comment":"Makes minimalist logfile from verbose default output."}
ddf_dict["DDFacet"]["report_version.py"]={"comment":"Returns version."}


ddf_dict["SkyModel"]={"comment":"This repository contains the sky model codebase"}
ddf_dict["SkyModel"]["Array"]={"comment":"Contains fundamental recarray manipulation functions."}
ddf_dict["SkyModel"]["Array"]["RecArrayOps.py"]={"comment":"Module to add and remove elements of a recarray."}
ddf_dict["SkyModel"]["Array"]["RecArrayOps.py"]["verbose"]="recarray: Construct an ndarray that allows field access using attributes."
ddf_dict["SkyModel"]["Array"]["RecArrayOps.py"]["dependencies"]=["from __future__ import division, absolute_import, print_function\n",
                                                                 "numpy.lib.recfunctions",
                                                                 "numpy"]
ddf_dict["SkyModel"]["Array"]["RecArrayOps.py"]["functions"]=["AppendField",
                                                              "RemoveField"]
ddf_dict["SkyModel"]["Array"]["__init__.py"]={"comment":"Empty."}
ddf_dict["SkyModel"]["Mask"]={"comment":"Contains functions to make imaging masks."}
ddf_dict["SkyModel"]["Mask"]["ClassBrightFaintOverAll.py"]={"comment":"Make SSD-island-based mask, with FFT filtering applied."}
ddf_dict["SkyModel"]["Mask"]["ClassBrightFaintOverAllDEAP.py"]={"comment":"As above, but for DEAP clustering."}
ddf_dict["SkyModel"]["Mask"]["ClassBrightFaintPerFacet.py"]={"comment":"Make mask for a single facet."}
ddf_dict["SkyModel"]["Mask"]["ClassBrightFaintPerFacet.py"]["dependencies"]=["numpy",
                                                                             "import DDFacet.Imager.SSD.ClassIslandDistanceMachine",
                                                                             "from DDFacet.Other import logger",
                                                                             "log=logger.getLogger(\"ClassBrightFaint\")",
                                                                             "from astropy.io import fits",
                                                                             "import DDFacet.Other.MyPickle",
                                                                             "from matplotlib.path import Path",
                                                                             "from DDFacet.ToolsDir import ModFFTW",
                                                                             "from pyrap.images import image",
                                                                             "import SkyModel.Sky.ModRegFile",
                                                                             "import scipy.signal"]
ddf_dict["SkyModel"]["Mask"]["__init__.py"]={"comment":"Empty"}
ddf_dict["SkyModel"]["Other"]={"comment":"Contains variety of convenience functions."}
ddf_dict["SkyModel"]["Other"]["ClassCasaImage.py"]={"comment":"Creates CASA image objects from inputs."}
ddf_dict["SkyModel"]["Other"]["ModColor.py"]={"comment":"Defines log colours used."}
ddf_dict["SkyModel"]["Other"]["ModCoord.py"]={"comment":"Coordinate operation functions."}
ddf_dict["SkyModel"]["Other"]["MyHist.py"]={"comment":"Percent progress calculator."}
ddf_dict["SkyModel"]["Other"]["MyLogger.py"]={"comment":"Defines DDF logger parameters."}
ddf_dict["SkyModel"]["Other"]["MyPickle.py"]={"comment":"Defines pickling dumping/loading params."}
ddf_dict["SkyModel"]["Other"]["__init__.py"]={"comment":"Empty."}
ddf_dict["SkyModel"]["Other"]["progressbar.py"]={"comment":"Animated progress bar by N. Alramli."}
ddf_dict["SkyModel"]["Other"]["rad2hmsdms.py"]={"comment":"Converts radians to HMSDMS coords."}
ddf_dict["SkyModel"]["Other"]["reformat.py"]={"comment":"Reformats strings with / characters."}
ddf_dict["SkyModel"]["Other"]["terminal.py"]={"comment":"Sets DDF terminal environment variables."}
ddf_dict["SkyModel"]["PSourceExtract"]={"comment":"Directory of methods and functions to find point source flux+position"}
ddf_dict["SkyModel"]["PSourceExtract"]["ClassFitIslands.py"]={"comment":"Fits flux within an island. Sequential, parallel, multiprocessed."}
ddf_dict["SkyModel"]["PSourceExtract"]["ClassGaussFit.py"]={"comment":"Finds Gaussian fits."}
ddf_dict["SkyModel"]["PSourceExtract"]["ClassIncreaseIsland.py"]={"comment":"Embiggens islands (fast and slow options)"}
ddf_dict["SkyModel"]["PSourceExtract"]["ClassIslands.py"]={"comment":"!CRITICAL! Defines island properties."}
ddf_dict["SkyModel"]["PSourceExtract"]["ClassPointFit.py"]={"comment":"Finds point-source fits."}
ddf_dict["SkyModel"]["PSourceExtract"]["ClassPointFit2.py"]={"comment":"Much more sophisticated version of the above."}
ddf_dict["SkyModel"]["PSourceExtract"]["Gaussian.py"]={"comment":"Defines Gaussian properties"}
ddf_dict["SkyModel"]["PSourceExtract"]["ModConvPSF.py"]={"comment":"Includes various Gaussian manipulation functions."}
ddf_dict["SkyModel"]["PSourceExtract"]["TestGaussFit.py"]={"comment":"Testss the GaussFit function."}
ddf_dict["SkyModel"]["PSourceExtract"]["__init__.py"]={"comment":"Empty."}
ddf_dict["SkyModel"]["PSourceExtract"]["findrms.py"]={"comment":"Function to find local rms value."}
ddf_dict["SkyModel"]["Sky"]={"comment":"Direcetory containing variety of sky-plane functions and operations"}
ddf_dict["SkyModel"]["Sky"]["Models"]={"comment":"Contains test models. Probably for A-team clipping."}
ddf_dict["SkyModel"]["Sky"]["Models"]["LOFAR"]={"comment":"Contains LOFAR-freq models."}
ddf_dict["SkyModel"]["Sky"]["Models"]["LOFAR"]["CasA.txt"]={"comment":"Cassiopeia A"}
ddf_dict["SkyModel"]["Sky"]["Models"]["LOFAR"]["CygA.txt"]={"comment":"Cygnus A"}
ddf_dict["SkyModel"]["Sky"]["Models"]["LOFAR"]["TauA.txt"]={"comment":"Tau A"}
ddf_dict["SkyModel"]["Sky"]["Models"]["LOFAR"]["VirA.txt"]={"comment":"Virgo A"}
ddf_dict["SkyModel"]["Sky"]["ClassAppendSource.py"]={"comment":"Adds sources in Model and ephemerids to the model"}
ddf_dict["SkyModel"]["Sky"]["ClassClusterClean.py"]={"comment":"Clusters the clean component source, can plot the process."}
ddf_dict["SkyModel"]["Sky"]["ClassClusterDEAP.py"]={"comment":"As above, but for Distributed Evolutionary Algorithms in Python."}
ddf_dict["SkyModel"]["Sky"]["ClassClusterKMean.py"]={"comment":"As above, but uses K-mean clustering."}
ddf_dict["SkyModel"]["Sky"]["ClassClusterRadial.py"]={"comment":"As above, but based on nearest cluster I think."}
ddf_dict["SkyModel"]["Sky"]["ClassClusterSquareRadial.py"]={"comment":"Not sure what the diff with above is."}
ddf_dict["SkyModel"]["Sky"]["ClassClusterTessel.py"]={"comment":"Clusters in tessels"}
ddf_dict["SkyModel"]["Sky"]["ClassMetricDEAP.py"]={"comment":"Contains DEAP variables and functions for clustering"}
ddf_dict["SkyModel"]["Sky"]["ClassSM.py"]={"comment":"!CRITICAL! Defines the Sky Model object"}
ddf_dict["SkyModel"]["Sky"]["DeapAlgo.py"]={"comment":"Contains DEAP algorithmic functions"}
ddf_dict["SkyModel"]["Sky"]["ModBBS2np.py"]={"comment":"Converter to read BBS format skymodels. Outputs DDF format npy skymodel."}
ddf_dict["SkyModel"]["Sky"]["ModKMean.py"]={"comment":"Tester for the K-mean clustering algorithm."}
ddf_dict["SkyModel"]["Sky"]["ModRegFile.py"]={"comment":"Region-Polygon object functions."}
ddf_dict["SkyModel"]["Sky"]["ModSMFromFITS.py"]={"comment":"Reads .fits file, outputs skymodel catalog."}
ddf_dict["SkyModel"]["Sky"]["ModSMFromNp.py"]={"comment":"Reads numpy array, outputs skymodel catalog."}
ddf_dict["SkyModel"]["Sky"]["ModTigger.py"]={"comment":"Reads Tigger model, outputs skymodel catalog."}
ddf_dict["SkyModel"]["Sky"]["ModVoronoi.py"]={"comment":"Voronoi tesselation functions + test."}
ddf_dict["SkyModel"]["Sky"]["ModVoronoiToReg.py"]={"comment":"Voronoi,Tessel -> Region functions."}
ddf_dict["SkyModel"]["Sky"]["__init__.py"]={"comment":"Empty."}
ddf_dict["SkyModel"]["Test"]={"comment":"Contains a test skymodel file"}
ddf_dict["SkyModel"]["Test"]["ModelRandom00.txt"]={"comment":"Randomly-scattered point sources."}
ddf_dict["SkyModel"]["Tools"]={"comment":"Contains modified (i)FFT functionalities and line-polygon interaction code"}
ddf_dict["SkyModel"]["Tools"]["ModFFTW.py"]={"comment":"Modified (i)FFT functions, including Gaussian convolution (restoring beam?)."}
ddf_dict["SkyModel"]["Tools"]["PolygonTools.py"]={"comment":"Functions to cut polygon objects with a line"}
ddf_dict["SkyModel"]["Tools"]["PolygonTools.py"]["functions"]=["GiveABLin",
                                                               "GiveA",
                                                               "CutLineInside"]
ddf_dict["SkyModel"]["Tools"]["PolygonTools.py"]["dependencies"]=["from __future__ import division, absolute_import, print_function",
                                                                  "Polygon",
                                                                  "numpy"]

ddf_dict["SkyModel"]["Tools"]["__init__.py"]={"comment":"Empty"}
ddf_dict["SkyModel"][".gitignore"]={"comment":"Excludes various compilation/build files from git repo"}
ddf_dict["SkyModel"]["ClusterCat.py"]={"comment":"Groups an input catalog into N clusters as specified by command"}
ddf_dict["SkyModel"]["ExtractPSources.py"]={"comment":"Finds islands of flux in an input image, then fits point sources per island. Outputs SkyModel object."}
ddf_dict["SkyModel"]["Gaussify.py"]={"comment":"As ExtractPSources, but also applies a restoring beam / gaussian blur."}
ddf_dict["SkyModel"]["MakeCatalog.py"]={"comment":"Wrapper for PyBDSF"}
ddf_dict["SkyModel"]["MakeMask.py"]={"comment":"Creates a mask for imaging purposes"}
ddf_dict["SkyModel"]["MakeModel.py"]={"comment":"Manipulates a SkyModel object. Invoked to change the clustering."}
ddf_dict["SkyModel"]["MaskDicoModel.py"]={"comment":"Apply mask object to DicoModel object for filtering purposes."}
ddf_dict["SkyModel"]["MyCasapy2bbs.py"]={"comment":"Convert CASA fits model to BBS format skymodel"}
ddf_dict["SkyModel"]["PEX.py"]={"comment":"Not sure what this does. Looks like the restored image builder? Returns a SkyModel object"}
ddf_dict["SkyModel"]["__init__.py"]={"comment":"Empty"}
ddf_dict["SkyModel"]["__main__.py"]={"comment":"Defines how all objects in Tools are invoked. Looks like it is basically plumbing."}
ddf_dict["SkyModel"]["dsm.py"]={"comment":"Python wrapper for ds9, with convenient defaults for image visualisation/comparison."}
ddf_dict["SkyModel"]["dsreg.py"]={"comment":"Python wrapper for ds9 with region loading."}
ddf_dict[".gitignore"]={"comment":"Defines files to avoid pushing to git repo"}
ddf_dict[".gitmodules"]={"comment":"Empty"}
ddf_dict["Jenkinsfile.sh"]={"comment":"Instructions for Jenkins deployment"}
ddf_dict["LICENSE.md"]={"comment":"GNU General Public License"}
ddf_dict["README.rst"]={"comment":"Deprecated README doc."}
ddf_dict["apt.sources.list"]={"comment":"Used for the docker build files"}
ddf_dict["docker.2004"]={"comment":"Deprecated 20.04 dockerfile build file, I assume"}
ddf_dict["docker.2204"]={"comment":"Ubuntu 22.04 dockerfile build instructions"}
ddf_dict["migratenumpy.sh"]={"comment":"looks like a hacky numpy bugfix"}
ddf_dict["pyproject.toml"]={"comment":"List of dependencies"}

def DicoDepth(dico):
    if isinstance(dico, dict):
        return 1 + (max(map(DicoDepth, dico.values())) if dico else 0)
    return 0

def MakeSubDico(dico,subdico,subdiconame):
    output=subdico.copy()
    output["Codebase"]=dico["Codebase"]+"/"+subdiconame#.copy()
    output["Branch"]=dico["Branch"]#.copy()
    return output


def PrintDictStructure(dico,exclude=["comment","verbose","dependencies","functions"],verbose=["comment"],depth=0,named_columns=[],filename=None):
    # prefix components:
    space  =  '    '
    branch = '│   '
    # pointers:
    tee    = '├── '
    last   = '└── '
    split  = '┌──'
    cont   = '───'
    # box components
    # find longest string in list to define column lengths
    maxstrlen = 0
    # make the head
    header=[]
    for key in ["Codebase","Branch"]:
        thisstrlen=len(tee+'{0:<8}'.format(key)+" : "+dico[key])        
        if thisstrlen>maxstrlen:
            maxstrlen=thisstrlen
    # print header here
    header.append("┏"+"━"*(maxstrlen)+"┓")
    header.append("┃"+" "*(maxstrlen)+"┃")
    for key in ["Codebase","Branch"]:
        thisstr=('{0:<%i}'%(maxstrlen-4)).format('{0:<8}'.format(key)+" : "+dico[key])
        header.append("┣━━ "+thisstr+" ┃")
    header.append("┣━━ Depth    : %2i"%depth+" "*(maxstrlen-16)+"┃")
    header.append("┃"+" "*(maxstrlen)+"┃")
    header.append("┡"+"━"*(maxstrlen)+"┛")
    for line in header:
        print(line)
    maxstrlen=0
    # start making multicolumn plot
    namedcols= [[] for _ in range(len(named_columns))]
    icol=0
    lastcol=[]

    for key in dico.keys():
        if key in exclude or key=="Codebase" or key=="Branch":
            continue
        elif key in named_columns:
            # make this column
            ### this is where depth comes in
            namedcols[icol].append(branch)
            namedcols[icol].append(tee+key+" ")
            for info in verbose:
                try:
                    namedcols[icol].append(branch+dico[key][info]+" ")
                except KeyError:
                    continue
            # add depth-1 elements
            if depth>0:
                if DicoDepth(dico[key])>0:
                    this_dict=dico[key]
                    for key1 in dico[key].keys():
                        if key1 not in exclude:
                            this_str=branch+tee+key1+" "
                            namedcols[icol].append(branch+branch)
                            namedcols[icol].append(this_str)
                            for info in verbose:
                                try:
                                    infostr=branch+branch+this_dict[key1][info]+" "
                                    namedcols[icol].append(infostr)
                                except KeyError:
                                    continue
                            # add depth-2 elements
                            if depth>1:
                                this_dict1=this_dict[key1]
                                if DicoDepth(this_dict1)>0:
                                    for key2 in this_dict1.keys():
                                        if key2 not in exclude:
                                            namedcols[icol].append(branch+branch+branch)
                                            namedcols[icol].append(branch+branch+tee+key2+" ")
                                            for info in verbose:
                                                try:
                                                    namedcols[icol].append(branch+branch+branch+this_dict1[key2][info]+" ")
                                                except KeyError:
                                                    continue
                                            # add depth-3 elements
                                            if depth>2:
                                                this_dict2=this_dict1[key2]
                                                if DicoDepth(this_dict2)>0:
                                                    for key3 in this_dict2.keys():
                                                        if key3 not in exclude:
                                                            namedcols[icol].append(branch+branch+branch+branch)
                                                            namedcols[icol].append(branch+branch+branch+tee+key3+" ")
                                                            for info in verbose:
                                                                try:
                                                                    namedcols[icol].append(branch+branch+branch+branch+this_dict2[key3][info]+" ")
                                                                except KeyError:
                                                                    continue
                                                            # add depth-4 elements
                                                            if depth>3:
                                                                this_dict3=this_dict2[key3]
                                                                for key4 in this_dict3.keys():
                                                                    if key4 not in exclude:
                                                                        namedcols[icol].append(branch+branch+branch+branch+tee+key4+" ")
                                                                        for info in verbose:
                                                                            try:
                                                                                namedcols[icol].append(branch+branch+branch+branch+branch+this_dict1[key4][info]+" ")
                                                                            except KeyError:
                                                                                continue
                                                                        # add depth-5 elements
                                                                        if depth>4:
                                                                            this_dict4=this_dict3[key4]
                                                                            for key5 in this_dict4.keys():
                                                                                if key5 not in exclude:
                                                                                    namedcols[icol].append(branch+branch+branch+branch+branch+branch)
                                                                                    namedcols[icol].append(branch+branch+branch+branch+branch+tee+key5+" ")
                                                                                    for info in verbose:
                                                                                        try:
                                                                                            namedcols[icol].append(branch+branch+branch+branch+branch+branch+this_dict1[key5][info]+" ")
                                                                                        except KeyError:
                                                                                            continue
            icol+=1
        else:
            # dump all other files in the "remainder" column
            ### this is where depth comes in
            lastcol.append(branch)
            lastcol.append(tee+key+" ")
            for info in verbose:
                try:
                    lastcol.append(branch+dico[key][info]+" ")
                except KeyError:
                    continue
            # add depth-1 elements
            if depth>0:
                if DicoDepth(dico[key])>0:
                    this_dict=dico[key]
                    for key1 in dico[key].keys():
                        if key1 not in exclude:
                            this_str=branch+tee+key1+" "
                            lastcol.append(branch+branch)
                            lastcol.append(this_str)
                            for info in verbose:
                                try:
                                    infostr=branch+branch+this_dict[key1][info]+" "
                                    lastcol.append(infostr)
                                except KeyError:
                                    continue
                            # add depth-2 elements
                            if depth>1:
                                this_dict1=this_dict[key1]
                                if DicoDepth(this_dict1)>0:
                                    for key2 in this_dict1.keys():
                                        if key2 not in exclude:
                                            lastcol.append(branch+branch+branch)
                                            lastcol.append(branch+branch+tee+key2+" ")
                                            for info in verbose:
                                                lastcol.append(branch+branch+branch+this_dict1[key2][info]+" ")
                                            # add depth-3 elements
                                            if depth>2:
                                                this_dict2=this_dict1[key2]
                                                if DicoDepth(this_dict2)>0:
                                                    for key3 in this_dict2.keys():
                                                        if key3 not in exclude:
                                                            lastcol.append(branch+branch+branch+branch)
                                                            lastcol.append(branch+branch+branch+tee+key3+" ")
                                                            for info in verbose:
                                                                lastcol.append(branch+branch+branch+branch+this_dict1[key3][info]+" ")
                                                            # add depth-4 elements
                                                            if depth>3:
                                                                this_dict3=this_dict2[key3]
                                                                for key4 in this_dict3.keys():
                                                                    if key4 not in exclude:
                                                                        lastcol.append(branch+branch+branch+branch+branch)
                                                                        lastcol.append(branch+branch+branch+branch+tee+key4+" ")
                                                                        for info in verbose:
                                                                            lastcol.append(branch+branch+branch+branch+branch+this_dict1[key4][info]+" ")
                                                                        # add depth-5 elements
                                                                        if depth>4:
                                                                            this_dict4=this_dict3[key4]
                                                                            for key5 in this_dict4.keys():
                                                                                if key5 not in exclude:
                                                                                    lastcol.append(branch+branch+branch+branch+branch+branch)
                                                                                    lastcol.append(branch+branch+branch+branch+branch+tee+key5+" ")
                                                                                    for info in verbose:
                                                                                        lastcol.append(branch+branch+branch+branch+branch+branch+this_dict1[key5][info]+" ")
    ### put cols together
    namedcols.append(lastcol)
    ### homogeneise file
    # homogenise column lengths
    maxcollength=0
    for col in namedcols:
        if len(col)>maxcollength:
            maxcollength=len(col)
    for col in namedcols:
        thiscollen=len(col)
        for i in range(maxcollength-thiscollen):
            col.append("")    
    # homogenise line lengths
    for icol,col in enumerate(namedcols):
        linelength=0
        for line in col:
            if len(line)>linelength:
                linelength=len(line)
        for iline,line in enumerate(col):
            namedcols[icol][iline]=line+" "*(linelength-len(line))
    # make line linking header to body
    headlines=[]
    for col in namedcols:
        headline = "┬"+"─"*(len(col[0])-1)
        headlines.append(headline)
    headlines[0]=headlines[0].replace("┬","├")
    if len(headlines)==1:
        headlines[-1]="│"
    else:
        headlines[-1]="┐"
    ncols=len(namedcols)
    lencols=len(namedcols[0])+1
    namedcols=np.append(np.array(headlines),np.array(namedcols).T).reshape(lencols,ncols)
    ### print to terminal
    for iline in range(namedcols.shape[0]):
        linestr=""
        for icol in range(namedcols.shape[1]):
            linestr = linestr+namedcols[iline,icol]
        print(linestr)
    ### if requested, write to file
    if filename!=None:
        f=open(filename,"w")
        for line in header:
            f.write(line+"\n")
        for iline in range(namedcols.shape[0]):
            linestr=""
            for icol in range(ncols):
                linestr = linestr+namedcols[iline,icol]
            f.write(linestr+"\n")
        f.close()



### make plots etc

                
# print basic DDF folder structure
PrintDictStructure(ddf_dict,named_columns=["DDFacet","SkyModel"],verbose=["comment"],depth=1,filename="DDF-structure.txt")

# print SkyModel folder structure
sky_model_dict=MakeSubDico(ddf_dict,ddf_dict["SkyModel"],"SkyModel")
PrintDictStructure(sky_model_dict,named_columns=["Sky","Other","PSourceExtract"],verbose=["comment","verbose"],depth=4,filename="SkyModel-structure.txt")

# print DDF folder structure
ddfacet_dir_dict=MakeSubDico(ddf_dict,ddf_dict["DDFacet"],"DDFacet")
PrintDictStructure(ddfacet_dir_dict,named_columns=["Imager","ToolsDir"],verbose=["comment"],depth=4,filename="DDFacet-structure.txt")


# print full structure
#PrintDictStructure(ddf_dict,named_columns=["DDFacet","SkyModel"],verbose=["comment"],depth=3)
