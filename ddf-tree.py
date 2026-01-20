# prefix components:
space =  '    '
branch = '│   '
# pointers:
tee =    '├── '
last =   '└── '

### build up branch dict
ddf_dict={}
ddf_dict["codebase"] = "DDFacet codebase"
ddf_dict["branch"]   = "MassiveMerge_PR_SSD3_FullParallel_OverlapIslands_ModelImage"

ddf_dict["DDFacet"]={"comment":"This is the main DDFacet repository."}

ddf_dict["DDFacet"]["Array"]={"comment":"describe"}
ddf_dict["DDFacet"]["Array"]["ModLinAlg.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Array"]["ModSharedArray.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Array"]["NpParallel.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Array"]["NpShared.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Array"]["PrintRecArray.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Array"][" __init__.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Array"]["lsqnonneg.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Array"]["shared_dict.py"]={"comment":"describe"}

ddf_dict["DDFacet"]["Data"]={"comment":"describe"}
ddf_dict["DDFacet"]["Data"]["ClassATCABeam.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Data"]["ClassBeamMean.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Data"]["ClassDaskMS.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Data"]["ClassData.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Data"]["ClassEveryBeam.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Data"]["ClassFITSBeam.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Data"]["ClassGMRTBeam.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Data"]["ClassJones.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Data"]["ClassLOFARBeam.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Data"]["ClassMS.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Data"]["ClassNenuBeam.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Data"]["ClassSmearMapping.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Data"]["ClassSmoothJones.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Data"]["ClassStokes.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Data"]["ClassVisServer.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Data"]["ClassWeightMachine.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Data"]["PointingProvider.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Data"]["__init__.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Data"]["sidereal.py"]={"comment":"describe"}

ddf_dict["DDFacet"]["Gridder"]={"comment":"describe"}
ddf_dict["DDFacet"]["Gridder"]["old_c_gridder"]={"comment":"describe"}
ddf_dict["DDFacet"]["Gridder"]["old_c_gridder"]["CMakeLists.txt"]={"comment":"describe"}
ddf_dict["DDFacet"]["Gridder"]["old_c_gridder"]["Constants.h"]={"comment":"describe"}
ddf_dict["DDFacet"]["Gridder"]["old_c_gridder"]["Gridder.c"]={"comment":"describe"}
ddf_dict["DDFacet"]["Gridder"]["old_c_gridder"]["Gridder.h"]={"comment":"describe"}
ddf_dict["DDFacet"]["Gridder"]["old_c_gridder"]["GridderSmearPols.c"]={"comment":"describe"}
ddf_dict["DDFacet"]["Gridder"]["old_c_gridder"]["GridderSmearPols.h"]={"comment":"describe"}
ddf_dict["DDFacet"]["Gridder"]["old_c_gridder"]["JonesServer.c"]={"comment":"describe"}
ddf_dict["DDFacet"]["Gridder"]["old_c_gridder"]["Matrix.c"]={"comment":"describe"}
ddf_dict["DDFacet"]["Gridder"]["old_c_gridder"]["Semaphores.h"]={"comment":"describe"}
ddf_dict["DDFacet"]["Gridder"]["old_c_gridder"]["Stokes.h"]={"comment":"describe"}
ddf_dict["DDFacet"]["Gridder"]["old_c_gridder"]["Tools.h"]={"comment":"describe"}
ddf_dict["DDFacet"]["Gridder"]["Arrays.cc"]={"comment":"describe"}
ddf_dict["DDFacet"]["Gridder"]["CMakeLists.txt"]={"comment":"describe"}
ddf_dict["DDFacet"]["Gridder"]["CorrelationCalculator.h"]={"comment":"describe"}
ddf_dict["DDFacet"]["Gridder"]["DecorrelationHelper.cc"]={"comment":"describe"}
ddf_dict["DDFacet"]["Gridder"]["DecorrelationHelper.h"]={"comment":"describe"}
ddf_dict["DDFacet"]["Gridder"]["GridderSmearPols.cc"]={"comment":"describe"}
ddf_dict["DDFacet"]["Gridder"]["JonesServer.h"]={"comment":"describe"}
ddf_dict["DDFacet"]["Gridder"]["JonesServer.cc"]={"comment":"describe"}
ddf_dict["DDFacet"]["Gridder"]["Semaphores.cc"]={"comment":"describe"}
ddf_dict["DDFacet"]["Gridder"]["Semaphores.h"]={"comment":"describe"}
ddf_dict["DDFacet"]["Gridder"]["Stokes.h"]={"comment":"describe"}
ddf_dict["DDFacet"]["Gridder"]["__init__.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Gridder"]["common.h"]={"comment":"describe"}
ddf_dict["DDFacet"]["Gridder"]["degridder.h"]={"comment":"describe"}
ddf_dict["DDFacet"]["Gridder"]["gridder.h"]={"comment":"describe"}

ddf_dict["DDFacet"]["Imager"]={"comment":"describe"}

ddf_dict["DDFacet"]["Imager"]["GA"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["GA"]["ClassArrayMethodGA.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["GA"]["ClassEvolveGA.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["GA"]["__init__.py"]={"comment":"describe"}

ddf_dict["DDFacet"]["Imager"]["HOGBOM"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["HOGBOM"]["ClassImageDeconvMachineHogbom.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["HOGBOM"]["ClassModelMachineHogbom.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["HOGBOM"]["__init__.py"]={"comment":"describe"}

ddf_dict["DDFacet"]["Imager"]["MSMF"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["MSMF"]["ClassImageDeconvMachineMSMF.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["MSMF"]["ClassModelMachineMSMF.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["MSMF"]["ClassMultiScaleMachine.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["MSMF"]["__init__.py"]={"comment":"describe"}

ddf_dict["DDFacet"]["Imager"]["MultiFields"]={"comment":"describe"}      
ddf_dict["DDFacet"]["Imager"]["MultiFields"]["AppendSubFieldInfo.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["MultiFields"]["ClassDeconvMachineMultiField.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["MultiFields"]["ClassFacetMachineMultiFields.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["MultiFields"]["ClassImageDeconvMachineMultiFields.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["MultiFields"]["ClassImageNoiseMachine.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["MultiFields"]["ClassImageNoiseMachineMultiField.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["MultiFields"]["ClassMaskMachineMultiFields.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["MultiFields"]["ClassModelMachineMultiField.py"]={"comment":"describe"}

ddf_dict["DDFacet"]["Imager"]["MultiSliceDeconv"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["MultiSliceDeconv"]["MORESANE"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["MultiSliceDeconv"]["MORESANE"]["ClassMoresane.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["MultiSliceDeconv"]["MORESANE"]["ClassMoresaneSingleSlice.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["MultiSliceDeconv"]["MORESANE"]["TryMORESANEDeconv.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["MultiSliceDeconv"]["MORESANE"]["__init__.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["MultiSliceDeconv"]["Orieux"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["MultiSliceDeconv"]["Orieux"]["Edwin"]={"comment":"plein de trucs là-dedans. Librairie d\'Orieux."}
ddf_dict["DDFacet"]["Imager"]["MultiSliceDeconv"]["Orieux"]["ClassOrieux.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["MultiSliceDeconv"]["Orieux"]["__init__.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["MultiSliceDeconv"]["ClassImageDeconvMachineMultiSlice.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["MultiSliceDeconv"]["ClassModelMachineMultiSlice.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["MultiSliceDeconv"]["__init__.py"]={"comment":"describe"}
      
ddf_dict["DDFacet"]["Imager"]["SASIR"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["SASIR"]["ClassSasir.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["SASIR"]["TrySasirDeconv.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["SASIR"]["__init__.py"]={"comment":"describe"}
      
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
      
ddf_dict["DDFacet"]["Imager"]["WSCMS"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["WSCMS"]["ClassImageDeconvMachineWSCMS.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["WSCMS"]["ClassModelMachineWSCMS.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["WSCMS"]["ClassScaleMachine.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["WSCMS"]["__init__.py"]={"comment":"describe"}

ddf_dict["DDFacet"]["Imager"]["WSCMS2"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["WSCMS2"]["ClassImageDeconvMachineWSCMS.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["WSCMS2"]["ClassModelMachineWSCMS.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["WSCMS2"]["ClassScaleMachine.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["WSCMS2"]["ClassWSCMS_MinorLoop.py"]={"comment":"describe. Looks like an update from WSCMS."}
ddf_dict["DDFacet"]["Imager"]["WSCMS2"]["__init__.py"]={"comment":"describe"}

ddf_dict["DDFacet"]["Imager"]["ClassCasaImage.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["ClassDDEGridMachine.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["ClassDeconvMachine.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["ClassFacetMachine.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["ClassFacetMachineTessel.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["ClassFrequencyMachine.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["ClassGainMachine.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["ClassImToGrid.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["ClassImageDeconvMachine.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["ClassImageNoiseMachine.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["ClassMaskMachine.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["ClassModelMachine.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["ClassMontblancMachine.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["ClassPSFServer.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["ClassWeighting.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["ModCF.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["ModModelMachine.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Imager"]["__init__.py"]={"comment":"describe"}

ddf_dict["DDFacet"]["Other"]={"comment":"describe"}
ddf_dict["DDFacet"]["Other"]["AsciiReader.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Other"]["AsyncProcessPool.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Other"]["CacheManager.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Other"]["ClassGiveSolsFile.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Other"]["ClassJonesDomains.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Other"]["ClassPrint.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Other"]["ClassTimeIt.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Other"]["Exceptions.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Other"]["MPIManager.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Other"]["ModColor.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Other"]["ModProbeCPU.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Other"]["Multiprocessing.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Other"]["MyImshow.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Other"]["MyLogger.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Other"]["MyPickle.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Other"]["PrintList.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Other"]["PrintOptParse.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Other"]["README-APP.md"]={"comment":"describe"}
ddf_dict["DDFacet"]["Other"]["__init__.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Other"]["grepall.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Other"]["logger.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Other"]["logo.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Other"]["progressbar.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Other"]["reformat.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Other"]["terminal.py"]={"comment":"describe"}

ddf_dict["DDFacet"]["Parset"]={"comment":"describe"}
ddf_dict["DDFacet"]["Parset"]["DefaultParset.cfg"]={"comment":"describe"}
ddf_dict["DDFacet"]["Parset"]["MyOptParse.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Parset"]["ParsetChanges"]={"comment":"describe"}
ddf_dict["DDFacet"]["Parset"]["ReadCFG.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Parset"]["__init__.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Parset"]["ddfacet_stimela_inputs_schema.yaml"]={"comment":"describe"}
ddf_dict["DDFacet"]["Parset"]["ddfacet_stimela_inputs_tweaks.yaml"]={"comment":"describe"}
ddf_dict["DDFacet"]["Parset"]["generate_stimela_schema.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Parset"]["test_recipe.yaml"]={"comment":"describe"}

ddf_dict["DDFacet"]["Tests"]={"comment":"describe"}
ddf_dict["DDFacet"]["Tests"]["DebugParsets"]={"comment":"describe"}
ddf_dict["DDFacet"]["Tests"]["DebugParsets"]["ParsetDDFacet.Imager.txt"]={"comment":"describe"}
ddf_dict["DDFacet"]["Tests"]["DebugParsets"]["ParsetDDFacet.JonesDefs.txt"]={"comment":"describe"}
ddf_dict["DDFacet"]["Tests"]["DebugParsets"]["ParsetDDFacet.txt"]={"comment":"describe"}
ddf_dict["DDFacet"]["Tests"]["DebugParsets"]["simms.sh"]={"comment":"describe"}
ddf_dict["DDFacet"]["Tests"]["DebugParsets"]["tdlconf.profiles"]={"comment":"describe"}
ddf_dict["DDFacet"]["Tests"]["DebugParsets"]["testxcen-f9-ddenorm.parset"]={"comment":"describe"}

ddf_dict["DDFacet"]["Tests"]["FastUnitTests"]={"comment":"describe"}
ddf_dict["DDFacet"]["Tests"]["FastUnitTests"]["TestFitter.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Tests"]["FastUnitTests"]["TestLibraries.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Tests"]["FastUnitTests"]["TestStokesConverter.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Tests"]["FastUnitTests"]["__init__.py"]={"comment":"describe"}

ddf_dict["DDFacet"]["Tests"]["ShortAcceptanceTests"]={"comment":"describe"}
ddf_dict["DDFacet"]["Tests"]["ShortAcceptanceTests"]["ClassCompareFITSImage.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Tests"]["ShortAcceptanceTests"]["TestClean.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Tests"]["ShortAcceptanceTests"]["TestFacetPredict.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Tests"]["ShortAcceptanceTests"]["TestHogbomClean.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Tests"]["ShortAcceptanceTests"]["TestLOFAR_J1329_p4729.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Tests"]["ShortAcceptanceTests"]["TestMontblancPredict.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Tests"]["ShortAcceptanceTests"]["TestOneMinorCycleSubtract.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Tests"]["ShortAcceptanceTests"]["TestSupernovaStokesV.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Tests"]["ShortAcceptanceTests"]["TestUltimateDeconvRealSolsSSD.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Tests"]["ShortAcceptanceTests"]["TestWSCMS.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Tests"]["ShortAcceptanceTests"]["TestWeighting.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Tests"]["ShortAcceptanceTests"]["TestWidefieldDirty.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Tests"]["ShortAcceptanceTests"]["__init__.py"]={"comment":"describe"}

ddf_dict["DDFacet"]["Tests"]["VeryLongAcceptanceTests"]={"comment":"describe"}
ddf_dict["DDFacet"]["Tests"]["VeryLongAcceptanceTests"]["Test3C147.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Tests"]["VeryLongAcceptanceTests"]["TestDEEP2.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Tests"]["VeryLongAcceptanceTests"]["TestDEEP2Montblanc.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Tests"]["VeryLongAcceptanceTests"]["TestDeepClean.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Tests"]["VeryLongAcceptanceTests"]["TestHogbomPolClean.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Tests"]["VeryLongAcceptanceTests"]["TestSupernova.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Tests"]["VeryLongAcceptanceTests"]["__init__.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Tests"]["__init.py__"]={"comment":"describe"}

ddf_dict["DDFacet"]["ToolsDir"]={"comment":"describe"}
ddf_dict["DDFacet"]["ToolsDir"]["CatToFreqs.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["ToolsDir"]["ClassAdaptShape.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["ToolsDir"]["ClassMovieMachine.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["ToolsDir"]["ClassSpectralFunctions.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["ToolsDir"]["Gaussian.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["ToolsDir"]["GeneDist.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["ToolsDir"]["GiveEdges.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["ToolsDir"]["GiveMDC.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["ToolsDir"]["ModCoord.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["ToolsDir"]["ModFFTW.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["ToolsDir"]["ModFitPSF.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["ToolsDir"]["ModFitPoly2D.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["ToolsDir"]["ModMosaic.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["ToolsDir"]["ModParset.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["ToolsDir"]["ModRotate.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["ToolsDir"]["ModTaper.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["ToolsDir"]["ModToolBox.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["ToolsDir"]["__init__.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["ToolsDir"]["casapy2bbs.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["ToolsDir"]["fft_comparison.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["ToolsDir"]["findrms.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["ToolsDir"]["gaussfitter2.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["ToolsDir"]["rad2hmsdms.py"]={"comment":"describe"}

ddf_dict["DDFacet"]["cmake"]={"comment":"describe"}
ddf_dict["DDFacet"]["cmake"]["FindCasaCore.cmake"]={"comment":"describe"}
ddf_dict["DDFacet"]["cmake"]["FindCfitsIO.cmake"]={"comment":"describe"}
ddf_dict["DDFacet"]["cmake"]["FindNumPy.cmake"]={"comment":"describe"}
ddf_dict["DDFacet"]["cmake"]["FindRT.cmake"]={"comment":"describe"}
ddf_dict["DDFacet"]["cmake"]["FindWcsLib.cmake"]={"comment":"describe"}
ddf_dict["DDFacet"]["cmake"]["Findpybind11.cmake"]={"comment":"describe"}
ddf_dict["DDFacet"]["CMakeLists.txt"]={"comment":"describe"}
ddf_dict["DDFacet"]["CompareImages.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["CompareImages.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["DDF.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["DDF_parallel.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["FindDiffsCache.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["MakeMovie.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["MemMonitor.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["Restore.Py"]={"comment":"describe"}
ddf_dict["DDFacet"]["SelfCal.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["SplitMS.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["TensorFlowServerFork.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["__init__.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["__main__.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["compatibility.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["fits2png.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["plot_clean_logs.py"]={"comment":"describe"}
ddf_dict["DDFacet"]["report_version.py"]={"comment":"describe"}


ddf_dict["SkyModel"]={"comment":"describe"}
ddf_dict["SkyModel"]["Array"]={"comment":"describe"}
ddf_dict["SkyModel"]["Array"]["RecArrayOps.py"]={"comment":"describe"}
ddf_dict["SkyModel"]["Array"]["__init__.py"]={"comment":"describe"}
ddf_dict["SkyModel"]["Mask"]={"comment":"describe"}
ddf_dict["SkyModel"]["Mask"]["ClassBrightFaintOverAll.py"]={"comment":"describe"}
ddf_dict["SkyModel"]["Mask"]["ClassBrightFaintOverAllDEAP.py"]={"comment":"describe"}
ddf_dict["SkyModel"]["Mask"]["ClassBrightFaintPerFacet.py"]={"comment":"describe"}
ddf_dict["SkyModel"]["Mask"]["__init__.py"]={"comment":"describe"}
ddf_dict["SkyModel"]["Other"]={"comment":"describe"}
ddf_dict["SkyModel"]["Other"]["ClassCasaImage.py"]={"comment":"describe"}
ddf_dict["SkyModel"]["Other"]["ModColor.py"]={"comment":"describe"}
ddf_dict["SkyModel"]["Other"]["ModCoord.py"]={"comment":"describe"}
ddf_dict["SkyModel"]["Other"]["MyHist.py"]={"comment":"describe"}
ddf_dict["SkyModel"]["Other"]["MyLogger.py"]={"comment":"describe"}
ddf_dict["SkyModel"]["Other"]["MyPickle.py"]={"comment":"describe"}
ddf_dict["SkyModel"]["Other"]["__init__.py"]={"comment":"describe"}
ddf_dict["SkyModel"]["Other"]["progressbar.py"]={"comment":"describe"}
ddf_dict["SkyModel"]["Other"]["rad2hmsdms.py"]={"comment":"describe"}
ddf_dict["SkyModel"]["Other"]["reformat.py"]={"comment":"describe"}
ddf_dict["SkyModel"]["Other"]["terminal.py"]={"comment":"describe"}
ddf_dict["SkyModel"]["PSourceExtract"]={"comment":"describe"}
ddf_dict["SkyModel"]["PSourceExtract"]["ClassFitIslands.py"]={"comment":"describe"}
ddf_dict["SkyModel"]["PSourceExtract"]["ClassGaussFit.py"]={"comment":"describe"}
ddf_dict["SkyModel"]["PSourceExtract"]["ClassIncreaseIsland.py"]={"comment":"describe"}
ddf_dict["SkyModel"]["PSourceExtract"]["ClassIslands.py"]={"comment":"describe"}
ddf_dict["SkyModel"]["PSourceExtract"]["ClassPointFit.py"]={"comment":"describe"}
ddf_dict["SkyModel"]["PSourceExtract"]["ClassPointFit2.py"]={"comment":"describe"}
ddf_dict["SkyModel"]["PSourceExtract"]["Gaussian.py"]={"comment":"describe"}
ddf_dict["SkyModel"]["PSourceExtract"]["ModConvPSF.py"]={"comment":"describe"}
ddf_dict["SkyModel"]["PSourceExtract"]["TestGaussFit.py"]={"comment":"describe"}
ddf_dict["SkyModel"]["PSourceExtract"]["__init__.py"]={"comment":"describe"}
ddf_dict["SkyModel"]["PSourceExtract"]["findrms.py"]={"comment":"describe"}
ddf_dict["SkyModel"]["Sky"]={"comment":"describe"}
ddf_dict["SkyModel"]["Sky"]["Models"]={"comment":"describe"}
ddf_dict["SkyModel"]["Sky"]["Models"]["LOFAR"]={"comment":"describe"}
ddf_dict["SkyModel"]["Sky"]["Models"]["LOFAR"]["CasA.txt"]={"comment":"describe"}
ddf_dict["SkyModel"]["Sky"]["Models"]["LOFAR"]["CygA.txt"]={"comment":"describe"}
ddf_dict["SkyModel"]["Sky"]["Models"]["LOFAR"]["TauA.txt"]={"comment":"describe"}
ddf_dict["SkyModel"]["Sky"]["Models"]["LOFAR"]["VirA.txt"]={"comment":"describe"}
ddf_dict["SkyModel"]["Sky"]["ClassAppendSource.py"]={"comment":"describe"}
ddf_dict["SkyModel"]["Sky"]["ClassClusterClean.py"]={"comment":"describe"}
ddf_dict["SkyModel"]["Sky"]["ClassClusterDEAP.py"]={"comment":"describe"}
ddf_dict["SkyModel"]["Sky"]["ClassClusterKMean.py"]={"comment":"describe"}
ddf_dict["SkyModel"]["Sky"]["ClassClusterRadial.py"]={"comment":"describe"}
ddf_dict["SkyModel"]["Sky"]["ClassClusterSquareRadial.py"]={"comment":"describe"}
ddf_dict["SkyModel"]["Sky"]["ClassClusterTessel.py"]={"comment":"describe"}
ddf_dict["SkyModel"]["Sky"]["ClassMetricDEAP.py"]={"comment":"describe"}
ddf_dict["SkyModel"]["Sky"]["ClassSM.py"]={"comment":"describe"}
ddf_dict["SkyModel"]["Sky"]["DeapAlgo.py"]={"comment":"describe"}
ddf_dict["SkyModel"]["Sky"]["ModBBS2np.py"]={"comment":"describe"}
ddf_dict["SkyModel"]["Sky"]["ModKMean.py"]={"comment":"describe"}
ddf_dict["SkyModel"]["Sky"]["ModRegFile.py"]={"comment":"describe"}
ddf_dict["SkyModel"]["Sky"]["ModSMFromFITS.py"]={"comment":"describe"}
ddf_dict["SkyModel"]["Sky"]["ModSMFromNp.py"]={"comment":"describe"}
ddf_dict["SkyModel"]["Sky"]["ModTigger.py"]={"comment":"describe"}
ddf_dict["SkyModel"]["Sky"]["ModVoronoi.py"]={"comment":"describe"}
ddf_dict["SkyModel"]["Sky"]["ModVoronoiToReg.py"]={"comment":"describe"}
ddf_dict["SkyModel"]["Sky"]["__init__.py"]={"comment":"describe"}
ddf_dict["SkyModel"]["Test"]={"comment":"describe"}
ddf_dict["SkyModel"]["Test"]["ModelRandom00.txt"]={"comment":"describe"}
ddf_dict["SkyModel"]["Tools"]={"comment":"describe"}
ddf_dict["SkyModel"]["Tools"]["ModFFTW.py"]={"comment":"describe"}
ddf_dict["SkyModel"]["Tools"]["PolygonTools.py"]={"comment":"describe"}
ddf_dict["SkyModel"]["Tools"]["__init__.py"]={"comment":"describe"}
ddf_dict["SkyModel"][".gitignore"]={"comment":"describe"}
ddf_dict["SkyModel"]["ClusterCat.py"]={"comment":"describe"}
ddf_dict["SkyModel"]["ExtractPSources.py"]={"comment":"describe"}
ddf_dict["SkyModel"]["Gaussify.py"]={"comment":"describe"}
ddf_dict["SkyModel"]["MakeCatalog.py"]={"comment":"describe"}
ddf_dict["SkyModel"]["MakeMask.py"]={"comment":"describe"}
ddf_dict["SkyModel"]["MakeModel.py"]={"comment":"describe"}
ddf_dict["SkyModel"]["MaskDicoModel.py"]={"comment":"describe"}
ddf_dict["SkyModel"]["MyCasapy2bbs.py"]={"comment":"describe"}
ddf_dict["SkyModel"]["PEX.py"]={"comment":"describe"}
ddf_dict["SkyModel"]["__init__.py"]={"comment":"describe"}
ddf_dict["SkyModel"]["__main__.py"]={"comment":"describe"}
ddf_dict["SkyModel"]["dsm.py"]={"comment":"describe"}
ddf_dict["SkyModel"]["dsreg.py"]={"comment":"describe"}
ddf_dict[".gitignore"]={"comment":"describe"}
ddf_dict[".gitmodules"]={"comment":"describe"}
ddf_dict["Jenkinsfile.sh"]={"comment":"describe"}
ddf_dict["LICENSE.md"]={"comment":"describe"}
ddf_dict["README.rst"]={"comment":"describe"}
ddf_dict["apt.sources.list"]={"comment":"describe"}
ddf_dict["docker.2004"]={"comment":"describe"}
ddf_dict["docker.2204"]={"comment":"describe"}
ddf_dict["migratenumpy.sh"]={"comment":"describe"}
ddf_dict["pyproject.toml"]={"comment":"describe"}

### build up branch strings
for key in ddf_dict:
    if key=="codebase":
        print("################################################################################")
        print(ddf_dict[key])
    elif key=="branch":
        print(ddf_dict[key])
        print("################################################################################")
    else:        
        if key!="comment":
            print(tee+key)
            print(branch+ddf_dict[key]["comment"])
            this_dict=ddf_dict[key]
            for key1 in ddf_dict[key].keys():
                if key1!="comment":
                    print(branch+tee+key1)
                    print(branch+branch+this_dict[key1]["comment"])
                    this_dict1=this_dict[key1]
                    for key2 in this_dict1.keys():
                        if key2!="comment":
                            print(branch+branch+tee+key2)
                            print(branch+branch+branch+this_dict1[key2]["comment"])
                            this_dict2=this_dict1[key2]
                            for key3 in this_dict2.keys():
                                if key3!="comment":
                                    print(branch+branch+branch+tee+key2)
                                    print(branch+branch+branch+branch+this_dict1[key2]["comment"])

