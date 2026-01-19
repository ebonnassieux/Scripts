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


* SkyModel
  * Array
    . RecArrayOps.py
    . __init__.py
  * Mask
    . ClassBrightFaintOverAll.py
    . ClassBrightFaintOverAllDEAP.py
    . ClassBrightFaintPerFacet.py
    . __init__.py
  * Other
    . ClassCasaImage.py
    . ModColor.py
    . ModCoord.py
    . MyHist.py
    . MyLogger.py
    . MyPickle.py
    . __init__.py
    . progressbar.py
    . rad2hmsdms.py
    . reformat.py
    . terminal.py
  * PSourceExtract
    . ClassFitIslands.py
    . ClassGaussFit.py
    . ClassIncreaseIsland.py
    . ClassIslands.py
    . ClassPointFit.py
    . ClassPointFit2.py
    . Gaussian.py
    . ModConvPSF.py
    . TestGaussFit.py
    . __init__.py
    . findrms.py
  * Sky
    * Models/LOFAR
      . CasA.txt
      . CygA.txt
      . TauA.txt
      . VirA.txt
    . ClassAppendSource.py
    . ClassClusterClean.py
    . ClassClusterDEAP.py
    . ClassClusterKMean.py
    . ClassClusterRadial.py
    . ClassClusterSquareRadial.py
    . ClassClusterTessel.py
    . ClassMetricDEAP.py
    . ClassSM.py
    . DeapAlgo.py
    . ModBBS2np.py
    . ModKMean.py
    . ModRegFile.py
    . ModSMFromFITS.py
    . ModSMFromNp.py
    . ModTigger.py
    . ModVoronoi.py
    . ModVoronoiToReg.py
    . __init__.py
  * Test
    . ModelRandom00.txt
  * Tools
    . ModFFTW.py
    . PolygonTools.py
    . __init__.py
  . .gitignore
  . ClusterCat.py
  . ExtractPSources.py
  . Gaussify.py
  . MakeCatalog.py
  . MakeMask.py
  . MakeModel.py
  . MaskDicoModel.py
  . MyCasapy2bbs.py
  . PEX.py
  . __init__.py
  . __main__.py
  . dsm.py
  . dsreg.py
. .gitignore
. .gitmodules
. Jenkinsfile.sh
. LICENSE.md
. README.rst
. apt.sources.list
. docker.2004
. docker.2204
. migratenumpy.sh
. pyproject.toml


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

