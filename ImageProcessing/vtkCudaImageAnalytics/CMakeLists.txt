PROJECT(vtkCudaImageAnalytics)

INCLUDE_DIRECTORIES(${vtkCudaCommon_INCLUDE_DIRS} ${vtkRobartsCommon_INCLUDE_DIRS})

# Add entries to project details
SET(${PROJECT_NAME}_INCLUDE_DIRS ${${PROJECT_NAME}_SOURCE_DIR} ${${PROJECT_NAME}_BINARY_DIR} CACHE INTERNAL "" FORCE)

SET(${PROJECT_NAME}_LIBS vtkCudaCommon vtkRobartsCommon vtkFiltersCore vtkCommonCore)

IF( MSVC )
  SET(${PROJECT_NAME}_HDRS 
    vtkCudaVoxelClassifier.h 
    CUDA_voxelclassifier.h 
    vtkCudaCT2USSimulation.h 
    CUDA_cttoussimulation.h 
    vtkCudaFuzzyConnectednessFilter.h 
    CUDA_fuzzyconnectednessfilter.h 
    vtkCudaKohonenGenerator.h 
    CUDA_kohonengenerator.h 
    vtkCudaKohonenApplication.h
    CUDA_kohonenapplication.h
    vtkCudaKohonenReprojector.h
    CUDA_kohonenreprojector.h
    vtkCudaPAGMMEstimator.h
    CUDA_PAGMMestimator.h
    vtkCudaKSOMLikelihood.h
    CUDA_KSOMlikelihood.h
    vtkCudaKSOMProbability.h
    CUDA_KSOMProbability.h
    vtkCudaHierarchicalMaxFlowSegmentation.h
    vtkCudaHierarchicalMaxFlowSegmentation2.h
    vtkCudaDirectedAcyclicGraphMaxFlowSegmentation.h
    vtkCudaMaxFlowSegmentationWorker.h
    vtkCudaMaxFlowSegmentationScheduler.h
    vtkCudaMaxFlowSegmentationTask.h
    CUDA_hierarchicalmaxflow.h
    vtkCudaHierarchicalMaxFlowDecomposition.h
    CUDA_hierarchicalmaxflowdecomp.h
    vtkCudaImageAtlasLabelProbability.h
    CUDA_atlasprobability.h
    vtkCudaImageLogLikelihood.h
    CUDA_loglikelihoodterm.h
    vtkCudaImageVote.h
    CUDA_imagevote.h
    )
ELSE()
  SET(${PROJECT_NAME}_HDRS)
ENDIF()

SET ( ${PROJECT_NAME}_SRCS
  vtkCudaVoxelClassifier.cxx
  CUDA_voxelclassifier.cu
  vtkCudaCT2USSimulation.cxx
  CUDA_cttoussimulation.cu
  vtkCudaFuzzyConnectednessFilter.cxx
  CUDA_fuzzyconnectednessfilter.cu
  vtkCudaKohonenGenerator.cxx
  CUDA_kohonengenerator.cu
  vtkCudaKohonenApplication.cxx
  CUDA_kohonenapplication.cu
  vtkCudaKohonenReprojector.cxx
  CUDA_kohonenreprojector.cu
  vtkCudaPAGMMEstimator.cxx
  CUDA_PAGMMestimator.cu
  vtkCudaKSOMLikelihood.cxx
  CUDA_KSOMlikelihood.cu
  vtkCudaKSOMProbability.cxx
  CUDA_KSOMProbability.cu
  vtkCudaHierarchicalMaxFlowSegmentation.cxx
  vtkCudaHierarchicalMaxFlowSegmentation2.cxx
  vtkCudaDirectedAcyclicGraphMaxFlowSegmentation.cxx
  vtkCudaMaxFlowSegmentationWorker.cxx
  vtkCudaMaxFlowSegmentationScheduler.cxx
  vtkCudaMaxFlowSegmentationTask.cxx
  CUDA_hierarchicalmaxflow.cu
  vtkCudaHierarchicalMaxFlowDecomposition.cxx
  CUDA_hierarchicalmaxflowdecomp.cu
  vtkCudaImageAtlasLabelProbability.cxx
  CUDA_atlasprobability.cu
  vtkCudaImageLogLikelihood.cxx
  CUDA_loglikelihoodterm.cu
  vtkCudaImageVote.cxx
  CUDA_imagevote.cu
)

REMOVE_VTK_DEFINITIONS()
CUDA_ADD_LIBRARY( ${PROJECT_NAME} ${${PROJECT_NAME}_SRCS} ${${PROJECT_NAME}_HDRS})
TARGET_LINK_LIBRARIES( ${PROJECT_NAME} ${${PROJECT_NAME}_LIBS} )