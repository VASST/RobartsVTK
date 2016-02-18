PROJECT(vtkCudaImageAnalytics)

INCLUDE(module.cmake)

# Add entries to project details
SET(${PROJECT_NAME}_INCLUDE_DIRS ${${PROJECT_NAME}_SOURCE_DIR} ${${PROJECT_NAME}_BINARY_DIR} CACHE INTERNAL "" FORCE)

IF( MSVC )
  SET(Module_HDRS 
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
  SET(Module_HDRS)
ENDIF()

SET ( Module_SRCS 
  vtkCudaVoxelClassifier.cxx
  vtkCudaCT2USSimulation.cxx
  vtkCudaFuzzyConnectednessFilter.cxx
  vtkCudaKohonenGenerator.cxx
  vtkCudaKohonenApplication.cxx
  vtkCudaKohonenReprojector.cxx
  vtkCudaPAGMMEstimator.cxx
  vtkCudaKSOMLikelihood.cxx
  vtkCudaKSOMProbability.cxx
  vtkCudaHierarchicalMaxFlowSegmentation.cxx
  vtkCudaHierarchicalMaxFlowSegmentation2.cxx
  vtkCudaDirectedAcyclicGraphMaxFlowSegmentation.cxx
  vtkCudaMaxFlowSegmentationWorker.cxx
  vtkCudaMaxFlowSegmentationScheduler.cxx
  vtkCudaMaxFlowSegmentationTask.cxx
  vtkCudaHierarchicalMaxFlowDecomposition.cxx
  vtkCudaImageAtlasLabelProbability.cxx
  vtkCudaImageLogLikelihood.cxx
  vtkCudaImageVote.cxx
)

SET( Module_CUDA_SRCS
  CUDA_voxelclassifier.cu
  CUDA_fuzzyconnectednessfilter.cu
  CUDA_kohonengenerator.cu
  CUDA_kohonenapplication.cu
  CUDA_kohonenreprojector.cu
  CUDA_PAGMMestimator.cu
  CUDA_KSOMlikelihood.cu
  CUDA_KSOMProbability.cu
  CUDA_hierarchicalmaxflow.cu
  CUDA_hierarchicalmaxflowdecomp.cu
  CUDA_atlasprobability.cu
  CUDA_loglikelihoodterm.cu
  CUDA_imagevote.cu
  )

SET_SOURCE_FILES_PROPERTIES(
  ${Module_CUDA_SRCS}
  WRAP_EXCLUDE
  )

# -----------------------------------------------------------------
# Build the library
include_directories(${${PROJECT_NAME}_INCLUDE_DIRS})
cuda_vtk_add_library( ${PROJECT_NAME} ${Module_CUDA_SRCS} ${Module_SRCS} ${Module_HDRS} )

# -----------------------------------------------------------------
# Build the python wrapping if enabled
IF( VTK_WRAP_PYTHON AND RobartsVTK_CUDA_WRAP_PYTHON )
  vtk_module_load(${PROJECT_NAME})
  vtk_module_headers_load(${PROJECT_NAME})
  vtk_include_recurse(${PROJECT_NAME})
  vtk_add_python_wrapping(${PROJECT_NAME})
ENDIF()