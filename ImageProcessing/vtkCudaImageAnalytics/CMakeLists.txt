PROJECT(vtkCudaImageAnalytics)

SET( Module_SRCS
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
  CUDA_cttoussimulation.cu
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

IF(BUILD_SHARED_LIBS)
  SET(GPGPU_BUILD_TYPE SHARED)
ENDIF()

REMOVE_VTK_DEFINITIONS()

#------------------------------------------------------------------
# See VTK/Accelerators/Pistons
SET(${PROJECT_NAME}_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS} ${VTK_INCLUDE_DIRS})
vtk_module_impl()

# -----------------------------------------------------------------
# Generate the cuda sources
cuda_compile(Module_CUDA_C_SRCS ${Module_CUDA_SRCS} ${GPGPU_BUILD_TYPE})

SET_SOURCE_FILES_PROPERTIES(
  ${Module_CUDA_C_SRCS} 
  vtkCudaMaxFlowSegmentationWorker.cxx  
  vtkCudaMaxFlowSegmentationScheduler.cxx 
  vtkCudaMaxFlowSegmentationTask.cxx 
  WRAP_EXCLUDE
  )
LIST(APPEND Module_SRCS ${Module_CUDA_C_SRCS})

# -----------------------------------------------------------------
# Build the library
vtk_module_library(${PROJECT_NAME} ${Module_SRCS} ${Module_CUDA_HDRS})
vtk_module_link_libraries(${PROJECT_NAME} LINK_PRIVATE ${CUDA_LIBRARIES})

# -----------------------------------------------------------------
SET_TARGET_PROPERTIES(${PROJECT_NAME} PROPERTIES module_uses_cuda ON)
LIST(APPEND VTK_MODULES_ENABLED ${PROJECT_NAME})
SET(VTK_MODULES_ENABLED ${VTK_MODULES_ENABLED} PARENT_SCOPE)