PROJECT(CudaImageAnalytics)

SET( ${PROJECT_NAME}_SRCS
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

IF( MSVC OR ${CMAKE_GENERATOR} MATCHES "Xcode")
  SET ( ${PROJECT_NAME}_HDRS 
    vtkCudaVoxelClassifier.h
    vtkCudaCT2USSimulation.h
    vtkCudaFuzzyConnectednessFilter.h
    vtkCudaKohonenGenerator.h
    vtkCudaKohonenApplication.h
    vtkCudaKohonenReprojector.h
    vtkCudaPAGMMEstimator.h
    vtkCudaKSOMLikelihood.h
    vtkCudaKSOMProbability.h
    vtkCudaHierarchicalMaxFlowSegmentation.h
    vtkCudaHierarchicalMaxFlowSegmentation2.h
    vtkCudaDirectedAcyclicGraphMaxFlowSegmentation.h
    vtkCudaMaxFlowSegmentationWorker.h
    vtkCudaMaxFlowSegmentationScheduler.h
    vtkCudaMaxFlowSegmentationTask.h
    vtkCudaHierarchicalMaxFlowDecomposition.h
    vtkCudaImageAtlasLabelProbability.h
    vtkCudaImageLogLikelihood.h
    vtkCudaImageVote.h
  )
ENDIF()

SET( ${PROJECT_NAME}_CUDA_SRCS
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

# -----------------------------------------------------------------
# Generate the cuda sources
foreach(dir ${vtkCudaCommon_INCLUDE_DIRS})
  LIST(APPEND CUDA_NVCC_FLAGS -I"${dir}")
endforeach()
foreach(dir ${vtkCudaCommonStatic_INCLUDE_DIRS})
  LIST(APPEND CUDA_NVCC_FLAGS -I"${dir}")
endforeach()
LIST(REMOVE_DUPLICATES CUDA_NVCC_FLAGS)
cuda_compile(${PROJECT_NAME}_CUDA_C_SRCS ${${PROJECT_NAME}_CUDA_SRCS} ${GPGPU_BUILD_TYPE})
LIST(APPEND ${PROJECT_NAME}_SRCS ${${PROJECT_NAME}_CUDA_C_SRCS})

# -----------------------------------------------------------------
# Build the library
ADD_LIBRARY(vtk${PROJECT_NAME} ${${PROJECT_NAME}_HDRS} ${${PROJECT_NAME}_SRCS})
target_include_directories(vtk${PROJECT_NAME} PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
  ${VTK_INCLUDE_DIRS} 
  ${CUDA_INCLUDE_DIRS}
  )
target_link_libraries(vtk${PROJECT_NAME} PUBLIC 
  ${CUDA_LIBRARIES}
  vtkRobartsCommon
  vtkCudaCommon
  vtkCudaCommonStatic
  vtkFiltersCore
  vtkCommonCore
  )
GENERATE_EXPORT_DIRECTIVE_FILE(vtk${PROJECT_NAME})

RobartsVTKInstallLibrary(vtk${PROJECT_NAME} ${PROJECT_NAME})
