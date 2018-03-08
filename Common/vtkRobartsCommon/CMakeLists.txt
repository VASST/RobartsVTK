PROJECT(RobartsCommon)

SET ( ${PROJECT_NAME}_SRCS
  vtkRobartsCommon.cxx
  vtkReadWriteLock.cxx
  vtkBarrierLock.cxx
  vtkImagePipe.cxx
  vtkImageSeedFilter.cxx
  vtkImageBasicAffinityFilter.cxx
  vtkImage2DHistogram.cxx
  vtkImageMultiStatistics.cxx
  vtkImageLogLikelihood.cxx
  vtkImageAtlasLabelProbability.cxx
  vtkImageDataTerm.cxx
  vtkImageVote.cxx
  vtkDiceCoefficient.cxx
  vtkMaxFlowSegmentationUtilities.cxx
  vtkHierarchicalMaxFlowSegmentation.cxx
  vtkDirectedAcyclicGraphMaxFlowSegmentation.cxx
  vtkImageEntropyPlaneSelection.cxx
  vtkRootedDirectedAcyclicGraph.cxx
  vtkRootedDirectedAcyclicGraphIterator.cxx
  vtkRootedDirectedAcyclicGraphForwardIterator.cxx
  vtkRootedDirectedAcyclicGraphBackwardIterator.cxx
  vtkImageFrangiFilter.cxx
)

IF( MSVC OR ${CMAKE_GENERATOR} MATCHES "Xcode")
  SET ( ${PROJECT_NAME}_HDRS
    vtkRobartsCommon.h
    vtkReadWriteLock.h
    vtkBarrierLock.h
    vtkImagePipe.h
    vtkImageSeedFilter.h
    vtkImageBasicAffinityFilter.h
    vtkImage2DHistogram.h
    vtkImageMultiStatistics.h
    vtkImageLogLikelihood.h
    vtkImageAtlasLabelProbability.h
    vtkImageDataTerm.h
    vtkImageVote.h
    vtkDiceCoefficient.h
    vtkMaxFlowSegmentationUtilities.h
    vtkHierarchicalMaxFlowSegmentation.h
    vtkDirectedAcyclicGraphMaxFlowSegmentation.h
    vtkImageEntropyPlaneSelection.h
    vtkRootedDirectedAcyclicGraph.h
    vtkRootedDirectedAcyclicGraphIterator.h
    vtkRootedDirectedAcyclicGraphForwardIterator.h
    vtkRootedDirectedAcyclicGraphBackwardIterator.h
    vtkImageFrangiFilter.h
  )
ENDIF()

IF(WIN32 AND "${CMAKE_GENERATOR}" MATCHES "Visual Studio" AND NOT "${CMAKE_GENERATOR}" MATCHES "x64")
  # No idea why, but turning on optimization in robarts common in 32 bit causes an internal compiler crash
  STRING(REPLACE "/O2" "/Od" CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
ENDIF()

SET(vtk${PROJECT_NAME}_INCLUDE_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
  ${VTK_INCLUDE_DIRS}
  )

# -----------------------------------------------------------------
# Build the library
ADD_LIBRARY(vtk${PROJECT_NAME} ${${PROJECT_NAME}_HDRS} ${${PROJECT_NAME}_SRCS})
target_include_directories(vtk${PROJECT_NAME} PUBLIC
  ${vtk${PROJECT_NAME}_INCLUDE_DIRS}
  )
target_link_libraries(vtk${PROJECT_NAME} PUBLIC
  vtkCommonCore
  vtkCommonSystem
  vtkParallelCore
  vtkImagingCore
  vtkFiltersParallel
  vtkFiltersCore
  )
GENERATE_EXPORT_DIRECTIVE_FILE(vtk${PROJECT_NAME})

RobartsVTKInstallLibrary(vtk${PROJECT_NAME} ${PROJECT_NAME})
