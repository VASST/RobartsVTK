PROJECT(vtkRobartsCommon)

include(module.cmake)

SET ( Module_SRCS
  vtkRobartsCommon.h
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

IF( MSVC )
  SET ( ${vtk-module}_HDRS 
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
  # No idea why, but turning on optimization in robarts common in 32 bit causes
  # an internal compiler crash
  STRING(REPLACE "/O2" "/Od" CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
ENDIF()

SET(${PROJECT_NAME}_INCLUDE_DIRS ${VTK_INCLUDE_DIRS})

# -----------------------------------------------------------------
# Build the library
vtk_module_library(${PROJECT_NAME} ${Module_SRCS})
  
# -----------------------------------------------------------------
LIST(APPEND VTK_MODULES_ENABLED ${PROJECT_NAME})
SET(VTK_MODULES_ENABLED ${VTK_MODULES_ENABLED} PARENT_SCOPE)