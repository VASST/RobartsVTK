PROJECT(vtkRobartsCommon)

include(module.cmake)

SET(${PROJECT_NAME}_INCLUDE_DIRS ${${PROJECT_NAME}_BINARY_DIR})
SET(${PROJECT_NAME}_PYTHON_INCLUDE_DIRS ${${PROJECT_NAME}_INCLUDE_DIRS})

# Add entries to project details
SET(${PROJECT_NAME}_INCLUDE_DIRS ${${PROJECT_NAME}_SOURCE_DIR} ${${PROJECT_NAME}_BINARY_DIR} CACHE INTERNAL "" FORCE)

SET ( Module_SRCS
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

# -----------------------------------------------------------------
# Build the library
include_directories(${${PROJECT_NAME}_INCLUDE_DIRS})
vtk_module_library(${PROJECT_NAME} ${Module_SRCS})

# -----------------------------------------------------------------
# Build the python wrapping if enabled
IF( VTK_WRAP_PYTHON )
  vtk_module_load(${PROJECT_NAME})
  vtk_module_headers_load(${PROJECT_NAME})
  vtk_include_recurse(${PROJECT_NAME})
  vtk_add_python_wrapping(${PROJECT_NAME})
ENDIF()