PROJECT(vtkRobartsUtilities)

SET(KIT vtkRobartsUtilities)
SET(UKIT vtkRobartsUtilities)
SET(KIT_LIBS vtkCommon vtkFiltering)
SET (KIT_EXTRA_LIBS ${KIT_EXTRA_LIBS} vtkCommon vtkParallel vtkFiltering)

SET ( Kit_SRCS
  vtkReadWriteLock.h vtkReadWriteLock.cxx
  vtkBarrierLock.h vtkBarrierLock.cxx
  
  vtkImagePipe.h vtkImagePipe.cxx
  vtkImageSeedFilter.h vtkImageSeedFilter.cxx
  vtkImageBasicAffinityFilter.h vtkImageBasicAffinityFilter.cxx

  vtkImage2DHistogram.h vtkImage2DHistogram.cxx
  vtkImageMultiStatistics.h vtkImageMultiStatistics.cxx

  vtkImageLogLikelihood.h vtkImageLogLikelihood.cxx
  vtkImageAtlasLabelProbability.h vtkImageAtlasLabelProbability.cxx
  vtkImageDataTerm.h vtkImageDataTerm.cxx
  vtkImageVote.h vtkImageVote.cxx
  vtkDiceCoefficient.h vtkDiceCoefficient.cxx
  
  vtkHierarchicalMaxFlowSegmentation.h vtkHierarchicalMaxFlowSegmentation.cxx
  vtkImageEntropyPlaneSelection.h vtkImageEntropyPlaneSelection.cxx
  
  vtkRootedDirectedAcyclicGraph.h vtkRootedDirectedAcyclicGraph.cxx
  vtkRootedDirectedAcyclicGraphIterator.h vtkRootedDirectedAcyclicGraphIterator.cxx
  vtkRootedDirectedAcyclicGraphForwardIterator.h vtkRootedDirectedAcyclicGraphForwardIterator.cxx
)

ADD_LIBRARY( vtkRobartsUtilities ${Kit_SRCS} )
