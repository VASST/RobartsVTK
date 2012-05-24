PROJECT(vtkRobartsUtilities)

SET(KIT vtkRobartsUtilities)
SET(UKIT vtkRobartsUtilities)
SET(KIT_LIBS vtkCommon vtkFiltering)
SET (KIT_EXTRA_LIBS ${KIT_EXTRA_LIBS} vtkCommon vtkParallel vtkFiltering)

SET ( Kit_SRCS
  vtkImagePipe.h vtkImagePipe.cxx
  vtkReadWriteLock.h vtkReadWriteLock.cxx
)

ADD_LIBRARY( vtkRobartsUtilities ${Kit_SRCS} )