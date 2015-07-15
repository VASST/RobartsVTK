PROJECT( Cuda2DTFVolumeRendering )

SET(KIT Cuda2DTFVolumeRendering )
SET(UKIT Cuda2DTFVolumeRendering )

IF(VTK_LIBRARIES)
  SET(KIT_LIBS vtkCudaCommon vtkCudaVisualization ${VTK_LIBRARIES} )
ELSE()
  SET(KIT_LIBS vtkVolumeRendering vtkRendering vtkFiltering vtkImaging vtkWidgets vtkIO vtkCommon vtkCudaCommon vtkCudaVisualization )
ENDIF()
SET(KIT_EXTRA_LIBS)

SET ( Kit_SRCS
  Cuda2DTFVolumeRendering.cxx
)

INCLUDE_DIRECTORIES ( ${RobartsVTK_INCLUDE_DIRS} )

CUDA_ADD_EXECUTABLE( Cuda2DTFVolumeRendering ${Kit_SRCS} )
TARGET_LINK_LIBRARIES( Cuda2DTFVolumeRendering ${KIT_LIBS} )
