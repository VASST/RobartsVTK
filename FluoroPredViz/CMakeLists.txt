PROJECT( FluoroPredViz )
INCLUDE( ${QT_USE_FILE} )

SET(KIT FluoroPredViz )
SET(UKIT FluoroPredViz )
SET(KIT_LIBS vtkVolumeRendering vtkRendering vtkFiltering vtkImaging vtkWidgets vtkIO vtkCommon vtkCudaCommon vtkCudaVisualization vtkRobartsUtilities QVTK)
SET(KIT_EXTRA_LIBS)

SET ( Kit_SRCS
  main.cxx
  FluoroPredViz.h
  FluoroPredViz.cxx
  ResizableQVTKWidget.h
  ResizableQVTKWidget.cxx
)

SET( mocHeaders  
  FluoroPredViz.h
)
QT_WRAP_CPP( FluoroPredViz Moc_SRCS ${mocHeaders} )

INCLUDE_DIRECTORIES ( ${RobartsVTK_INCLUDE_DIRS} )

CUDA_ADD_EXECUTABLE( FluoroPredViz ${Kit_SRCS} ${Moc_SRCS} )
TARGET_LINK_LIBRARIES( FluoroPredViz ${KIT_LIBS} )