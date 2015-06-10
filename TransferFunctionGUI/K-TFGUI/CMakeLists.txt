PROJECT(K-TFGUI)

SET(KIT K-TFGUI)
SET(UKIT K-TFGUI)
# possible fix for vtk6+
#IF(VTK_LIBRARIES)
#  SET(KIT_LIBS SharedComponents vtkCudaVisualization vtkCudaImageAnalytics vtkCudaCommon ${VTK_LIBRARIES} )
#ELSE()
SET(KIT_LIBS SharedComponents vtkCudaVisualization vtkCudaImageAnalytics vtkCudaCommon vtkVolumeRendering vtkRendering vtkFiltering vtkImaging vtkWidgets vtkIO vtkCommon QVTK)
#ENDIF()
SET(KIT_EXTRA_LIBS)

SET ( Kit_SRCS
  main.cpp
  transferFunctionWindowWidget.h
  transferFunctionWindowWidget.cpp
  HistogramHolderDefault.h
  HistogramHolderDefault.cpp
  transferFunctionDefinitionWidget.cxx
  transferFunctionDefinitionWidget.h
  segmentationWidget.cxx
  segmentationWidget.h
  kFileManagementWidget.cxx
  kFileManagementWidget.h
)

SET( mocHeaders
  transferFunctionWindowWidget.h
  transferFunctionDefinitionWidget.h
  segmentationWidget.h
  HistogramHolderDefault.h
  kFileManagementWidget.h
)

QT4_WRAP_CPP( Moc_SRCS ${mocHeaders} )

INCLUDE_DIRECTORIES ( ${TransferFunctionUI_INCLUDE_DIRS} )
INCLUDE_DIRECTORIES ( ${SharedComponents_SOURCE_DIR} )

CUDA_ADD_EXECUTABLE( K-TFGUI ${Kit_SRCS} ${Moc_SRCS} )
TARGET_LINK_LIBRARIES( K-TFGUI ${KIT_LIBS} )
