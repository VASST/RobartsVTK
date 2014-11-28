
SET(KIT_LIBS SharedComponents vtkCudaVisualization vtkCudaImageAnalytics vtkCudaCommon vtkVolumeRendering vtkRendering vtkFiltering vtkImaging vtkWidgets vtkIO vtkCommon QVTK)

SET ( Kit_SRCS
  DUALTF_main.cpp
  DUALTF_transferFunctionWindowWidget.h
  DUALTF_transferFunctionWindowWidget.cpp
  DUALTF_HistogramHolderDefault.h
  DUALTF_HistogramHolderDefault.cpp
  DUALTF_transferFunctionDefinitionWidget.cxx
  DUALTF_transferFunctionDefinitionWidget.h
  DUALTF_segmentationWidget.cxx
  DUALTF_segmentationWidget.h
)

SET( mocHeaders  
  DUALTF_transferFunctionWindowWidget.h
  DUALTF_transferFunctionDefinitionWidget.h
  DUALTF_segmentationWidget.h
  DUALTF_HistogramHolderDefault.h
)

QT4_WRAP_CPP( Moc_SRCS ${mocHeaders} )

INCLUDE_DIRECTORIES ( ${TransferFunctionUI_INCLUDE_DIRS} )
INCLUDE_DIRECTORIES ( ${SharedComponents_SOURCE_DIR} )

CUDA_ADD_EXECUTABLE( DualTFUI ${Kit_SRCS} ${Moc_SRCS} )
TARGET_LINK_LIBRARIES( DualTFUI ${KIT_LIBS} )