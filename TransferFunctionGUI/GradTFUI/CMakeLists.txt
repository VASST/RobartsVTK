PROJECT(GradTFUI)

SET(KIT GradTFUI)
SET(UKIT GradTFUI)
SET(KIT_LIBS SharedComponents vtkCudaVisualization vtkCudaImageAnalytics vtkCudaCommon vtkVolumeRendering vtkRendering vtkFiltering vtkImaging vtkWidgets vtkIO vtkCommon QVTK)
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
)

SET( mocHeaders  
  transferFunctionWindowWidget.h
  transferFunctionDefinitionWidget.h
  segmentationWidget.h
  HistogramHolderDefault.h
)

QT4_WRAP_CPP( Moc_SRCS ${mocHeaders} )

INCLUDE_DIRECTORIES ( ${TransferFunctionUI_INCLUDE_DIRS} )
INCLUDE_DIRECTORIES ( ${SharedComponents_SOURCE_DIR} )

CUDA_ADD_EXECUTABLE( GradTFUI ${Kit_SRCS} ${Moc_SRCS} )
TARGET_LINK_LIBRARIES( GradTFUI ${KIT_LIBS} )