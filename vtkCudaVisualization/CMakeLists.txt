PROJECT(vtkCudaVisualization)

INCLUDE_DIRECTORIES(${vtkCudaCommon_SOURCE_DIR})

SET(KIT vtkCudaVisualization)
SET(UKIT vtkCudaVisualization)
SET(KIT_LIBS vtkVolumeRendering vtkRendering vtkFiltering vtkImaging vtkWidgets vtkIO vtkCommon vtkCudaCommon cudart)
SET(KIT_EXTRA_LIBS ${KIT_EXTRA_LIBS} cuda cudart cudartemu cufft cufftemu OpenCL cublas cublasemu)

SET ( Kit_SRCS
  vtkCudaVolumeMapper.cxx
  vtkCudaVolumeMapper.h
  vtkCuda1DVolumeMapper.cxx
  vtkCuda1DVolumeMapper.h
  vtkCuda2DVolumeMapper.cxx
  vtkCuda2DVolumeMapper.h
  vtkCudaRendererInformationHandler.cxx
  vtkCudaRendererInformationHandler.h
  CUDA_containerRendererInformation.h
  vtkCudaVolumeInformationHandler.cxx
  vtkCudaVolumeInformationHandler.h
  CUDA_containerVolumeInformation.h
  vtkCudaOutputImageInformationHandler.cxx
  vtkCudaOutputImageInformationHandler.h
  CUDA_containerOutputImageInformation.h
  vtkCuda1DTransferFunctionInformationHandler.cxx
  vtkCuda1DTransferFunctionInformationHandler.h
  vtkCuda2DTransferFunctionInformationHandler.cxx
  vtkCuda2DTransferFunctionInformationHandler.h
  CUDA_container1DTransferFunctionInformation.h
  CUDA_container2DTransferFunctionInformation.h
  CUDA_vtkCudaVolumeMapper_renderAlgo.cuh
  CUDA_vtkCudaVolumeMapper_renderAlgo.h
  CUDA_vtkCuda1DVolumeMapper_renderAlgo.cu
  CUDA_vtkCuda1DVolumeMapper_renderAlgo.h
  CUDA_vtkCuda2DVolumeMapper_renderAlgo.cu
  CUDA_vtkCuda2DVolumeMapper_renderAlgo.h
  vtkCuda2DTransferFunction.cxx
  vtkCuda2DTransferFunction.h
  vtkCudaFunctionObject.cxx
  vtkCudaFunctionObject.h
  vtkCudaFunctionPolygon.cxx
  vtkCudaFunctionPolygon.h
  vtkCudaFunctionPolygonReader.cxx
  vtkCudaFunctionPolygonReader.h
  vtkCudaFunctionPolygonWriter.cxx
  vtkCudaFunctionPolygonWriter.h
)

CUDA_ADD_LIBRARY( vtkCudaVisualization ${Kit_SRCS} )