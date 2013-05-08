PROJECT(vtkCudaVisualization)

INCLUDE_DIRECTORIES(${vtkCudaCommon_SOURCE_DIR})

SET(KIT vtkCudaVisualization)
SET(UKIT vtkCudaVisualization)
SET(KIT_LIBS vtkVolumeRendering vtkRendering vtkFiltering vtkImaging vtkWidgets vtkIO vtkCommon vtkCudaCommon cudart)
SET(KIT_EXTRA_LIBS ${KIT_EXTRA_LIBS} cuda cudart cudartemu cufft cufftemu OpenCL cublas cublasemu)

SET ( Kit_SRCS

  #Volume rendering base classes
  vtkCudaVolumeMapper.cxx
  vtkCudaVolumeMapper.h
  vtkCudaRendererInformationHandler.cxx
  vtkCudaRendererInformationHandler.h
  vtkCudaVolumeInformationHandler.cxx
  vtkCudaVolumeInformationHandler.h
  vtkCudaOutputImageInformationHandler.cxx
  vtkCudaOutputImageInformationHandler.h
  CUDA_containerRendererInformation.h
  CUDA_containerVolumeInformation.h
  CUDA_containerOutputImageInformation.h
  CUDA_vtkCudaVolumeMapper_renderAlgo.cu
  CUDA_vtkCudaVolumeMapper_renderAlgo.h

  #1D Transfer function rendering subclass
  vtkCuda1DVolumeMapper.cxx
  vtkCuda1DVolumeMapper.h
  vtkCuda1DTransferFunctionInformationHandler.cxx
  vtkCuda1DTransferFunctionInformationHandler.h
  CUDA_container1DTransferFunctionInformation.h
  CUDA_vtkCuda1DVolumeMapper_renderAlgo.cuh
  CUDA_vtkCuda1DVolumeMapper_renderAlgo.h

  #2D Transfer function rendering subclass
  vtkCuda2DVolumeMapper.cxx
  vtkCuda2DVolumeMapper.h
  vtkCuda2DTransferFunctionInformationHandler.cxx
  vtkCuda2DTransferFunctionInformationHandler.h
  CUDA_container2DTransferFunctionInformation.h
  CUDA_vtkCuda2DVolumeMapper_renderAlgo.cuh
  CUDA_vtkCuda2DVolumeMapper_renderAlgo.h

  #2D transfer function rendering with inclusion/exclusion logic with slicing planes
  vtkCuda2DInExLogicVolumeMapper.cxx
  vtkCuda2DInExLogicVolumeMapper.h
  CUDA_vtkCuda2DInExLogicVolumeMapper_renderAlgo.cuh
  CUDA_vtkCuda2DInExLogicVolumeMapper_renderAlgo.h
  CUDA_container2DInExTransferFunctionInformation.h
  vtkCuda2DInExLogicTransferFunctionInformationHandler.cxx
  vtkCuda2DInExLogicTransferFunctionInformationHandler.h

  #Dual image rendering subclass
  vtkCudaDualImageVolumeMapper.cxx
  vtkCudaDualImageVolumeMapper.h
  vtkCudaDualImageTransferFunctionInformationHandler.cxx
  vtkCudaDualImageTransferFunctionInformationHandler.h
  CUDA_containerDualImageTransferFunctionInformation.h
  CUDA_vtkCudaDualImageVolumeMapper_renderAlgo.cuh
  CUDA_vtkCudaDualImageVolumeMapper_renderAlgo.h
)

CUDA_ADD_LIBRARY( vtkCudaVisualization ${Kit_SRCS} )