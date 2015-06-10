PROJECT(vtkCudaCommon)


SET(KIT vtkCudaCommon)
SET(UKIT vtkCudaCommon)
# possible fix for vtk6+
#IF(VTK_LIBRARIES)
#  SET(KIT_LIBS cudart ${VTK_LIBRARIES} )
#ELSE()
SET(KIT_LIBS vtkCommon cudart)
#ENDIF()
SET (KIT_EXTRA_LIBS ${KIT_EXTRA_LIBS} cuda cudart cudartemu cufft cufftemu OpenCL cublas cublasemu)

SET ( Kit_SRCS

  #CUDA card management
  vtkCudaObject.h vtkCudaObject.cxx
  vtkCudaDeviceManager.h vtkCudaDeviceManager.cxx

  vtkCudaMemoryTexture.h vtkCudaMemoryTexture.cxx

  #very common kernels
  CUDA_commonKernels.h
  CUDA_commonKernels.cu

  #2D Transfer function definition
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

CUDA_ADD_LIBRARY( vtkCudaCommon ${Kit_SRCS} )
