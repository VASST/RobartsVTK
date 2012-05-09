PROJECT(vtkCudaCommon)


SET(KIT vtkCudaCommon)
SET(UKIT vtkCudaCommon)
SET(KIT_LIBS vtkCommon cudart)
SET (KIT_EXTRA_LIBS ${KIT_EXTRA_LIBS} cuda cudart cudartemu cufft cufftemu OpenCL cublas cublasemu)

SET ( Kit_SRCS
  vtkCudaMemoryTexture.cxx
  vtkCudaMemoryTexture.h
)

CUDA_ADD_LIBRARY( vtkCudaCommon ${Kit_SRCS} )