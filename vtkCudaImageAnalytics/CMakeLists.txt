PROJECT(vtkCudaImageAnalytics)

INCLUDE_DIRECTORIES(${vtkCudaCommon_SOURCE_DIR})

SET(KIT vtkCudaImageAnalytics)
SET(UKIT vtkCudaImageAnalytics)
SET(KIT_LIBS vtkCommon vtkFiltering vtkCudaCommon cudart)
SET (KIT_EXTRA_LIBS ${KIT_EXTRA_LIBS} cuda cudart cudartemu cufft cufftemu OpenCL cublas cublasemu)

SET ( Kit_SRCS
  
  #CT to US simulation
  vtkCudaCT2USSimulation.h
  vtkCudaCT2USSimulation.cxx
  CUDA_cttoussimulation.h
  CUDA_cttoussimulation.cu

  #Kohonen self-organizing map
  vtkCudaKohonenGenerator.h
  vtkCudaKohonenGenerator.cxx
  CUDA_kohonengenerator.h
  CUDA_kohonengenerator.cu

)

CUDA_ADD_LIBRARY( vtkCudaImageAnalytics ${Kit_SRCS} )