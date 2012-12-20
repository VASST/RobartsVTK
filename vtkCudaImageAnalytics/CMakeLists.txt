PROJECT(vtkCudaImageAnalytics)

INCLUDE_DIRECTORIES(${vtkCudaCommon_SOURCE_DIR})

SET(KIT vtkCudaImageAnalytics)
SET(UKIT vtkCudaImageAnalytics)
SET(KIT_LIBS vtkCommon vtkFiltering vtkCudaCommon cudart)
SET (KIT_EXTRA_LIBS ${KIT_EXTRA_LIBS} cuda cudart cudartemu cufft cufftemu OpenCL cublas cublasemu)

SET ( Kit_SRCS

  #Voxel classifier
  vtkCudaVoxelClassifier.h
  vtkCudaVoxelClassifier.cxx
  CUDA_voxelclassifier.h
  CUDA_voxelclassifier.cu
  
  #CT to US simulation
  vtkCudaCT2USSimulation.h
  vtkCudaCT2USSimulation.cxx
  CUDA_cttoussimulation.h
  CUDA_cttoussimulation.cu

  #Fuzzy Connectedness filter
  vtkCudaFuzzyConnectednessFilter.h
  vtkCudaFuzzyConnectednessFilter.cxx
  CUDA_fuzzyconnectednessfilter.h
  CUDA_fuzzyconnectednessfilter.cu

  #Kohonen self-organizing map
  vtkCudaKohonenGenerator.h
  vtkCudaKohonenGenerator.cxx
  CUDA_kohonengenerator.h
  CUDA_kohonengenerator.cu

  #Custom Kohonen SOM applier and re-projector
  vtkCudaKohonenApplication.h
  vtkCudaKohonenApplication.cxx
  CUDA_kohonenapplication.h
  CUDA_kohonenapplication.cu
  vtkCudaKohonenReprojector.h
  vtkCudaKohonenReprojector.cxx
  CUDA_kohonenreprojector.h
  CUDA_kohonenreprojector.cu

)

CUDA_ADD_LIBRARY( vtkCudaImageAnalytics ${Kit_SRCS} )