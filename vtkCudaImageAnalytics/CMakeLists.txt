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

  #Custom PAGMM estimators
  vtkCudaPAGMMEstimator.h
  vtkCudaPAGMMEstimator.cxx
  CUDA_PAGMMestimator.h
  CUDA_PAGMMestimator.cu

  #Accelerated hierarchical max flow algorithm
  vtkCudaHierarchicalMaxFlowSegmentation.h
  vtkCudaHierarchicalMaxFlowSegmentation.cxx
  vtkCudaHierarchicalMaxFlowSegmentation2.h
  vtkCudaHierarchicalMaxFlowSegmentation2.cxx
  vtkCudaHierarchicalMaxFlowSegmentation2Worker.cxx
  vtkCudaHierarchicalMaxFlowSegmentation2Task.h
  CUDA_hierarchicalmaxflow.h
  CUDA_hierarchicalmaxflow.cu

  #Accelerated atlas-location data term
  vtkCudaImageAtlasLabelProbability.h
  vtkCudaImageAtlasLabelProbability.cxx
  CUDA_atlasprobability.h
  CUDA_atlasprobability.cu

  #Accelerated log likelihood data term
  vtkCudaImageLogLikelihood.h
  vtkCudaImageLogLikelihood.cxx
  CUDA_loglikelihoodterm.h
  CUDA_loglikelihoodterm.cu

  #Accelerated image vote
  vtkCudaImageVote.h
  vtkCudaImageVote.cxx
  CUDA_imagevote.h
  CUDA_imagevote.cu

)

CUDA_ADD_LIBRARY( vtkCudaImageAnalytics ${Kit_SRCS} )