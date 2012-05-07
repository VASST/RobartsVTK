PROJECT(vtkCudaCommon)

SET(CUDA_DIR "C:/CUDA" )
SET(CUDA_LIB_PATH "${CUDA_DIR}/lib" )
SET(CUDA_INC_PATH "${CUDA_DIR}/include" )
SET(CUDA_BIN_PATH "${CUDA_DIR}/bin" )

INCLUDE_DIRECTORIES(${CUDA_INC_PATH})

SET(KIT vtkCudaCommon)
SET(UKIT vtkCudaCommon)
SET(KIT_LIBS vtkCommon cudart)
SET (KIT_EXTRA_LIBS ${KIT_EXTRA_LIBS} cuda cudart cudartemu cufft cufftemu OpenCL cublas cublasemu)

SET ( Kit_SRCS
  vtkCudaMemoryTexture.cxx
  vtkCudaMemoryTexture.h
)

CUDA_ADD_LIBRARY( vtkCudaCommon ${Kit_SRCS} )