PROJECT(ROBARTSCUDA)

SET(KIT RobartsCuda)
SET(UKIT ROBARTSCUDA)
#SET(KIT_LIBS vtkHybrid)
SET(KIT_EXTRA_LIBS)


SET(CUDA_DIR "C:/CUDA" )
SET(CUDA_LIB_PATH "${CUDA_DIR}/lib" )
SET(CUDA_INC_PATH "${CUDA_DIR}/include" )
SET(CUDA_BIN_PATH "${CUDA_DIR}/bin" )

#INCLUDE_REGULAR_EXPRESSION("^(vtkCuda).*$")
INCLUDE_DIRECTORIES(${CUDA_INC_PATH})

SET ( Kit_SRCS
	vtkCuda2DTransferClassificationFunction.cxx
	vtkCudaFunctionObject.cxx
	vtkCudaFunctionSquare.cxx
	vtkCudaMemoryTexture.cxx
	vtkCudaRendererInformationHandler.cxx
	vtkCudaSegmentor.cxx
	vtkCudaVolumeInformationHandler.cxx
	vtkCudaVolumeMapper.cxx
)

INCLUDE(FindCuda.cmake)

# LIBS
SET (KIT_EXTRA_LIBS ${KIT_EXTRA_LIBS}	cuda cudart cudartemu cufft cufftemu OpenCL cublas cublasemu)

# DLLS
#TARGET_LINK_LIBRARIES(${CUDA_BIN_PATH}/NAME_OF_FILE.dll)
#  TARGET_LINK_LIBRARIES( vtkRobartsCuda cuda cudart cudartemu cufft cufftemu OpenCL cublas cublasemu)


SET(Kit_EXTRA_SRCS)
SET(Kit_EXTRA_CMDS)
SET(Kit_TCL_EXTRA_SRCS)
SET(Kit_PYTHON_EXTRA_SRCS)
SET(Kit_JAVA_EXTRA_SRCS)

# -----------------------------------------------------------------

# Include the common setting file
INCLUDE(${RobartsVTK_SOURCE_DIR}/CMake/KitCommonBlock.cmake)