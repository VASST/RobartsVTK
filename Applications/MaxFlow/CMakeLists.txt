PROJECT( MaxFlow )

SET ( ${PROJECT_NAME}_SRCS 
  MaxFlow.cxx
)

# -----------------------------------------------------------------
# Build the library
ADD_EXECUTABLE(${PROJECT_NAME} ${${PROJECT_NAME}_SRCS} )
TARGET_LINK_LIBRARIES(${PROJECT_NAME} PUBLIC 
  vtkRenderingCore 
  vtkFiltersCore 
  vtkImagingCore 
  vtkIOImage 
  vtkCommonCore 
  vtkRobartsCommon 
  vtkCudaImageAnalytics 
  vtkCudaCommon
  vtksys
  )
  
# -----------------------------------------------------------------
# Build the GHMF_Segment executable
SET ( Module_SRCS GHMF_Segment.cxx)
ADD_EXECUTABLE(GHMFSegment ${Module_SRCS})
target_link_libraries(GHMFSegment
  vtkFiltersCore 
  vtkImagingCore 
  vtkIOCore 
  vtkIOLegacy
  vtkIOImage 
  vtkCommonCore 
  vtkCudaCommon 
  vtkCudaImageAnalytics 
  vtkRobartsCommon
  )

# -----------------------------------------------------------------
# Build the KSOM_train executable
SET ( Module_SRCS KSOM_train.cxx)
ADD_EXECUTABLE(KSOMTrain ${Module_SRCS})
target_link_libraries(KSOMTrain
  vtkFiltersCore 
  vtkImagingCore 
  vtkIOCore 
  vtkIOImage 
  vtkCommonCore 
  vtkCudaCommon 
  vtkCudaImageAnalytics 
  vtkRobartsCommon
  )

# -----------------------------------------------------------------
# Build the KSOM_apply executable
SET ( Module_SRCS KSOM_apply.cxx)
ADD_EXECUTABLE(KSOMApply ${Module_SRCS})
target_link_libraries(KSOMApply
  vtkFiltersCore 
  vtkImagingCore 
  vtkIOCore 
  vtkIOImage 
  vtkCommonCore 
  vtkCudaCommon 
  vtkCudaImageAnalytics 
  vtkRobartsCommon
  )
