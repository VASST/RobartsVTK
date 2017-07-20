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
  vtkRobartsCommon
  )