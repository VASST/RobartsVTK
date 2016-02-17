PROJECT( MaxFlow )

SET(${PROJECT_NAME}_LIBS vtkFiltersCore vtkImagingCore vtkIOImage vtkCommonCore vtkCudaCommon vtkCudaImageAnalytics vtkRobartsCommon )

INCLUDE_DIRECTORIES ( ${VTK_INCLUDE_DIRS} ${vtkCudaCommon_INCLUDE_DIRS} ${vtkCudaImageAnalytics_INCLUDE_DIRS} ${vtkRobartsCommon_INCLUDE_DIRS})

CUDA_ADD_EXECUTABLE( GHMF_Segment GHMF_Segment.cxx )
TARGET_LINK_LIBRARIES( GHMF_Segment ${${PROJECT_NAME}_LIBS} )

CUDA_ADD_EXECUTABLE( KSOM_train KSOM_train.cxx )
TARGET_LINK_LIBRARIES( KSOM_train ${${PROJECT_NAME}_LIBS} )

CUDA_ADD_EXECUTABLE( KSOM_apply KSOM_apply.cxx )
TARGET_LINK_LIBRARIES( KSOM_apply ${${PROJECT_NAME}_LIBS} )