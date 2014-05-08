PROJECT( MaxFlowUtils )

SET(KIT MaxFlowUtils )
SET(UKIT MaxFlowUtils )
SET(KIT_LIBS vtkFiltering vtkImaging vtkIO vtkCommon vtkCudaCommon vtkCudaImageAnalytics vtkRobartsUtilities )
SET(KIT_EXTRA_LIBS)

INCLUDE_DIRECTORIES ( ${RobartsVTK_INCLUDE_DIRS} )

CUDA_ADD_EXECUTABLE( GHMF_Segment GHMF_Segment.cxx )
TARGET_LINK_LIBRARIES( GHMF_Segment ${KIT_LIBS} )

CUDA_ADD_EXECUTABLE( KSOM_train KSOM_train.cxx )
TARGET_LINK_LIBRARIES( KSOM_train ${KIT_LIBS} )

CUDA_ADD_EXECUTABLE( KSOM_apply KSOM_apply.cxx )
TARGET_LINK_LIBRARIES( KSOM_apply ${KIT_LIBS} )