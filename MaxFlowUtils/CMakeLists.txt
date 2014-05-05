PROJECT( MaxFlowUtils )

SET(KIT MaxFlowUtils )
SET(UKIT MaxFlowUtils )
SET(KIT_LIBS vtkFiltering vtkImaging vtkIO vtkCommon vtkCudaCommon vtkCudaImageAnalytics vtkRobartsUtilities )
SET(KIT_EXTRA_LIBS)

INCLUDE_DIRECTORIES ( ${RobartsVTK_INCLUDE_DIRS} )

CUDA_ADD_EXECUTABLE( GHMF_Segment GHMF_Segment.cxx )
TARGET_LINK_LIBRARIES( GHMF_Segment ${KIT_LIBS} )