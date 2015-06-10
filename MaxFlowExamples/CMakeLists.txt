PROJECT( MaxFlowExample )

SET(KIT MaxFlowExample )
SET(UKIT MaxFlowExample )
# possible fix for vtk6+
#IF(VTK_LIBRARIES)
#  SET(KIT_LIBS ${RobartsVTK_LIBS} vtkRobartsUtilities vtkCudaImageAnalytics vtkCudaCommon ${VTK_LIBRARIES} )
#ELSE()
SET(KIT_LIBS ${RobartsVTK_LIBS} vtkRendering vtkFiltering vtkImaging vtkWidgets vtkIO vtkCommon vtkRobartsUtilities vtkCudaImageAnalytics vtkCudaCommon)
#ENDIF()
SET(KIT_EXTRA_LIBS)

SET ( Kit_SRCS
  MaxFlowExample.cxx
)

INCLUDE_DIRECTORIES ( ${RobartsVTK_INCLUDE_DIRS} )
LINK_DIRECTORIES( ${RobartsVTK_LIBRARY_DIRS} )
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS} ${OpenCV_LIB_DIR} )

ADD_EXECUTABLE( MaxFlowExample ${Kit_SRCS} )
TARGET_LINK_LIBRARIES( MaxFlowExample ${KIT_LIBS} )
