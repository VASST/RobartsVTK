PROJECT( MaxFlowExample )

SET(KIT MaxFlowExample )
SET(UKIT MaxFlowExample )
SET(KIT_LIBS ${RobartsVTK_LIBS} vtkRendering vtkFiltering vtkImaging vtkWidgets vtkIO vtkCommon vtkRobartsUtilities)
SET(KIT_EXTRA_LIBS)

SET ( Kit_SRCS
  MaxFlowExample.cxx
)

INCLUDE_DIRECTORIES ( ${RobartsVTK_INCLUDE_DIRS} )
LINK_DIRECTORIES( ${RobartsVTK_LIBRARY_DIRS} )
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS} ${OpenCV_LIB_DIR} )

ADD_EXECUTABLE( MaxFlowExample ${Kit_SRCS} )
TARGET_LINK_LIBRARIES( MaxFlowExample ${KIT_LIBS} )