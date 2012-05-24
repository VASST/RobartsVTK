PROJECT( DirectShowVideoCapture )

SET(KIT DirectShowVideoCapture )
SET(UKIT DirectShowVideoCapture )
SET(KIT_LIBS ${RobartsVTK_LIBS} vtkRendering vtkFiltering vtkImaging vtkWidgets vtkIO vtkCommon vtkRobartsVideo )
SET(KIT_EXTRA_LIBS)

SET ( Kit_SRCS
  DirectShowVideoCapture.cxx
)

INCLUDE_DIRECTORIES ( ${RobartsVTK_INCLUDE_DIRS} )
LINK_DIRECTORIES( ${RobartsVTK_LIBRARY_DIRS} )
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS} ${OpenCV_LIB_DIR} )

CUDA_ADD_EXECUTABLE( DirectShowVideoCapture ${Kit_SRCS} )
TARGET_LINK_LIBRARIES( DirectShowVideoCapture ${KIT_LIBS} )