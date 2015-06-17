PROJECT(ROBARTSLEGACY)

SET(KIT RobartsLegacy)
SET(UKIT ROBARTSLEGACY)
# possible fix for vtk6+
#IF(VTK_LIBRARIES)
#  SET(KIT_LIBS ${VTK_LIBRARIES} )
#ELSE()
SET(KIT_LIBS vtkHybrid)
#ENDIF()
SET(KIT_EXTRA_LIBS)

SET ( Kit_SRCS
)

#SET_SOURCE_FILES_PROPERTIES(
#)

OPTION (VTK_USE_SHAPETAPE "Enable Shape Tape" ON)
IF (VTK_USE_SHAPETAPE)
  SET ( Kit_SRCS ${Kit_SRCS}
        vtkSHAPETAPETracker.cxx
  vtkTrackerSensor.cxx
  vtkTrackerToolSensor.cxx)

  SET(KIT_EXTRA_LIBS)

  SET (KIT_EXTRA_LIBS ${KIT_EXTRA_LIBS} ShapeApi)
  INCLUDE_DIRECTORIES(${VTKSHAPEAPI_SOURCE_DIR}
                      ${VTKSHAPEAPI_BINARY_DIR})

  #ADD_SUBDIRECTORY(vtkShapeApi vtkRobartsLegacy)
  SET (KIT_EXTRA_LIBS ${KIT_EXTRA_LIBS} vtkTracking)
  SET (KIT_EXTRA_LIBS ${KIT_EXTRA_LIBS} vtkTrackingPython)
  SUBDIR_DEPENDS(vtkRobartsLegacy ShapeApi)
  ADD_SUBDIRECTORY(ShapeApi)


ENDIF (VTK_USE_SHAPETAPE)

SET(Kit_EXTRA_SRCS)
SET(Kit_EXTRA_CMDS)
SET(Kit_TCL_EXTRA_SRCS)
SET(Kit_PYTHON_EXTRA_SRCS)
SET(Kit_JAVA_EXTRA_SRCS)

# -----------------------------------------------------------------

# Include the common setting file
INCLUDE(${RobartsVTK_SOURCE_DIR}/CMake/KitCommonBlock.cmake)
