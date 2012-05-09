PROJECT(ROBARTSHAPTICS)

SET(KIT RobartsHaptics)
SET(UKIT ROBARTSHAPTICS)
SET(KIT_LIBS vtkHybrid)
SET(KIT_EXTRA_LIBS)


SET ( Kit_SRCS
      	vtkHapticsDevice.cxx
	vtkForceFeedback.cxx
	vtkHapticForce.cxx
	vtkImplicitSplineForce.cxx
	vtkPolydataForce.cxx
)

SET_SOURCE_FILES_PROPERTIES(
vtkHapticsDevice
vtkForceFeedback
ABSTRACT
)

OPTION (RobartsVTK_USE_FREEDOM6S "Enable Freedom6S Device" OFF)
IF (RobartsVTK_USE_FREEDOM6S)
  SET ( Kit_SRCS ${Kit_SRCS}
        vtkHapticsFreedom6S.cxx
	)
  INCLUDE_DIRECTORIES(${VTK_SOURCE_DIR}
                      ${VTK_BINARY_DIR})

  SET (KIT_EXTRA_LIBS ${KIT_EXTRA_LIBS} vtkTracking)
  SET (KIT_EXTRA_LIBS ${KIT_EXTRA_LIBS} vtkTrackingPython)
  SET (KIT_EXTRA_LIBS ${KIT_EXTRA_LIBS} ${RobartsVTK_SOURCE_DIR}/vtkRobartsHaptics/Freedom6S/freedom6s)

ENDIF (RobartsVTK_USE_FREEDOM6S)


SET(Kit_EXTRA_SRCS)
SET(Kit_EXTRA_CMDS)
SET(Kit_TCL_EXTRA_SRCS)
SET(Kit_PYTHON_EXTRA_SRCS)
SET(Kit_JAVA_EXTRA_SRCS)

# -----------------------------------------------------------------

# Include the common setting file
INCLUDE(${RobartsVTK_SOURCE_DIR}/CMake/KitCommonBlock.cmake)
