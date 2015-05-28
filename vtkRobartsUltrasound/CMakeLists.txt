PROJECT(ROBARTSULTRASOUND)

SET(KIT RobartsUltrasound)
SET(UKIT ROBARTSULTRASOUND)
if(VTK_LIBRARIES)
  SET(KIT_LIBS ${VTK_LIBRARIES})
else()
  SET(KIT_LIBS vtkFiltering vtkHybrid)
endif()
SET(KIT_EXTRA_LIBS)

SET (Kit_SRCS ${Kit_SRCS} 
  vtkCT2USSimulation.cxx vtkCT2USSimulation.h
)

OPTION (VTK_USE_ULTRASOUNDTESTING "Enable Ultrasound Testing Classes" OFF)
IF (VTK_USE_ULTRASOUNDTESTING)
  SET (Kit_SRCS ${Kit_SRCS} vtkUltrasoundSphereDetectionRayCaster.cxx )
ENDIF (VTK_USE_ULTRASOUNDTESTING)

IF (VTK_USE_ROBARTSECG)
  SET ( KIT_LIBS ${KIT_LIBS} vtkRobartsECG vtkRobartsVideo vtkAtamaiTracking )
  SET ( Kit_SRCS ${Kit_SRCS}
    vtkFreehandUltrasound2.cxx
  )

  INCLUDE_DIRECTORIES(${ATAMAITRACKING_SOURCE_DIR}
                      ${ATAMAITRACKING_BINARY_DIR}
                      ${ROBARTSECG_SOURCE_DIR}
                      ${ROBARTSECG_BINARY_DIR}
                      ${ROBARTSVIDEO_BINARY_DIR}
                      ${ROBARTSVIDEO_SOURCE_DIR} )

ENDIF(VTK_USE_ROBARTSECG)

SET(Kit_EXTRA_SRCS)
SET(Kit_EXTRA_CMDS)
SET(Kit_TCL_EXTRA_SRCS)
SET(Kit_PYTHON_EXTRA_SRCS)
SET(Kit_JAVA_EXTRA_SRCS)

# -----------------------------------------------------------------

# Include the common setting file
INCLUDE(${RobartsVTK_SOURCE_DIR}/CMake/KitCommonBlock.cmake)
