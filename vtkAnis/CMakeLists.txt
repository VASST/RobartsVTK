PROJECT(Anis)

SET(KIT Anis)
SET(UKIT Anis)
SET(KIT_LIBS vtkHybrid)
SET(KIT_EXTRA_LIBS glu32)

SET(KIT_TCL_LIBS vtkHybridTCL vtkAtamaiTrackingTCL ${KIT_LIBS})
SET(KIT_JAVA_LIBS vtkHybridJava vtkAtamaiTrackingJava ${KIT_LIBS})
SET(KIT_LIBS vtkHybrid vtkAtamaiTracking ${KIT_LIBS})


SET ( Kit_SRCS
  #vtkDiscreteDynamicContour.cxx
  #vtkFastRBFMesher.cxx
  vtkVideoCalibration.cxx
  vtkStereoCamera.cxx
)



SET(GLEW_DIR ${RobartsVTK_SOURCE_DIR}/vtkAnis/glew)
SET(MATLAB_DIR c:/MATLAB6p5/extern)
SET(MATLAB_LIB ${MATLAB_DIR}/lib/win32/microsoft/msvc60 )
SET(MIL_DIR "C:/Program Files/Matrox Imaging/Mil" )

INCLUDE_DIRECTORIES(${RobartsVTK_SOURCE_DIR}/vtkAnis/OpenCV/cv/include
        ${RobartsVTK_SOURCE_DIR}/vtkAnis/OpenCV/cxcore/include
        ${RobartsVTK_SOURCE_DIR}/vtkAnis/OpenCV/otherlibs/highgui
        ${GLEW_DIR}/include 
        ${MATLAB_DIR}/include
        ${MIL_DIR}/include

)


SET (KIT_EXTRA_LIBS ${KIT_EXTRA_LIBS} ${RobartsVTK_SOURCE_DIR}/vtkAnis/OpenCV/lib/cv.lib)
SET (KIT_EXTRA_LIBS ${KIT_EXTRA_LIBS} ${RobartsVTK_SOURCE_DIR}/vtkAnis/OpenCV/lib/cxcore.lib)
SET (KIT_EXTRA_LIBS ${KIT_EXTRA_LIBS} ${GLEW_DIR}/lib/glew32.lib)
SET (KIT_EXTRA_LIBS ${KIT_EXTRA_LIBS} ${MATLAB_LIB}/libmx.lib)
SET (KIT_EXTRA_LIBS ${KIT_EXTRA_LIBS} ${MATLAB_LIB}/libmat.lib)
SET (KIT_EXTRA_LIBS ${KIT_EXTRA_LIBS} ${MATLAB_LIB}/libeng.lib)
SET (KIT_EXTRA_LIBS ${KIT_EXTRA_LIBS} ${RobartsVTK_SOURCE_DIR}/vtkAnis/OpenCV/lib/highgui.lib)
SET (KIT_EXTRA_LIBS ${KIT_EXTRA_LIBS} ${MIL_DIR}/LIB/Mil.lib )

TARGET_LINK_LIBRARIES(${MIL_DIR}/DLL/Mil.dll)
TARGET_LINK_LIBRARIES(${RobartsVTK_SOURCE_DIR}/vtkAnis/glew/bin/glew32.dll)

SET(Kit_EXTRA_SRCS)
SET(Kit_EXTRA_CMDS)
SET(Kit_TCL_EXTRA_SRCS)
SET(Kit_PYTHON_EXTRA_SRCS)
SET(Kit_JAVA_EXTRA_SRCS)

# Include the common setting file
INCLUDE(${RobartsVTK_SOURCE_DIR}/CMake/KitCommonBlock.cmake)
