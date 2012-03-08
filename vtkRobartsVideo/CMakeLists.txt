PROJECT(ROBARTSVIDEO)

SET(KIT RobartsVideo)
SET(UKIT RobartsVideo)
SET(KIT_PYTHON_LIBS vtkHybridPythonD RobartsVideoPythonD)
SET(KIT_LIBS vtkHybrid)
SET(KIT_EXTRA_LIBS)

SET ( Kit_SRCS
vtkVisionSenseVideoSource.cxx
vtkVisionSenseNetworkSource.cxx
vtkTestECGVideoSource.cxx

)


OPTION (VTK_USE_VIDEOSOURCE2 "Enable New Video Source Modules" OFF)
IF (WIN32)
  OPTION (VTK_USE_DIRECTSHOW "Enable DirectShow Video Source Classes" OFF)
ENDIF (WIN32)

IF (VTK_USE_DIRECTSHOW)
  SET (Kit_SRCS ${Kit_SRCS} 
       vtkDirectShowVideoSource.cxx )

	SET (KIT_EXTRA_LIBS ${KIT_EXTRA_LIBS} strmiids)
 	FIND_LIBRARY(DIRECTSHOW_BASECLASS_LIBRARY strmbase
	    "C:/Program Files/Microsoft SDKs/Windows/v7.1/Samples/multimedia/directshow/baseclasses/x64/Release"
	    "C:/Program Files/Microsoft SDKs/Windows/v7.1/Samples/multimedia/directshow/baseclasses/Release"
	    "C:/Program Files/Microsoft SDKs/Windows/v7.1/Samples/multimedia/directshow/baseclasses/Debug"
	    "C:/Program Files/Microsoft SDKs/Windows/v7.1/Samples/multimedia/directshow/baseclasses/x64/Debug"
	)

	FIND_LIBRARY(DIRECTX_LIBRARY d3dx9
	    "C:/Program Files/Microsoft DirectX SDK (June 2010)/lib/x64"
	    "C:/Program Files/Microsoft DirectX SDK (June 2010)/lib/x86"
	)

	FIND_PATH(WINDOWS_SDK_PATH DShow.h
	    "C:/Program Files/Microsoft SDKs/Windows/v7.1/Include"
	)

	FIND_PATH(DIRECTSHOW_PATH streams.h
	    "C:/Program Files/Microsoft SDKs/Windows/v7.1/Samples/multimedia/directshow/"
	)

	FIND_PATH(DIRECTX_PATH d3dx9.h
	    "C:/Program Files/Microsoft DirectX SDK (June 2010)/Include"
	)
        find_library(STRMIIDS strmiids "C:/WinDDK/7600.16385.1/lib/win7/")


        SET(KIT_EXTRA_LIBS ${KIT_EXTRA_LIBS} ${DIRECTSHOW_BASECLASS_LIBRARY} ${DIRECTX_LIBRARY})
        INCLUDE_DIRECTORIES( "${DIRECTSHOW_PATH}/baseclasses/"  "${DIRECTSHOW_PATH}/common/" ${DIRECTX_PATH} ${WINDOWS_SDK_PATH})


ENDIF (VTK_USE_DIRECTSHOW)

IF (VTK_USE_VIDEOSOURCE2)
  SET (Kit_SRCS ${Kit_SRCS} 
       vtkVideoBuffer2.cxx
       vtkVideoFrame2.cxx
       vtkVideoSource2.cxx )

  SET (KIT_EXTRA_LIBS ${KIT_EXTRA_LIBS} ws2_32.lib)

  IF (WIN32)
    IF (VTK_VFW_SUPPORTS_CAPTURE)
      SET (Kit_SRCS ${Kit_SRCS} vtkWin32VideoSource2.cxx)
    ENDIF (VTK_VFW_SUPPORTS_CAPTURE)

    IF (VTK_USE_MATROX_IMAGING)
      SET(KIT_LIBS ${KIT_LIBS} ${MIL_LIBRARY})
      IF (MIL_INCLUDE_PATH)
      ELSE (MIL_INCLUDE_PATH)
      FIND_PATH(MIL_INCLUDE_PATH mil.h "C:/Program Files/Matrox Imaging/mil/include" )
      ENDIF (MIL_INCLUDE_PATH)
      INCLUDE_DIRECTORIES( ${MIL_INCLUDE_PATH})
      SET (Kit_SRCS ${Kit_SRCS} vtkMILVideoSource2.cxx)
      # Include directory needed for vtkMILVideoSource.cxx.
    ENDIF (VTK_USE_MATROX_IMAGING)
  ENDIF (WIN32)
ENDIF (VTK_USE_VIDEOSOURCE2)

SET(Kit_EXTRA_SRCS)
SET(Kit_EXTRA_CMDS)
SET(Kit_TCL_EXTRA_SRCS)
SET(Kit_PYTHON_EXTRA_SRCS)
SET(Kit_JAVA_EXTRA_SRCS)

# -----------------------------------------------------------------

# Include the common setting file
INCLUDE(${RobartsVTK_SOURCE_DIR}/CMake/KitCommonBlock.cmake)
