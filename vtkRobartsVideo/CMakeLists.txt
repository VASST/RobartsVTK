PROJECT(ROBARTSVIDEO)

SET(KIT RobartsVideo)
SET(UKIT RobartsVideo)
SET(KIT_PYTHON_LIBS vtkHybridPythonD RobartsVideoPythonD)
SET(KIT_LIBS vtkHybrid)
SET(KIT_EXTRA_LIBS)

SET ( Kit_SRCS
	vtkVisionSenseVideoSource.cxx
	vtkVisionSenseNetworkSource.cxx
	#vtkTestECGVideoSource.cxx
	vtkReplayImageVideoSource.cxx
)

IF (WIN32)
  OPTION (RobartsVTK_USE_DIRECTSHOW "Enable DirectShow Video Source Classes" OFF)
  OPTION (RobartsVTK_USE_EPIPHAN    "Enable Epiphan Video Source Classes" ON)
  OPTION (RobartsVTK_USE_SONIX 	    "Enable UltraSonix Video Source" OFF)
ENDIF (WIN32)

OPTION (RobartsVTK_USE_VIDEOSOURCE2 "Enable New Video Source Modules" OFF)

IF (RobartsVTK_USE_DIRECTSHOW)
  SET (Kit_SRCS ${Kit_SRCS} 
       vtkDirectShowVideoSource.cxx )

	FIND_LIBRARY(STRMIIDS strmiids "C:/WinDDK/7600.16385.1/lib/win7/")
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
    
	FIND_PATH(DIRECTSHOW_QEDIT_PATH qedit.h
	    "C:/Program Files/Microsoft Platform SDK for Windows Server 2003 R2/Include")

        SET(KIT_EXTRA_LIBS ${KIT_EXTRA_LIBS} ${DIRECTSHOW_BASECLASS_LIBRARY} ${DIRECTX_LIBRARY})
        INCLUDE_DIRECTORIES( "${DIRECTSHOW_PATH}/baseclasses/"  "${DIRECTSHOW_PATH}/common/" ${DIRECTX_PATH} ${WINDOWS_SDK_PATH})
	INCLUDE_DIRECTORIES( ${DIRECTSHOW_QEDIT_PATH} )
ENDIF (RobartsVTK_USE_DIRECTSHOW)

IF (RobartsVTK_USE_EPIPHAN)
  SET (Kit_SRCS ${Kit_SRCS} 
       vtkEpiphanVideoSource.cxx
       #vtkEpiphanDualVideoSource.cxx 
  )
  set(bitness 32)
  if(CMAKE_SIZEOF_VOID_P EQUAL 8)
  	set(bitness 64)
  endif()

  IF (bitness EQUAL 32)
	  find_library(FRMGRAB frmgrab ${RobartsVTK_SOURCE_DIR}/vtkRobartsVideo/epiphan/Win32)
  ELSE (bitness EQUAL 64)
	  find_library(FRMGRAB frmgrab ${RobartsVTK_SOURCE_DIR}/vtkRobartsVideo/epiphan/x64)
  ENDIF()

  SET (KIT_EXTRA_LIBS ${KIT_EXTRA_LIBS} ${FRMGRAB} comsuppw.lib )
ENDIF (RobartsVTK_USE_EPIPHAN)


IF ( RobartsVTK_USE_SONIX )
  SET (Kit_SRCS ${Kit_SRCS} 
       vtkSonixVideoSource.cxx )

	SET (ULTERIUS_ROOT "${RobartsVTK_SOURCE_DIR}/vtkRobartsVideo/ulterius-2.0/")

	FIND_PATH(ULTERIUS_INCLUDE_DIR "ulterius.h" "${ULTERIUS_ROOT}/inc")
	SET( CMAKE_FIND_LIBRARY_SUFFIXES .lib )
	FIND_LIBRARY(ULTERIUS_LIBRARY ulterius "${ULTERIUS_ROOT}/lib/")

	SET (KIT_EXTRA_LIBS ${KIT_EXTRA_LIBS} ${ULTERIUS_LIBRARY})
	INCLUDE_DIRECTORIES(${ULTERIUS_INCLUDE_DIR})
ENDIF (	RobartsVTK_USE_SONIX )

IF (RobartsVTK_USE_VIDEOSOURCE2)
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
	  FIND_LIBRARY(MIL_LIBRARY MIL
		"C:/Program Files/Matrox Imaging/mil/library/winnt/msc/dll"
		"C:/Program Files/Matrox Imaging/mil/library/windows/msc/dll"
	  )

      SET(KIT_EXTRA_LIBS ${KIT_EXTRA_LIBS} ${MIL_LIBRARY})
      IF (MIL_INCLUDE_PATH)
      ELSE (MIL_INCLUDE_PATH)
      FIND_PATH(MIL_INCLUDE_PATH mil.h "C:/Program Files/Matrox Imaging/mil/include" )
      ENDIF (MIL_INCLUDE_PATH)
      INCLUDE_DIRECTORIES( ${MIL_INCLUDE_PATH})
      SET (Kit_SRCS ${Kit_SRCS} vtkMILVideoSource2.cxx)
      # Include directory needed for vtkMILVideoSource.cxx.
    ENDIF (VTK_USE_MATROX_IMAGING)
  ENDIF (WIN32)
ENDIF (RobartsVTK_USE_VIDEOSOURCE2)

OPTION (RobartsVTK_USE_OPENCV_VIDEOSOURCE "Enable the OpenCV Video Source Modules" OFF)
IF (RobartsVTK_USE_OPENCV_VIDEOSOURCE)
  
  SET (Kit_SRCS ${Kit_SRCS} 
	vtkOpenCVVideoSource.cxx )

  FIND_PACKAGE( OpenCV REQUIRED )
  INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS} ${OpenCV_LIB_DIR} )
  SET (KIT_EXTRA_LIBS ${KIT_EXTRA_LIBS} ${OpenCV_LIBS} )
  SET (RobartsVTK_LIBRARY_DIRS ${RobartsVTK_LIBRARY_DIRS} ${OpenCV_LIB_DIR}/../3rdparty/lib)
  LINK_DIRECTORIES(${OpenCV_LIB_DIR}/../3rdparty/lib )

ENDIF (RobartsVTK_USE_OPENCV_VIDEOSOURCE)

#manage the vuzix base AR classes
OPTION (RobartsVTK_USE_VUZIX_UTILITIES "Enable the stereo utilities for Vuzix AR Applications" OFF)
IF (RobartsVTK_USE_VUZIX_UTILITIES)
  
  SET (Kit_SRCS ${Kit_SRCS} 
	vtkVuzixARCamera.h vtkVuzixARCamera.cxx
	vtkVuzixARScene.h  vtkVuzixARScene.cxx )

ENDIF (RobartsVTK_USE_VUZIX_UTILITIES)

SET(Kit_EXTRA_SRCS)
SET(Kit_EXTRA_CMDS)
SET(Kit_TCL_EXTRA_SRCS)
SET(Kit_PYTHON_EXTRA_SRCS)
SET(Kit_JAVA_EXTRA_SRCS)

# -----------------------------------------------------------------

# Include the common setting file
INCLUDE(${RobartsVTK_SOURCE_DIR}/CMake/KitCommonBlock.cmake)
