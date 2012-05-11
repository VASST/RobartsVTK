PROJECT(ROBARTSECG)

SET(KIT RobartsECG)
SET(UKIT ROBARTSECG)
SET(KIT_LIBS vtkHybrid)
SET(KIT_EXTRA_LIBS)

###### Base signal box ######
SET ( Kit_SRCS ${Kit_SRCS}
    vtkSignalBox.cxx
    #vtkTestECGSignalBox.cxx
)


  INCLUDE_DIRECTORIES(${ROBARTSVIDEO_BINARY_DIR}
                      ${ROBARTSVIDEO_SOURCE_DIR} )

###### Heart signal box ######
OPTION (RobartsVTK_USE_HEARTSIGNALBOX "Enable Heart Signal Box" ON)
IF (RobartsVTK_USE_HEARTSIGNALBOX)
    SET (Kit_SRCS ${Kit_SRCS}
         vtkHeartSignalBox.cxx
        )

  SET (KIT_EXTRA_LIBS ${KIT_EXTRA_LIBS} ${RobartsVTK_SOURCE_DIR}/vtkRobartsECG/Inpout/inpout32.lib)
ENDIF (RobartsVTK_USE_HEARTSIGNALBOX)

###### Data box ######
OPTION (RobartsVTK_USE_DATABOX "Enable Data Acquisition Box" ON)
IF (RobartsVTK_USE_DATABOX)
    SET (Kit_SRCS ${Kit_SRCS}
          vtkDataBox.cxx
          vtkDataBoxBuffer.cxx
         )
    SET (KIT_EXTRA_LIBS ${KIT_EXTRA_LIBS}	${RobartsVTK_SOURCE_DIR}/vtkRobartsECG/databox/Graph32.lib
                                          ${RobartsVTK_SOURCE_DIR}/vtkRobartsECG/databox//oldaapi32.lib
                                          ${RobartsVTK_SOURCE_DIR}/vtkRobartsECG/databox//OLMEM32.lib )
ENDIF (RobartsVTK_USE_DATABOX)

###### USB Signal box ######
OPTION (RobartsVTK_USE_USBECGBOX "Enable USB ECG Signal Box" ON)

IF (RobartsVTK_USE_USBECGBOX)
  SET ( Kit_SRCS ${Kit_SRCS}
        vtkUSBECGBox.cxx
      )
  SET (KIT_EXTRA_LIBS ${KIT_EXTRA_LIBS}	${RobartsVTK_SOURCE_DIR}/vtkRobartsECG/cbw/cbw32.lib )
ENDIF (RobartsVTK_USE_USBECGBOX)

###### Lego signal box ######
OPTION (RobartsVTK_USE_LEGO "Provide support for triggering using LEGO" OFF)
  IF (RobartsVTK_USE_LEGO)
    SET ( Kit_SRCS ${Kit_SRCS}
          vtkLegoSignalBox.cxx
	  )

  # include the libusb directory
  SET(USB_INCLUDE_DIR ${RobartsVTK_SOURCE_DIR}/vtkRobartsECG/LegoFiles/include/ )
  SET(USB_LIBRARIES ${RobartsVTK_SOURCE_DIR}/vtkRobartsECG/LegoFiles/lib/msvc/libusb.lib )


  SET(CMAKE_CXX_FLAGS
    "-funsigned-char"
  )

  INCLUDE_DIRECTORIES(${VTKLEGO_SOURCE_DIR}
                      ${VTKLEGO_BINARY_DIR}
                      ${USB_INCLUDE_DIR}
			   )

  SET (KIT_EXTRA_LIBS ${KIT_EXTRA_LIBS} vtklego)
  SET (KIT_EXTRA_LIBS ${KIT_EXTRA_LIBS} ${USB_LIBRARIES})

  SUBDIRS(LegoFiles)

ENDIF (RobartsVTK_USE_LEGO)

SET(Kit_EXTRA_SRCS)
SET(Kit_EXTRA_CMDS)
SET(Kit_TCL_EXTRA_SRCS)
SET(Kit_PYTHON_EXTRA_SRCS)
SET(Kit_JAVA_EXTRA_SRCS)

# -----------------------------------------------------------------

# Include the common setting file
INCLUDE(${RobartsVTK_SOURCE_DIR}/CMake/KitCommonBlock.cmake)