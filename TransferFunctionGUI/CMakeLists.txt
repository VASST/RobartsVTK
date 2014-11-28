#necessary for some reason
CMAKE_MINIMUM_REQUIRED(VERSION 2.6) 

#project name
PROJECT (TransferFunctionGUI)

# Look for a Qt build directory
FIND_PACKAGE(QT REQUIRED)
IF (QT_FOUND)
  INCLUDE (${QT_USE_FILE})
ENDIF (QT_FOUND)

# Look for CUDA
FIND_PACKAGE(CUDA)

# Look for RobartsVTK
FIND_PACKAGE(RobartsVTK REQUIRED)
INCLUDE(${RobartsVTK_USE_FILE})

# Look up some variables that weren't included in VTKConfig.cmake until VTK-4-4. 
IF(${VTK_MAJOR_VERSION} EQUAL 4 AND ${VTK_MINOR_VERSION} LESS 4) 
  INCLUDE(${TransferFunctionGUI_SOURCE_DIR}/CMake/GetMissingVTKConfig42.cmake)
ENDIF(${VTK_MAJOR_VERSION} EQUAL 4 AND ${VTK_MINOR_VERSION} LESS 4) 

# Starting in VTK 5, we need to include some Utilities directories.
IF (${VTK_MAJOR_VERSION} GREATER 4)
  INCLUDE(${TransferFunctionGUI_SOURCE_DIR}/CMake/ImportVTKUtilities.cmake)
ENDIF (${VTK_MAJOR_VERSION} GREATER 4)

# ---------------------------------------------------------------

FIND_PACKAGE( VTK )
IF (VTK_FOUND)
  INCLUDE (${VTK_USE_FILE})
ENDIF (VTK_FOUND)

# ---------------------------------------------------------------
# Initialize configuration

# Kits
SET(TransferFunctionUI_KITS)

# Libraries
SET(TransferFunctionUI_LIBRARIES)

# Include directories
SET(TransferFunctionUI_INCLUDE_DIRS)

# ---------------------------------------------------------------

#build the UI module
INCLUDE_DIRECTORIES ("${PROJECT_SOURCE_DIR}/UI")
ADD_SUBDIRECTORY (SharedComponents)
ADD_SUBDIRECTORY (GradTFUI)
ADD_SUBDIRECTORY (DualTFUI)
ADD_SUBDIRECTORY (K-TFGUI)