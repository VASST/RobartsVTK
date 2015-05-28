#
# This module is provided as RobartsVTK_USE_FILE by RobartsVTKConfig.cmake.  
# It can be INCLUDEd in a project to load the needed compiler and linker
# settings to use RobartsVTK:
#   FIND_PACKAGE(RobartsVTK REQUIRED)
#   INCLUDE(${RobartsVTK_USE_FILE})
#

IF(NOT RobartsVTK_USE_FILE_INCLUDED)
  SET(RobartsVTK_USE_FILE_INCLUDED 1)

  # Add include directories needed to use RobartsVTK.
  INCLUDE_DIRECTORIES(${RobartsVTK_INCLUDE_DIRS})

  # Add link directories needed to use RobartsVTK.
  LINK_DIRECTORIES(${RobartsVTK_LIBRARY_DIRS})

  #
  # VTK
  #
  IF(NOT VTK_DIR)
    # Use RobartsVTK_VTK_DIR or find a new one
    IF(RobartsVTK_VTK_DIR)
      SET(VTK_DIR ${RobartsVTK_VTK_DIR} CACHE PATH "Path to VTK build dir")
      INCLUDE(${VTK_DIR}/VTKConfig.cmake)
    ELSE(RobartsVTK_VTK_DIR)
      FIND_PACKAGE(VTK REQUIRED)
    ENDIF(RobartsVTK_VTK_DIR)
  ELSE(NOT VTK_DIR)
    INCLUDE(${VTK_DIR}/VTKConfig.cmake)
  ENDIF(NOT VTK_DIR)

  # Include the VTK use file
  INCLUDE(${VTK_USE_FILE})

  #
  # ITK (optional)
  #
  IF(RobartsVTK_USE_ITK)
    IF(NOT ITK_DIR)
      # Use RobartsVTK_ITK_DIR or find a new one
      IF(RobartsVTK_ITK_DIR)
  SET(ITK_DIR ${RobartsVTK_ITK_DIR} CACHE PATH "Path to ITK build dir")
  INCLUDE(${ITK_DIR}/ITKConfig.cmake)
      ELSE(RobartsVTK_ITK_DIR)
  FIND_PACKAGE(ITK REQUIRED)
      ENDIF(RobartsVTK_ITK_DIR)
    ELSE(NOT ITK_DIR)
      INCLUDE(${ITK_DIR}/ITKConfig.cmake)
    ENDIF(NOT ITK_DIR)
    
    # Include the ITK use file
    INCLUDE(${ITK_USE_FILE})

  ENDIF(RobartsVTK_USE_ITK)

ENDIF(NOT RobartsVTK_USE_FILE_INCLUDED)
