# Generate the RobartsVTKConfig.cmake file in the build tree. This does not 
# configure one for installation. The file tells external projects how to use
# RobartsVTK.

# Help store a literal dollar in a string.  CMake 2.2 allows escaped
# dollars but we have to support CMake 2.0.
SET(DOLLAR "$")

#-----------------------------------------------------------------------------
# Settings for the build tree.

#EXPORT_LIBRARY_DEPENDENCIES(
#  ${RobartsVTK_BINARY_DIR}/RobartsVTKLibraryDepends.cmake)

# Set the source dir
SET(RobartsVTK_SOURCE_DIR_CONFIG ${RobartsVTK_SOURCE_DIR})

# The library dependencies file.
SET(RobartsVTK_LIBRARY_DEPENDS_FILE
  ${RobartsVTK_BINARY_DIR}/RobartsVTKLibraryDepends.cmake)

#INCLUDE(${CMAKE_ROOT}/Modules/CMakeExportBuildSettings.cmake)

#CMAKE_EXPORT_BUILD_SETTINGS(
#  ${RobartsVTK_BINARY_DIR}/RobartsVTKBuildSettings.cmake)

# The "use" file.
SET(RobartsVTK_USE_FILE_CONFIG
  ${RobartsVTK_BINARY_DIR}/UseRobartsVTK.cmake)

# The build settings file.
SET(RobartsVTK_BUILD_SETTINGS_FILE_CONFIG
  ${RobartsVTK_BINARY_DIR}/RobartsVTKBuildSettings.cmake)

# The library directories.
SET(RobartsVTK_LIBRARY_DIRS_CONFIG ${RobartsVTK_LIBRARY_DIR})

# The kits.
SET(RobartsVTK_KITS_CONFIG ${RobartsVTK_KITS})

# The libraries.
SET(RobartsVTK_LIBRARIES_CONFIG ${RobartsVTK_LIBRARIES})

# The include directories.
SET(RobartsVTK_INCLUDE_DIRS_CONFIG "")
FOREACH(dir ${RobartsVTK_INCLUDE_DIRS})
  SET(RobartsVTK_INCLUDE_DIRS_CONFIG "${RobartsVTK_INCLUDE_DIRS_CONFIG};${dir}")
ENDFOREACH(dir ${RobartsVTK_INCLUDE_DIRS})

# The VTK options.
SET(Robarts_VTK_DIR_CONFIG ${VTK_DIR})

# The ITK options.
SET(RobartsVTK_USE_ITK_CONFIG "${RobartsVTK_USE_ITK}")
SET(RobartsVTK_ITK_DIR_CONFIG ${ITK_DIR})

# The libxml2 options
SET(RobartsVTK_USE_LIBXML2_CONFIG "${RobartsVTK_USE_LIBXML2}")
SET(LIBXML2_FOUND_CONFIG "${LIBXML2_FOUND}") # Necessary because it's not build by cmake

# AtamaVTK_USE_SYSTEM_DCMTK
SET(RobartsVTK_USE_SYSTEM_DCMTK_CONFIG "${VTK_USE_SYSTEM_DCMTK}")

# The library dependencies file.
#IF(NOT RobartsVTK_NO_LIBRARY_DEPENDS)
#  INCLUDE("@RobartsVTK_LIBRARY_DEPENDS_FILE@")
#ENDIF(NOT RobartsVTK_NO_LIBRARY_DEPENDS)

# Configure RobartsVTKConfig.cmake for the build tree.
CONFIGURE_FILE(
  ${RobartsVTK_SOURCE_DIR}/RobartsVTKConfig.cmake.in
  ${RobartsVTK_BINARY_DIR}/RobartsVTKConfig.cmake @ONLY)

# Configure the UseRobartsVTK file
CONFIGURE_FILE(${RobartsVTK_SOURCE_DIR}/UseRobartsVTK.cmake
               ${RobartsVTK_BINARY_DIR}/UseRobartsVTK.cmake COPYONLY)
