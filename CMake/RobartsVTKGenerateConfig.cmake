# Generate the RobartsVTKConfig.cmake file in the build tree. This does not 
# configure one for installation. The file tells external projects how to use
# RobartsVTK.

#-----------------------------------------------------------------------------
# Settings for the build tree.

# The "use" file.
SET(RobartsVTK_USE_FILE
  ${RobartsVTK_SOURCE_DIR}/UseRobartsVTK.cmake)

# The library directories.
SET(RobartsVTK_LIBRARY_DIRS ${RobartsVTK_LIBRARY_DIR})

# The libraries.
SET(RobartsVTK_LIBRARIES ${RobartsVTK_LIBRARIES})

# The include directories.
SET(RobartsVTK_INCLUDE_DIRS ${RobartsVTK_INCLUDE_DIRS})

# The VTK options.
SET(Robarts_VTK_DIR ${VTK_DIR})

# The ITK options.
SET(RobartsVTK_USE_ITK "${RobartsVTK_USE_ITK}")
SET(RobartsVTK_ITK_DIR ${ITK_DIR})

# The libxml2 options
SET(RobartsVTK_USE_LIBXML2 "${RobartsVTK_USE_LIBXML2}")

# Configure RobartsVTKConfig.cmake for the build tree.
CONFIGURE_FILE(
  ${RobartsVTK_SOURCE_DIR}/RobartsVTKConfig.cmake.in
  ${RobartsVTK_BINARY_DIR}/RobartsVTKConfig.cmake @ONLY)