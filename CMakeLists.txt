PROJECT(RobartsVTK)

# CMake version requirements
CMAKE_MINIMUM_REQUIRED(VERSION 2.4)

#SET(CMAKE_SKIP_RPATH ON CACHE BOOL "rpath not included in .so files" FORCE)

# Look for a VTK build directory
FIND_PACKAGE(VTK)
IF (VTK_FOUND)
  INCLUDE (${VTK_USE_FILE})
ENDIF (VTK_FOUND)

# Look for a AtamaiVTK build directory)
FIND_PACKAGE(AIGS)
IF (AIGS_FOUND)
  INCLUDE (${AIGS_USE_FILE})
ENDIF (AIGS_FOUND)

#FIND_PACKAGE(PythonLibs)
#INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_PATH})


# Look for an ITK build directory (optional)
OPTION(RobartsVTK_USE_ITK "Use ITK in RobartsITK" OFF)
IF(RobartsVTK_USE_ITK)
  FIND_PACKAGE(ITK QUIET)
  IF(ITK_FOUND)
    INCLUDE(${ITK_USE_FILE})
  ENDIF(ITK_FOUND)
ENDIF(RobartsVTK_USE_ITK)


# Look for a libxml2 (optional), only VTK 5 and higher
IF(${VTK_MAJOR_VERSION} GREATER 4)
  # LIBXML2_FOUND: We need this variable because CMake's FindLibXml2.cmake module does 
  #                not set LibXml2_DIR like most FIND_PACKAGE modules.
  SET(LIBXML2_FOUND 0) 
  OPTION(RobartsVTK_USE_LIBXML2 "Use libxml2 in RobartsVTK" ON)
  IF(RobartsVTK_USE_LIBXML2)
    FIND_PACKAGE(LibXml2 QUIET)
    IF(LIBXML2_FOUND)
      INCLUDE_DIRECTORIES(${LIBXML2_INCLUDE_DIR})
      LINK_LIBRARIES(${LIBXML2_LIBRARIES})
      ADD_DEFINITIONS(${LIBXML2_DEFINITIONS})
    ENDIF(LIBXML2_FOUND)
  ENDIF(RobartsVTK_USE_LIBXML2)
ENDIF(${VTK_MAJOR_VERSION} GREATER 4)

# Look up some variables that weren't included in VTKConfig.cmake until VTK-4-4. 
IF(${VTK_MAJOR_VERSION} EQUAL 4 AND ${VTK_MINOR_VERSION} LESS 4) 
  INCLUDE(${RobartsVTK_SOURCE_DIR}/CMake/GetMissingVTKConfig42.cmake)
ENDIF(${VTK_MAJOR_VERSION} EQUAL 4 AND ${VTK_MINOR_VERSION} LESS 4) 

# Starting in VTK 5, we need to include some Utilities directories.
IF (${VTK_MAJOR_VERSION} GREATER 4)
  INCLUDE(${RobartsVTK_SOURCE_DIR}/CMake/ImportVTKUtilities.cmake)
ENDIF (${VTK_MAJOR_VERSION} GREATER 4)

# Choose static or shared libraries.
OPTION(BUILD_SHARED_LIBS "Build shared libraries." OFF)
SET (RobartsVTK_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
IF (BUILD_SHARED_LIBS)
  SET (BUILD_SHARED_LIBS_CONFIG 1)
ELSE (BUILD_SHARED_LIBS)
  SET (BUILD_SHARED_LIBS_CONFIG 0)
ENDIF (BUILD_SHARED_LIBS)

# Build documentation
OPTION(BUILD_DOCUMENTATION "Build Documentation" OFF)
IF(BUILD_DOCUMENTATION)
  FIND_PACKAGE(VTK)
  IF (USE_VTK_FILE)
    INCLUDE (${USE_VTK_FILE})
  ELSE (USE_VTK_FILE)
    MESSAGE( FATAL_ERROR "This application requires VTK. One of these
components is missing. Please verify configuration")
  ENDIF (USE_VTK_FILE)
  SUBDIRS(Documentation/Doxygen)
ENDIF(BUILD_DOCUMENTATION)

# Output directories.
SET(LIBRARY_OUTPUT_PATH ${RobartsVTK_BINARY_DIR}/bin CACHE INTERNAL "Single output directory for building all libraries.")
SET(EXECUTABLE_OUTPUT_PATH ${RobartsVTK_BINARY_DIR}/bin CACHE INTERNAL "Single output directory for building all executables.")
SET(RobartsVTK_LIBRARY_DIR ${LIBRARY_OUTPUT_PATH}/${CMAKE_CFG_INTDIR})
SET(RobartsVTK_EXECUTABLE_DIR ${EXECUTABLE_OUTPUT_PATH}/${CMAKE_CFG_INTDIR})
SET(CXX_TEST_PATH ${EXECUTABLE_OUTPUT_PATH})

# Configure Dart testing support.
INCLUDE(${CMAKE_ROOT}/Modules/Dart.cmake)

# ---------------------------------------------------------------
# Initialize configuration

# Kits
SET(RobartsVTK_KITS)

# Libraries
SET(RobartsVTK_LIBRARIES)

# Include directories
SET(RobartsVTK_INCLUDE_DIRS)
# ---------------------------------------------------------------

# Include the Tcl and Python wrappers
IF(VTK_WRAP_TCL AND ${VTK_MAJOR_VERSION} GREATER 4)
    INCLUDE(${VTK_CMAKE_DIR}/vtkWrapTcl.cmake)
ENDIF(VTK_WRAP_TCL AND ${VTK_MAJOR_VERSION} GREATER 4)

IF(VTK_WRAP_PYTHON AND ${VTK_MAJOR_VERSION} GREATER 4)
  IF(VTK_PYTHON_INCLUDE_DIR)
    INCLUDE_DIRECTORIES("${VTK_PYTHON_INCLUDE_DIR}")
  ENDIF(VTK_PYTHON_INCLUDE_DIR)
  INCLUDE(${VTK_CMAKE_DIR}/vtkWrapPython.cmake)
ENDIF(VTK_WRAP_PYTHON AND ${VTK_MAJOR_VERSION} GREATER 4)

IF(VTK_WRAP_JAVA AND ${VTK_MAJOR_VERSION} GREATER 4)
    SET(VTK_WRAP_JAVA 0)
ENDIF(VTK_WRAP_JAVA AND ${VTK_MAJOR_VERSION} GREATER 4)

# Add subdirectories to build
INCLUDE(${VTK_SOURCE_DIR}/CMake/vtkThirdParty.cmake)

# ---------------------------------------------------------------
# Start listing the RobartsVTK kits

# vtkRobartsHaptics
OPTION (RobartsVTK_USE_ROBARTSHAPTICS "Build the vtkRobartsHaptics module" OFF)
IF (RobartsVTK_USE_ROBARTSHAPTICS)
  ADD_SUBDIRECTORY(vtkRobartsHaptics)
  SET(RobartsVTK_INCLUDE_DIRS ${RobartsVTK_INCLUDE_DIRS} ${RobartsVTK_SOURCE_DIR}/vtkRobartsHaptics )
  SET(RobartsVTK_LIBRARIES ${RobartsVTK_LIBRARIES} vtkRobartsHaptics)
ENDIF (RobartsVTK_USE_ROBARTSHAPTICS)

# vtkRobartsLegacy
OPTION (RobartsVTK_USE_ROBARTSLEGACY "Build the vtkRobartsLegacy module" OFF)
IF (RobartsVTK_USE_ROBARTSLEGACY)
  SUBDIRS(vtkRobartsLegacy)
  SET(RobartsVTK_INCLUDE_DIRS ${RobartsVTK_INCLUDE_DIRS} ${RobartsVTK_SOURCE_DIR}/vtkRobartsLegacy )
  SET(RobartsVTK_LIBRARIES ${RobartsVTK_LIBRARIES} vtkRobartsLegacy)
ENDIF (RobartsVTK_USE_ROBARTSLEGACY)

# vtkRobartsECG
OPTION (RobartsVTK_USE_ROBARTSECG "Build the vtkRobartsECG module" ON)
IF (RobartsVTK_USE_ROBARTSECG)
  SUBDIRS(vtkRobartsECG)
  SET(RobartsVTK_INCLUDE_DIRS ${RobartsVTK_INCLUDE_DIRS} ${RobartsVTK_SOURCE_DIR}/vtkRobartsECG )
  SET(RobartsVTK_LIBRARIES ${RobartsVTK_LIBRARIES} vtkRobartsECG)
ENDIF (RobartsVTK_USE_ROBARTSECG)

# vtkRobartsVideo
OPTION (RobartsVTK_USE_ROBARTSVIDEO "Build the vtkRobartsVideo module" ON)
IF (RobartsVTK_USE_ROBARTSVIDEO)
  SUBDIRS(vtkRobartsVideo)
  SET(RobartsVTK_INCLUDE_DIRS ${RobartsVTK_INCLUDE_DIRS} ${RobartsVTK_SOURCE_DIR}/vtkRobartsVideo )
  SET(RobartsVTK_LIBRARIES ${RobartsVTK_LIBRARIES} vtkRobartsVideo)
ENDIF (RobartsVTK_USE_ROBARTSVIDEO)

# vtkMwierz
OPTION (RobartsVTK_USE_MWIERZ "Build the vtkMwierz module" OFF)
IF (RobartsVTK_USE_MWIERZ)
  SUBDIRS(vtkMwierz)
  SET(RobartsVTK_INCLUDE_DIRS ${RobartsVTK_INCLUDE_DIRS} ${RobartsVTK_SOURCE_DIR}/vtkMwierz )
  SET(RobartsVTK_LIBRARIES ${RobartsVTK_LIBRARIES} vtkMwierz)
ENDIF (RobartsVTK_USE_MWIERZ)

# vtkAnix
OPTION (RobartsVTK_USE_ANIS "Build the vtkAnis module" OFF)
IF (RobartsVTK_USE_ANIS)
  SUBDIRS(vtkAnis)
  SET(RobartsVTK_INCLUDE_DIRS ${RobartsVTK_INCLUDE_DIRS} ${RobartsVTK_SOURCE_DIR}/vtkAnis )
  SET(RobartsVTK_LIBRARIES ${RobartsVTK_LIBRARIES} vtkAnis)
ENDIF (RobartsVTK_USE_ANIS)

# vtkCuda
OPTION (RobartsVTK_USE_CUDA "Build the CUDA module" OFF)
IF (RobartsVTK_USE_CUDA)
  
  # Look for CUDA facilities
  FIND_PACKAGE(CUDA REQUIRED)
  
  # Add Robart's CUDA libraries
  SUBDIRS(vtkCudaCommon)
  SUBDIRS(vtkCudaVisualization)
  SET(RobartsVTK_INCLUDE_DIRS ${RobartsVTK_INCLUDE_DIRS} ${RobartsVTK_SOURCE_DIR}/vtkCudaCommon) 
  SET(RobartsVTK_INCLUDE_DIRS ${RobartsVTK_INCLUDE_DIRS} ${RobartsVTK_SOURCE_DIR}/vtkCudaVisualization)
  SET(RobartsVTK_LIBRARIES ${RobartsVTK_LIBRARIES} vtkCudaCommon vtkCudaVisualization)
ENDIF (RobartsVTK_USE_CUDA)

# vtkRobartsUltrasound - contains new freehand ultrasound code
OPTION (RobartsVTK_USE_ROBARTSULTRASOUND "Build the vtkRobartsUltrasound module" OFF)
IF (RobartsVTK_USE_ROBARTSULTRASOUND)
  SUBDIRS (vtkRobartsUltrasound)
  SET(RobartsVTK_INCLUDE_DIRS ${RobartsVTK_INCLUDE_DIRS} ${RobartsVTK_SOURCE_DIR}/vtkCuda ${RobartsVTK_SOURCE_DIR}/vtkRobartsUltrasound)
  SET(RobartsVTK_LIBRARIES ${RobartsVTK_LIBRARIES} vtkRobartsUltrasound)
ENDIF (RobartsVTK_USE_ROBARTSULTRASOUND)


# Add RobartsVTK.py.in file.  On Win32, this has to go into
# all config 
IF(VTK_WRAP_PYTHON)
  IF(WIN32 AND CMAKE_CONFIGURATION_TYPES)
    FOREACH(config ${CMAKE_CONFIGURATION_TYPES})
      CONFIGURE_FILE(
        ${RobartsVTK_SOURCE_DIR}/RobartsVTK.py.in
        ${LIBRARY_OUTPUT_PATH}/${config}/RobartsVTK.py)
    ENDFOREACH(config)
  ELSE(WIN32 AND CMAKE_CONFIGURATION_TYPES)
    CONFIGURE_FILE(
      ${RobartsVTK_SOURCE_DIR}/RobartsVTK.py.in
      ${LIBRARY_OUTPUT_PATH}/RobartsVTK.py)
  ENDIF(WIN32 AND CMAKE_CONFIGURATION_TYPES)
ENDIF(VTK_WRAP_PYTHON)

#-----------------------------------------------------------------------------
# The commands in this directory are intended to be executed as
# the end of the whole configuration process, as a "last step".
# This directory is typically the last SUBDIRS in the main CMakeLists.txt.
SUBDIRS(CMake/LastConfigureStep)

