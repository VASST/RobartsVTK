PROJECT(RobartsVTK)

# CMake version requirements
CMAKE_MINIMUM_REQUIRED(VERSION 3.2)
IF(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
ENDIF(COMMAND cmake_policy)

#SET(CMAKE_SKIP_RPATH ON CACHE BOOL "rpath not included in .so files" FORCE)

# Look for a VTK build directory
FIND_PACKAGE(VTK)
IF (VTK_FOUND)
  INCLUDE (${VTK_USE_FILE})
ENDIF (VTK_FOUND)

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
  OPTION(RobartsVTK_USE_LIBXML2 "Use libxml2 in RobartsVTK (only VTK 5 and higher)" ON)
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
  ADD_SUBDIRECTORY(Documentation/Doxygen)
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
    INCLUDE_DIRECTORIES("${VTK_TCL_INCLUDE_DIR}" "${VTK_TK_INCLUDE_DIR}")
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

# ---------------------------------------------------------------
# Start listing the RobartsVTK kits

# Qt
OPTION (RobartsVTK_USE_QT "Build with QT enabled" OFF)
IF (RobartsVTK_USE_QT)
  FIND_PACKAGE(QT REQUIRED)
ENDIF (RobartsVTK_USE_QT)

# vtkMwierz - Registration
OPTION (RobartsVTK_USE_MWIERZ "Build the vtkMwierz module" OFF)
IF (RobartsVTK_USE_MWIERZ)
  ADD_SUBDIRECTORY(vtkMwierz)
  SET(RobartsVTK_INCLUDE_DIRS ${RobartsVTK_INCLUDE_DIRS} ${RobartsVTK_SOURCE_DIR}/vtkMwierz )
  SET(RobartsVTK_LIBRARIES ${RobartsVTK_LIBRARIES} vtkMwierz)
ENDIF (RobartsVTK_USE_MWIERZ)

# vtkRobartsUtilities - contains random utilities like a Read-Write lock and Image Pipe over TCI/IP
OPTION (RobartsVTK_USE_ROBARTSUTILITIES "Build the vtkRobartsUtilities module" OFF)
IF (RobartsVTK_USE_ROBARTSUTILITIES)
  ADD_SUBDIRECTORY (vtkRobartsUtilities)
  SET(RobartsVTK_INCLUDE_DIRS ${RobartsVTK_INCLUDE_DIRS} ${RobartsVTK_SOURCE_DIR}/vtkRobartsUtilities)
  SET(RobartsVTK_LIBRARIES ${RobartsVTK_LIBRARIES} vtkRobartsUtilities)
ENDIF (RobartsVTK_USE_ROBARTSUTILITIES)

# vtkCuda
OPTION (RobartsVTK_USE_CUDA "Build the CUDA modules" OFF)
IF (RobartsVTK_USE_CUDA)

  # Remove vtk definitions
  # This is used for CUDA targets, because nvcc does not like VTK 6+ definitions style.
  macro(REMOVE_VTK_DEFINITIONS)
    get_directory_property(_dir_defs DIRECTORY ${CMAKE_SOURCE_DIR} COMPILE_DEFINITIONS)
    set(_vtk_definitions)
    foreach(_item ${_dir_defs})
        if(_item MATCHES "vtk*")
            list(APPEND _vtk_definitions -D${_item})
        endif()
    endforeach()
    remove_definitions(${_vtk_definitions})
  endmacro(REMOVE_VTK_DEFINITIONS)

  # Look for CUDA facilities
  FIND_PACKAGE(CUDA REQUIRED)

  INCLUDE(FindCUDA)
  # Add Robart's CUDA libraries
  ADD_SUBDIRECTORY(vtkCudaCommon)
  SET(RobartsVTK_INCLUDE_DIRS ${RobartsVTK_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${RobartsVTK_SOURCE_DIR}/vtkCudaCommon)
  SET(RobartsVTK_LIBRARIES ${RobartsVTK_LIBRARIES} vtkCudaCommon)

ENDIF (RobartsVTK_USE_CUDA)

# Add the CUDA Visualization Libraries
IF (RobartsVTK_USE_CUDA)
  OPTION (RobartsVTK_USE_CUDA_VISUALIZATION "Build the CUDA visualization modules" ON)
  IF (RobartsVTK_USE_CUDA_VISUALIZATION)
    ADD_SUBDIRECTORY(vtkCudaVisualization)
    SET(RobartsVTK_INCLUDE_DIRS ${RobartsVTK_INCLUDE_DIRS} ${RobartsVTK_SOURCE_DIR}/vtkCudaVisualization)
    SET(RobartsVTK_LIBRARIES ${RobartsVTK_LIBRARIES} vtkCudaVisualization)
  ENDIF (RobartsVTK_USE_CUDA_VISUALIZATION)
ENDIF (RobartsVTK_USE_CUDA)

# Add the CUDA Analytics Libraries
IF (RobartsVTK_USE_CUDA AND RobartsVTK_USE_ROBARTSUTILITIES)
  OPTION (RobartsVTK_USE_CUDA_ANALYTICS "Build the CUDA image analytics modules" ON)
  IF (RobartsVTK_USE_CUDA_ANALYTICS)
    ADD_SUBDIRECTORY(vtkCudaImageAnalytics)
    ADD_SUBDIRECTORY(MaxFlowUtils)
    SET(RobartsVTK_INCLUDE_DIRS ${RobartsVTK_INCLUDE_DIRS} ${RobartsVTK_SOURCE_DIR}/vtkCudaImageAnalytics)
    SET(RobartsVTK_LIBRARIES ${RobartsVTK_LIBRARIES} vtkCudaImageAnalytics)
  ENDIF ()
ENDIF (RobartsVTK_USE_CUDA AND RobartsVTK_USE_ROBARTSUTILITIES)

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
# Build structure for examples, segregated by other options
OPTION(RobartsVTK_BUILD_EXAMPLES "Build RobartsVTK examples." OFF)
IF(RobartsVTK_BUILD_EXAMPLES)
  IF(RobartsVTK_USE_CUDA)
    ADD_SUBDIRECTORY(CudaExamples)
  IF(RobartsVTK_USE_QT)
    ADD_SUBDIRECTORY(FluoroPredViz)
    ADD_SUBDIRECTORY(TransferFunctionGUI)
  ENDIF(RobartsVTK_USE_QT)
  ENDIF(RobartsVTK_USE_CUDA)
  IF(RobartsVTK_USE_ROBARTSUTILITIES)
    ADD_SUBDIRECTORY(MaxFlowExamples)
  ENDIF(RobartsVTK_USE_ROBARTSUTILITIES)
ENDIF(RobartsVTK_BUILD_EXAMPLES)

#-----------------------------------------------------------------------------
# The commands in this directory are intended to be executed as
# the end of the whole configuration process, as a "last step".
# This directory is typically the last ADD_SUBDIRECTORY in the main CMakeLists.txt.
ADD_SUBDIRECTORY(CMake/LastConfigureStep)
