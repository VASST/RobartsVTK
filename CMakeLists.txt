PROJECT(RobartsVTK)

# todo python wrapping
# todo module setup for cuda entries
# todo module export doesn't have dllexport/import - build_shared_libs?

# CMake version requirements
CMAKE_MINIMUM_REQUIRED(VERSION 3.2)
cmake_policy(SET CMP0003 NEW)

SET(RobartsVTK_MAJOR_VERSION 2)
SET(RobartsVTK_MINOR_VERSION 0)
SET(RobartsVTK_PATCH_VERSION 0)

INCLUDE(ExternalProject)
INCLUDE(${CMAKE_SOURCE_DIR}/CMake/RobartsCommon.cmake)

#-----------------------------------------------------------------------------
# RobartsVTK executable output path 
#-----------------------------------------------------------------------------
SET (RobartsVTK_EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}/bin")

#-----------------------------------------------------------------------------
# Look for a VTK build directory
#-----------------------------------------------------------------------------
FIND_PACKAGE(VTK 6.2.0 REQUIRED NO_MODULE PATHS ${VTK_DIR} NO_DEFAULT_PATH)

IF( VTK_FOUND )
  MESSAGE(STATUS "Using VTK available at: ${VTK_DIR}")

  # Copy libraries to RobartsVTK_EXECUTABLE_OUTPUT_PATH
  IF ( ${CMAKE_GENERATOR} MATCHES "Visual Studio" )
    FOREACH(lib ${VTK_LIBRARIES})
      IF(NOT TARGET ${lib})
        continue()
      ENDIF()

      GET_TARGET_PROPERTY(VTK_DEBUG_FILE ${lib} IMPORTED_LOCATION_DEBUG)
      GET_TARGET_PROPERTY(VTK_RELEASE_FILE ${lib} IMPORTED_LOCATION_RELEASE)

      IF( EXISTS ${VTK_RELEASE_FILE} )
        FILE(COPY ${VTK_RELEASE_FILE} DESTINATION ${RobartsVTK_EXECUTABLE_OUTPUT_PATH}/Release)
      ENDIF()
      IF( EXISTS ${VTK_DEBUG_FILE} )
        FILE(COPY ${VTK_DEBUG_FILE} DESTINATION ${RobartsVTK_EXECUTABLE_OUTPUT_PATH}/Debug)
      ENDIF()
    ENDFOREACH()
  ELSE()
    FILE(COPY 
      ${VTK_LIBRARY_DIRS}/
      DESTINATION ${RobartsVTK_EXECUTABLE_OUTPUT_PATH}
      FILES_MATCHING REGEX .*${CMAKE_SHARED_LIBRARY_SUFFIX}
      )
  ENDIF()
  SET (RobartsVTK_VTK_DIR "${VTK_DIR}" CACHE INTERNAL "Path to store vtk binaries")

  IF(RobartsVTK_USE_QT)
    IF (${VTK_MAJOR_VERSION} VERSION_LESS 6)
      IF (NOT VTK_USE_QT)
        MESSAGE( SEND_ERROR "You have to build VTK with VTK_USE_QT flag ON if you need to use RobartsVTK_USE_QT.")
      ENDIF (NOT VTK_USE_QT)
    ELSE()
      IF (NOT TARGET vtkGUISupportQt)
        MESSAGE( SEND_ERROR "You have to build VTK with VTK_USE_QT flag ON if you need to use RobartsVTK_USE_QT.")
      ENDIF()
    ENDIF()
  ENDIF()

  INCLUDE (${VTK_USE_FILE})
ENDIF ()

#-----------------------------------------------------------------------------
# Qt
#-----------------------------------------------------------------------------
OPTION (RobartsVTK_USE_QT "Build with QT enabled" OFF)
IF (RobartsVTK_USE_QT)
  FIND_FILE(QT_QMAKE_EXECUTABLE qmake${CMAKE_EXECUTABLE_SUFFIX} 
    PATH
      $ENV{QTDIR}
      "C:/Qt/4.7.4/bin"
      "../Qt/4.7.4/bin"
      "../../Qt/4.7.4/bin"
      "C:/Qt/4.8.7/bin"
      "../Qt/4.8.7/bin"
      "../../Qt/4.8.7/bin"
    )

  FIND_PACKAGE(Qt4 QUIET)
  IF ( NOT QT4_FOUND )
    IF(EXISTS ${QT_QMAKE_EXECUTABLE} )
      GET_FILENAME_COMPONENT(QT_BINARY_DIR ${QT_QMAKE_EXECUTABLE} DIRECTORY)
      GET_FILENAME_COMPONENT(CMAKE_PREFIX_PATH ${QT_BINARY_DIR} DIRECTORY)
    ENDIF()

    FIND_PACKAGE(Qt5 QUIET COMPONENTS Widgets NO_MODULE)

    IF ( NOT Qt5_FOUND )
      MESSAGE( FATAL_ERROR "This project requires Qt4 or Qt5 for building PlusApp. These components are missing. Please verify configuration by selecting QT_QMAKE_EXECUTABLE (Qt4) or Qt5_DIR (Qt5) or turn off RobartsVTK_USE_QT option.")
    ENDIF()
    SET(QT_VERSION_MAJOR ${Qt5Widgets_VERSION_MAJOR})
    UNSET(QT_QMAKE_EXECUTABLE CACHE)
  ENDIF()
  
  SET(CMAKE_AUTOMOC ON)
ENDIF ()

#-----------------------------------------------------------------------------
# VTK configuration
IF(NOT VTK_INSTALL_ARCHIVE_DIR)
  SET(VTK_INSTALL_ARCHIVE_DIR lib)
ENDIF()
IF(NOT VTK_INSTALL_PACKAGE_DIR)
  SET(VTK_INSTALL_PACKAGE_DIR "lib/cmake/RobartsVTK-${RobartsVTK_MAJOR_VERSION}.${RobartsVTK_MINOR_VERSION}")
ENDIF()

SET (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${VTK_CMAKE_DIR})
INCLUDE(vtkExternalModuleMacros)

IF( VTK_FOUND AND VTK_WRAP_PYTHON )
  # Get the include directories for the module and all its dependencies.
  macro(vtk_include_recurse module)
    _vtk_module_config_recurse("${module}_PYTHON" ${module})
    INCLUDE_DIRECTORIES(${${module}_PYTHON_INCLUDE_DIRS})
  endmacro()

  INCLUDE(vtkModuleMacros)
  INCLUDE(vtkPythonWrapping)
  
  # Export location of python module dirs in install and build tree for every vtkpython module to use
  # As long as those modules depend on vtkpython, they can retrieve and use these
  IF(NOT VTK_INSTALL_PYTHON_MODULE_DIR)
    SET(VTK_INSTALL_PYTHON_MODULE_DIR
        "${CMAKE_INSTALL_PREFIX}/python${PYTHON_MAJOR_VERSION}.${PYTHON_MINOR_VERSION}/site-packages"
        CACHE
        PATH "Directory where python modules will be installed" FORCE)
    MARK_AS_ADVANCED(VTK_INSTALL_PYTHON_MODULE_DIR)
  ENDIF()

  IF(NOT VTK_BUILD_PYTHON_MODULE_DIR)
    SET(VTK_BUILD_PYTHON_MODULE_DIR
        "${RobartsVTK_BINARY_DIR}/Wrapping/Python"
        CACHE
        PATH "Directory where python modules will be built" FORCE)
    MARK_AS_ADVANCED(VTK_BUILD_PYTHON_MODULE_DIR)
  ENDIF()
  
  OPTION(RobartsVTK_CUDA_WRAP_PYTHON "Wrap CUDA projects in Python." OFF)
ENDIF()

#-----------------------------------------------------------------------------
# Look for an ITK build directory (optional)
#-----------------------------------------------------------------------------
OPTION(RobartsVTK_USE_ITK "Use ITK in RobartsITK" OFF)
IF(RobartsVTK_USE_ITK)
  # ITK has been built already
  FIND_PACKAGE(ITK REQUIRED NO_MODULE PATHS ${ITK_DIR} NO_DEFAULT_PATH)
  
  IF(ITK_FOUND)
    MESSAGE(STATUS "Using ITK available at: ${ITK_DIR}")

    IF ("${ITK_LIBRARY_DIRS}" STREQUAL "")
      # ITKv4 does not seem to specify ITK_LIBRARY_DIRS any more, so we generate the binary directory from the main ITK directory
      SET(ITK_LIBRARY_DIRS "${ITK_DIR}/bin")
    ENDIF()
    
    # Copy libraries to RobartsVTK_EXECUTABLE_OUTPUT_PATH
    IF ( ${CMAKE_GENERATOR} MATCHES "Visual Studio" )
      FOREACH(lib ${ITK_LIBRARIES})
        GET_TARGET_PROPERTY(ITK_DEBUG_FILE ${lib} IMPORTED_LOCATION_DEBUG)
        GET_TARGET_PROPERTY(ITK_RELEASE_FILE ${lib} IMPORTED_LOCATION_RELEASE)

        IF( EXISTS ${ITK_RELEASE_FILE} )
          FILE(COPY ${ITK_RELEASE_FILE} DESTINATION ${RobartsVTK_EXECUTABLE_OUTPUT_PATH}/Release)
        ENDIF()
        IF( EXISTS ${ITK_DEBUG_FILE} )
          FILE(COPY ${ITK_DEBUG_FILE} DESTINATION ${RobartsVTK_EXECUTABLE_OUTPUT_PATH}/Debug)
        ENDIF()
      ENDFOREACH()
    ELSE()
      FILE(COPY 
        ${ITK_LIBRARY_DIRS}/
        DESTINATION ${RobartsVTK_EXECUTABLE_OUTPUT_PATH}
        FILES_MATCHING REGEX .*${CMAKE_SHARED_LIBRARY_SUFFIX}
        )
    ENDIF()
    SET (RobartsVTK_ITK_DIR "${ITK_DIR}" CACHE INTERNAL "Path to store itk binaries")

    INCLUDE(${ITK_USE_FILE})
  ENDIF()
ENDIF()

#-----------------------------------------------------------------------------
# Look for a LibXml2 (optional)
#-----------------------------------------------------------------------------
OPTION(RobartsVTK_USE_LIBXML2 "Use LibXml2 in RobartsVTK" ON)
IF(RobartsVTK_USE_LIBXML2)
  FIND_PACKAGE(LibXml2 QUIET)
  IF(LIBXML2_FOUND)
    INCLUDE_DIRECTORIES(${LIBXML2_INCLUDE_DIR})
    LINK_LIBRARIES(${LIBXML2_LIBRARIES})
    ADD_DEFINITIONS(${LIBXML2_DEFINITIONS})
  ENDIF()
ENDIF()

#-----------------------------------------------------------------------------
# Choose static or shared libraries.
#-----------------------------------------------------------------------------
OPTION(BUILD_SHARED_LIBS "Build shared libraries." OFF)
SET (RobartsVTK_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
IF (BUILD_SHARED_LIBS)
  SET (BUILD_SHARED_LIBS_CONFIG 1)
ELSE (BUILD_SHARED_LIBS)
  SET (BUILD_SHARED_LIBS_CONFIG 0)
ENDIF (BUILD_SHARED_LIBS)

#-----------------------------------------------------------------------------
# Build documentation
#-----------------------------------------------------------------------------
OPTION(BUILD_DOCUMENTATION "Build Documentation" OFF)
IF(BUILD_DOCUMENTATION)
  ADD_SUBDIRECTORY(Documentation/Doxygen)
ENDIF()

#-----------------------------------------------------------------------------
# Output directories.
#-----------------------------------------------------------------------------
SET(LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR}/bin CACHE INTERNAL "Single output directory for building all libraries.")
SET(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR}/bin CACHE INTERNAL "Single output directory for building all executables.")
SET(RobartsVTK_LIBRARY_DIR ${LIBRARY_OUTPUT_PATH}/${CMAKE_CFG_INTDIR})
SET(RobartsVTK_EXECUTABLE_DIR ${EXECUTABLE_OUTPUT_PATH}/${CMAKE_CFG_INTDIR})
SET(CXX_TEST_PATH ${EXECUTABLE_OUTPUT_PATH})

SET(RobartsVTK_Export_Template ${CMAKE_CURRENT_SOURCE_DIR}/RobartsVTKExport.h.in)

#-----------------------------------------------------------------------------
# Configure Dart testing support.
#-----------------------------------------------------------------------------
INCLUDE(${CMAKE_ROOT}/Modules/Dart.cmake)

# ---------------------------------------------------------------
# Initialize configuration

#-----------------------------------------------------------------------------
# Libraries
SET(RobartsVTK_LIBRARIES)

#-----------------------------------------------------------------------------
# Include directories
SET(RobartsVTK_INCLUDE_DIRS CACHE INTERNAL "")

# ---------------------------------------------------------------
# Start listing the RobartsVTK kits
#-----------------------------------------------------------------------------
OPTION(RobartsVTK_USE_PLUS "Use the PLUSToolkit for data collection." OFF)
IF(RobartsVTK_USE_PLUS)
  # PlusLib has been built already
  FIND_PACKAGE(PlusLib REQUIRED NO_MODULE PATHS ${PlusLib_DIR} NO_DEFAULT_PATH)
  
  IF(ITK_FOUND)
    MESSAGE(STATUS "Using PlusLib available at: ${PlusLib_DIR}")
    
    # Copy libraries to RobartsVTK_EXECUTABLE_OUTPUT_PATH
    IF ( ${CMAKE_GENERATOR} MATCHES "Visual Studio" )
      FOREACH(lib ${PlusLib_LIBRARIES})
        GET_TARGET_PROPERTY(PlusLib_DEBUG_FILE ${lib} IMPORTED_LOCATION_DEBUG)
        GET_TARGET_PROPERTY(PlusLib_RELEASE_FILE ${lib} IMPORTED_LOCATION_RELEASE)

        IF( EXISTS ${PlusLib_DEBUG_FILE} )
          FILE(COPY ${PlusLib_DEBUG_FILE} DESTINATION ${RobartsVTK_EXECUTABLE_OUTPUT_PATH}/Release)
        ENDIF()
        IF( EXISTS ${PlusLib_RELEASE_FILE} )
          FILE(COPY ${PlusLib_RELEASE_FILE} DESTINATION ${RobartsVTK_EXECUTABLE_OUTPUT_PATH}/Debug)
        ENDIF()
      ENDFOREACH()
    ELSE()
      FILE(COPY 
        ${PlusLib_LIBRARY_DIRS}/
        DESTINATION ${RobartsVTK_EXECUTABLE_OUTPUT_PATH}
        FILES_MATCHING REGEX .*${CMAKE_SHARED_LIBRARY_SUFFIX}
        )
    ENDIF()
    SET (RobartsVTK_PlusLib_DIR "${PlusLib_DIR}" CACHE INTERNAL "Path to store PlusLib binaries")

    INCLUDE(${PlusLib_USE_FILE})
  ENDIF()
ENDIF()

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})

#-----------------------------------------------------------------------------
# vtkRobartsRegistration
#-----------------------------------------------------------------------------
OPTION (RobartsVTK_USE_REGISTRATION "Build the vtkRobartsRegistration module" OFF)
IF (RobartsVTK_USE_REGISTRATION)
  OPTION(RobartsVTK_Include_Outdated_Registration "Include classes that still use VTK4 pipeline style." OFF)
  ADD_SUBDIRECTORY(ImageProcessing/vtkRobartsRegistration)
  SET(RobartsVTK_INCLUDE_DIRS ${RobartsVTK_INCLUDE_DIRS} ${vtkRobartsRegistration_INCLUDE_DIRS} CACHE INTERNAL "" )
ENDIF ()

#-----------------------------------------------------------------------------
# vtkRobartsCommon - contains common utilities
#-----------------------------------------------------------------------------
OPTION (RobartsVTK_USE_COMMON "Build the vtkRobartsCommon module" ON)
IF (RobartsVTK_USE_COMMON)
  ADD_SUBDIRECTORY (Common/vtkRobartsCommon)
  SET(RobartsVTK_INCLUDE_DIRS ${RobartsVTK_INCLUDE_DIRS} ${vtkRobartsCommon_INCLUDE_DIRS} CACHE INTERNAL "" )
ENDIF()

#-----------------------------------------------------------------------------
# CUDA Dependencies
#-----------------------------------------------------------------------------
OPTION (RobartsVTK_USE_CUDA "Build the CUDA modules" OFF)
IF (RobartsVTK_USE_CUDA)
  # Look for CUDA facilities
  FIND_PACKAGE(CUDA REQUIRED)
  
  INCLUDE(${CMAKE_CURRENT_SOURCE_DIR}/CMake/RobartsCUDAVTKModule.cmake)

  # Add Robart's CUDA libraries
  ADD_SUBDIRECTORY(Common/vtkCudaCommon)
  
  SET(RobartsVTK_INCLUDE_DIRS ${RobartsVTK_INCLUDE_DIRS} ${vtkCudaCommon_INCLUDE_DIRS} CACHE INTERNAL "" )
ENDIF ()

#-----------------------------------------------------------------------------
# Add the CUDA Visualization Libraries
#-----------------------------------------------------------------------------
IF (RobartsVTK_USE_CUDA)
  OPTION (RobartsVTK_USE_CUDA_VISUALIZATION "Build the CUDA visualization modules" ON)
  IF (RobartsVTK_USE_CUDA_VISUALIZATION)
    ADD_SUBDIRECTORY(Visualization/vtkCudaVisualization)
    SET(RobartsVTK_INCLUDE_DIRS ${RobartsVTK_INCLUDE_DIRS} ${vtkCudaVisualization_INCLUDE_DIRS} CACHE INTERNAL "" )
  ENDIF()
ENDIF()

#-----------------------------------------------------------------------------
# Add the CUDA Analytics Libraries
#-----------------------------------------------------------------------------
IF (RobartsVTK_USE_CUDA AND RobartsVTK_USE_COMMON)
  OPTION (RobartsVTK_USE_CUDA_ANALYTICS "Build the CUDA image analytics modules" ON)
  IF (RobartsVTK_USE_CUDA_ANALYTICS)
    ADD_SUBDIRECTORY(ImageProcessing/vtkCudaImageAnalytics)
    SET(RobartsVTK_INCLUDE_DIRS ${RobartsVTK_INCLUDE_DIRS} ${vtkCudaImageAnalytics_INCLUDE_DIRS} CACHE INTERNAL "" )
    ADD_SUBDIRECTORY(Utilities/MaxFlow)
  ENDIF()
ENDIF()

#-----------------------------------------------------------------------------
# Add RobartsVTK.py.in file.  On Win32, this has to go into all config
#-----------------------------------------------------------------------------
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
#-----------------------------------------------------------------------------
OPTION(RobartsVTK_BUILD_EXAMPLES "Build RobartsVTK examples." OFF)
IF(RobartsVTK_BUILD_EXAMPLES)
  IF(RobartsVTK_USE_CUDA)
    ADD_SUBDIRECTORY(Examples/Cuda)
    IF(RobartsVTK_USE_QT)
      ADD_SUBDIRECTORY(Examples/FluoroPredViz)
      ADD_SUBDIRECTORY(Examples/TransferFunctionGUI)
    ENDIF()
  ENDIF()

  IF(RobartsVTK_USE_COMMON AND RobartsVTK_USE_CUDA AND RobartsVTK_USE_CUDA_ANALYTICS)
    ADD_SUBDIRECTORY(Examples/MaxFlow)
  ENDIF()
ENDIF()

#-----------------------------------------------------------------------------
# Exports
#-----------------------------------------------------------------------------
CONFIGURE_FILE(
  ${CMAKE_CURRENT_SOURCE_DIR}/RobartsVTKConfigure.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/RobartsVTKConfigure.h
  )

INCLUDE(${RobartsVTK_SOURCE_DIR}/CMake/RobartsVTKGenerateConfig.cmake)