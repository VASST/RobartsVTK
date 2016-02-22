PROJECT(RobartsVTK)

# todo module export doesn't have dllexport/import - build_shared_libs?
# todo cuda module dll export?

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

# Set up our directory structure for output libraries and binaries
if(NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY)
  SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
ENDIF()
IF(NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)
  IF(UNIX)
    SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
  ELSE()
    SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
  ENDIF()
ENDIF()
IF(NOT CMAKE_ARCHIVE_OUTPUT_DIRECTORY)
  SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
ENDIF()
MARK_AS_ADVANCED(CMAKE_RUNTIME_OUTPUT_DIRECTORY
  CMAKE_LIBRARY_OUTPUT_DIRECTORY
  CMAKE_ARCHIVE_OUTPUT_DIRECTORY
  )

# Set all these VTK variables to use the VTK modules system
IF(NOT VTK_INSTALL_RUNTIME_DIR)
  SET(VTK_INSTALL_RUNTIME_DIR bin)
ENDIF()
IF(NOT VTK_INSTALL_LIBRARY_DIR)
  SET(VTK_INSTALL_LIBRARY_DIR lib)
ENDIF()
IF(NOT VTK_INSTALL_ARCHIVE_DIR)
  SET(VTK_INSTALL_ARCHIVE_DIR lib)
ENDIF()
IF(NOT VTK_INSTALL_PACKAGE_DIR)
  SET(VTK_INSTALL_PACKAGE_DIR "lib/cmake/RobartsVTK-${RobartsVTK_MAJOR_VERSION}.${RobartsVTK_MINOR_VERSION}")
ENDIF()
IF(NOT VTK_INSTALL_INCLUDE_DIR)
  SET(VTK_INSTALL_INCLUDE_DIR include/RobartsVTK-${RobartsVTK_MAJOR_VERSION}.${RobartsVTK_MINOR_VERSION})
ENDIF()
IF(NOT VTK_INSTALL_DATA_DIR)
  SET(VTK_INSTALL_DATA_DIR share/RobartsVTK-${RobartsVTK_MAJOR_VERSION}.${RobartsVTK_MINOR_VERSION})
ENDIF()
IF(NOT VTK_INSTALL_DOC_DIR)
  SET(VTK_INSTALL_DOC_DIR share/doc/RobartsVTK-${RobartsVTK_MAJOR_VERSION}.${RobartsVTK_MINOR_VERSION})
ENDIF()
IF(NOT VTK_INSTALL_EXPORT_NAME)
  SET(VTK_INSTALL_EXPORT_NAME RobartsVTKTargets)
ENDIF()
IF(NOT VTK_MODULES_DIR)
  SET(VTK_MODULES_DIR "${RobartsVTK_BINARY_DIR}/${VTK_INSTALL_PACKAGE_DIR}/Modules")
ENDIF()

#-----------------------------------------------------------------------------
# Add configure file to included directories
#-----------------------------------------------------------------------------
INCLUDE_DIRECTORIES(${CMAKE_BINARY_DIR})

#-----------------------------------------------------------------------------
# Look for a VTK build directory
#-----------------------------------------------------------------------------
FIND_PACKAGE(VTK 6.2.0 REQUIRED NO_MODULE PATHS ${VTK_DIR} NO_DEFAULT_PATH)

MESSAGE(STATUS "Using VTK available at: ${VTK_DIR}")

# Copy libraries to RobartsVTK_EXECUTABLE_OUTPUT_PATH
FOREACH(lib ${VTK_LIBRARIES})
  IF(NOT TARGET ${lib})
    continue()
  ENDIF()

  GET_TARGET_PROPERTY(VTK_DEBUG_FILE ${lib} IMPORTED_LOCATION_DEBUG)
  GET_TARGET_PROPERTY(VTK_RELEASE_FILE ${lib} IMPORTED_LOCATION_RELEASE)

  IF ( ${CMAKE_GENERATOR} MATCHES "Visual Studio" )
    IF( EXISTS ${VTK_RELEASE_FILE} )
      FILE(COPY ${VTK_RELEASE_FILE} DESTINATION ${RobartsVTK_EXECUTABLE_OUTPUT_PATH}/Release)
    ENDIF()
    IF( EXISTS ${VTK_DEBUG_FILE} )
      FILE(COPY ${VTK_DEBUG_FILE} DESTINATION ${RobartsVTK_EXECUTABLE_OUTPUT_PATH}/Debug)
    ENDIF()
  ELSE()
    IF( VTK_DEBUG_FILE EQUAL VTK_RELEASE_FILE AND EXISTS ${VTK_RELEASE_FILE})
      FILE(COPY ${VTK_RELEASE_FILE} DESTINATION ${RobartsVTK_EXECUTABLE_OUTPUT_PATH})
    ELSE()
      IF( EXISTS ${VTK_RELEASE_FILE} )
        FILE(COPY ${VTK_RELEASE_FILE} DESTINATION ${RobartsVTK_EXECUTABLE_OUTPUT_PATH})
      ENDIF()
      IF( EXISTS ${VTK_DEBUG_FILE} )
        FILE(COPY ${VTK_DEBUG_FILE} DESTINATION ${RobartsVTK_EXECUTABLE_OUTPUT_PATH})
      ENDIF()
    ENDIF()
  ENDIF()
ENDFOREACH()

#-----------------------------------------------------------------------------
# Include the important stuff for using VTK
INCLUDE (${VTK_USE_FILE})
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# Confirm that VTK was built with Qt if requested
#-----------------------------------------------------------------------------
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

#-----------------------------------------------------------------------------
# Setup python wrapping details
#-----------------------------------------------------------------------------
OPTION(RobartsVTK_WRAP_PYTHON "Wrap the Robarts classes in Python." OFF)
IF( RobartsVTK_WRAP_PYTHON AND VTK_WRAP_PYTHON)
  FIND_PACKAGE(PythonLibs REQUIRED)
  
  # Add the vtkpython executable target so that we can properly depend on it
  add_executable(vtkpython IMPORTED)
  # Import target "vtkpython" for configuration "Debug"
  set_property(TARGET vtkpython APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
  set_target_properties(vtkpython PROPERTIES
    IMPORTED_LOCATION_DEBUG "${VTK_BINARY_DIR}/Debug/vtkpython.exe"
    )
  # Import target "vtkpython" for configuration "Release"
  set_property(TARGET vtkpython APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
  set_target_properties(vtkpython PROPERTIES
    IMPORTED_LOCATION_RELEASE "${VTK_BINARY_DIR}/Release/vtkpython.exe"
    )
    
  # Import target "vtkpython" for configuration "MinSizeRel"
  set_property(TARGET vtkpython APPEND PROPERTY IMPORTED_CONFIGURATIONS MINSIZEREL)
  set_target_properties(vtkpython PROPERTIES
    IMPORTED_LOCATION_MINSIZEREL "${VTK_BINARY_DIR}/MinSizeRel/vtkpython.exe"
    )
    
  # Import target "vtkpython" for configuration "RelWithDebInfo"
  set_property(TARGET vtkpython APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
  set_target_properties(vtkpython PROPERTIES
    IMPORTED_LOCATION_RELWITHDEBINFO "${VTK_BINARY_DIR}/RelWithDebInfo/vtkpython.exe"
    )

  # Export location of python module dirs in install and build tree for every vtkpython module to use
  # As long as those modules depend on vtkpython, they can retrieve and use these
  IF(NOT VTK_INSTALL_PYTHON_MODULE_DIR)
    SET(VTK_INSTALL_PYTHON_MODULE_DIR
        "${VTK_INSTALL_LIBRARY_DIR}/python${PYTHON_MAJOR_VERSION}.${PYTHON_MINOR_VERSION}/site-packages"
        CACHE
        PATH "Directory where python modules will be installed" FORCE)
    MARK_AS_ADVANCED(VTK_INSTALL_PYTHON_MODULE_DIR)
  ENDIF()

  # this is only necessary if there is a module written in python
  IF(NOT VTK_BUILD_PYTHON_MODULE_DIR)
    SET(VTK_BUILD_PYTHON_MODULE_DIR
        "Wrapping/Python"
        CACHE
        PATH "Directory where python modules will be built" FORCE)
    MARK_AS_ADVANCED(VTK_BUILD_PYTHON_MODULE_DIR)
  ENDIF()
ENDIF()

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

  IF(NOT RobartsVTK_USE_QT5_ONLY)
    FIND_PACKAGE(Qt4 QUIET)
  ENDIF()

  IF ( NOT QT4_FOUND )
    IF(EXISTS ${QT_QMAKE_EXECUTABLE} )
      GET_FILENAME_COMPONENT(QT_BINARY_DIR ${QT_QMAKE_EXECUTABLE} DIRECTORY)
      GET_FILENAME_COMPONENT(QT_ROOT_DIR ${QT_BINARY_DIR} DIRECTORY)
      SET(Qt5_DIR ${QT_ROOT_DIR}/lib/cmake/Qt5 CACHE PATH "Path to Qt5Config.cmake" FORCE)
    ENDIF()

    FIND_PACKAGE(Qt5 QUIET COMPONENTS Widgets NO_MODULE)

    IF ( NOT Qt5_FOUND )
      MESSAGE( FATAL_ERROR "This project requires Qt4 or Qt5 for building PlusApp. These components are missing. Please verify configuration by selecting QT_QMAKE_EXECUTABLE (Qt4) or Qt5_DIR (Qt5) or turn off RobartsVTK_USE_QT option.")
    ENDIF()
    SET(QT_VERSION_MAJOR ${Qt5Widgets_VERSION_MAJOR})
  ENDIF()
  
  SET(CMAKE_AUTOMOC ON)
  IF( QT_QMAKE_EXECUTABLE STREQUAL "NOTFOUND" AND Qt5_DIR )
    GET_FILENAME_COMPONENT(QT_CMAKE_DIR ${Qt5_DIR} DIRECTORY)
    GET_FILENAME_COMPONENT(QT_LIB_DIR ${QT_CMAKE_DIR} DIRECTORY)
    GET_FILENAME_COMPONENT(CMAKE_PREFIX_PATH ${QT_LIB_DIR} DIRECTORY)
    SET(QT_BINARY_DIR ${CMAKE_PREFIX_PATH}/bin)
    SET(QT_QMAKE_EXECUTABLE ${QT_BINARY_DIR}/qmake${CMAKE_EXECUTABLE_SUFFIX} CACHE FILEPATH "Location of Qt QMake executable." FORCE)
  ELSEIF( NOT QT_QMAKE_EXECUTABLE STREQUAL "NOTFOUND" )
    GET_FILENAME_COMPONENT(QT_BINARY_DIR ${QT_QMAKE_EXECUTABLE} DIRECTORY)
  ENDIF()
  
  # --------------------------------------------------------------------------
  # Copy Qt binaries to RobartsVTK_EXECUTABLE_OUTPUT_PATH

  # Determine shared library extension without the dot (dll instead of .dll)
  STRING(SUBSTRING ${CMAKE_SHARED_LIBRARY_SUFFIX} 1 -1 CMAKE_SHARED_LIBRARY_SUFFIX_NO_SEPARATOR)

  # Get all Qt shared library names
  IF( ${QT_VERSION_MAJOR} AND ${QT_VERSION_MAJOR} EQUAL 5 )
    SET(RELEASE_REGEX_PATTERN .t5.*[^d][.]${CMAKE_SHARED_LIBRARY_SUFFIX_NO_SEPARATOR})
    SET(DEBUG_REGEX_PATTERN .t5.*d[.]${CMAKE_SHARED_LIBRARY_SUFFIX_NO_SEPARATOR})
  ELSE()
    SET(RELEASE_REGEX_PATTERN .*[^d]4[.]${CMAKE_SHARED_LIBRARY_SUFFIX_NO_SEPARATOR} )
    SET(DEBUG_REGEX_PATTERN .*d4[.]${CMAKE_SHARED_LIBRARY_SUFFIX_NO_SEPARATOR} )
  ENDIF()

  # Copy shared libraries to bin directory to allow running Plus applications in the build tree
  IF ( ${CMAKE_GENERATOR} MATCHES "Visual Studio" )
    FILE(COPY "${QT_BINARY_DIR}/"
      DESTINATION ${RobartsVTK_EXECUTABLE_OUTPUT_PATH}/Release
      FILES_MATCHING REGEX ${RELEASE_REGEX_PATTERN}
      )
    FILE(COPY "${QT_BINARY_DIR}/"
      DESTINATION ${RobartsVTK_EXECUTABLE_OUTPUT_PATH}/Debug
      FILES_MATCHING REGEX ${DEBUG_REGEX_PATTERN}
      )
  ELSE()
    FILE(COPY "${QT_BINARY_DIR}/"
      DESTINATION ${RobartsVTK_EXECUTABLE_OUTPUT_PATH}
      FILES_MATCHING REGEX .*${CMAKE_SHARED_LIBRARY_SUFFIX}
      )
  ENDIF()
ENDIF ()

#-----------------------------------------------------------------------------
# Load the necessary macros to use the VTK module environment
#-----------------------------------------------------------------------------
SET (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${VTK_CMAKE_DIR})
INCLUDE(vtkExternalModuleMacros)

#-----------------------------------------------------------------------------
# Look for an ITK build directory (optional)
#-----------------------------------------------------------------------------
OPTION(RobartsVTK_USE_ITK "Use ITK in RobartsITK" OFF)
IF(RobartsVTK_USE_ITK)
  # ITK has been built already
  FIND_PACKAGE(ITK REQUIRED NO_MODULE PATHS ${ITK_DIR} NO_DEFAULT_PATH)
  
  IF(ITK_FOUND)
    MESSAGE(STATUS "Using ITK available at: ${ITK_DIR}")
    
    # Copy libraries to RobartsVTK_EXECUTABLE_OUTPUT_PATH
    FOREACH(lib ${ITK_LIBRARIES})
      IF(NOT TARGET ${lib})
        continue()
      ENDIF()

      GET_TARGET_PROPERTY(ITK_DEBUG_FILE ${lib} IMPORTED_LOCATION_DEBUG)
      GET_TARGET_PROPERTY(ITK_RELEASE_FILE ${lib} IMPORTED_LOCATION_RELEASE)

      IF ( ${CMAKE_GENERATOR} MATCHES "Visual Studio" )
        IF( EXISTS ${ITK_RELEASE_FILE} )
          FILE(COPY ${ITK_RELEASE_FILE} DESTINATION ${RobartsVTK_EXECUTABLE_OUTPUT_PATH}/Release)
        ENDIF()
        IF( EXISTS ${ITK_DEBUG_FILE} )
          FILE(COPY ${ITK_DEBUG_FILE} DESTINATION ${RobartsVTK_EXECUTABLE_OUTPUT_PATH}/Debug)
        ENDIF()
      ELSE()
        IF( ITK_DEBUG_FILE EQUAL ITK_RELEASE_FILE AND EXISTS ${ITK_RELEASE_FILE})
          FILE(COPY ${ITK_RELEASE_FILE} DESTINATION ${RobartsVTK_EXECUTABLE_OUTPUT_PATH})
        ELSE()
          IF( EXISTS ${ITK_RELEASE_FILE} )
            FILE(COPY ${ITK_RELEASE_FILE} DESTINATION ${RobartsVTK_EXECUTABLE_OUTPUT_PATH})
          ENDIF()
          IF( EXISTS ${ITK_DEBUG_FILE} )
            FILE(COPY ${ITK_DEBUG_FILE} DESTINATION ${RobartsVTK_EXECUTABLE_OUTPUT_PATH})
          ENDIF()
        ENDIF()
      ENDIF()
    ENDFOREACH()
 
    SET (RobartsVTK_ITK_DIR "${ITK_DIR}" CACHE INTERNAL "Path to store itk binaries")

    INCLUDE(${ITK_USE_FILE})
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
SET(LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR}/lib CACHE INTERNAL "Single output directory for building all libraries.")
SET(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR}/bin CACHE INTERNAL "Single output directory for building all executables.")
SET(RobartsVTK_LIBRARY_DIR ${LIBRARY_OUTPUT_PATH}/${CMAKE_CFG_INTDIR})
SET(RobartsVTK_EXECUTABLE_DIR ${EXECUTABLE_OUTPUT_PATH}/${CMAKE_CFG_INTDIR})
SET(CXX_TEST_PATH ${EXECUTABLE_OUTPUT_PATH})

SET(RobartsVTK_Export_Template ${CMAKE_CURRENT_SOURCE_DIR}/RobartsVTKExport.h.in)

#-----------------------------------------------------------------------------
# PLUS configuration (optional)
#-----------------------------------------------------------------------------
OPTION(RobartsVTK_USE_PLUS "Use the PLUSToolkit for data collection." OFF)
IF(RobartsVTK_USE_PLUS)
    # PlusLib has been built already
  FIND_PACKAGE(PlusLib REQUIRED NO_MODULE PATHS ${PlusLib_DIR} NO_DEFAULT_PATH)
  
  IF(PlusLib_FOUND)
    MESSAGE(STATUS "Using PlusLib available at: ${PlusLib_DIR}")

    # Copy libraries to RobartsVTK_EXECUTABLE_OUTPUT_PATH
    FOREACH(lib ${PlusLib_LIBRARIES})
      IF(NOT TARGET ${lib})
        continue()
      ENDIF()

      GET_TARGET_PROPERTY(PlusLib_DEBUG_FILE ${lib} IMPORTED_LOCATION_DEBUG)
      GET_TARGET_PROPERTY(PlusLib_RELEASE_FILE ${lib} IMPORTED_LOCATION_RELEASE)

      IF ( ${CMAKE_GENERATOR} MATCHES "Visual Studio" )
        IF( EXISTS ${PlusLib_RELEASE_FILE} )
          FILE(COPY ${PlusLib_RELEASE_FILE} DESTINATION ${RobartsVTK_EXECUTABLE_OUTPUT_PATH}/Release)
        ENDIF()
        IF( EXISTS ${PlusLib_DEBUG_FILE} )
          FILE(COPY ${PlusLib_DEBUG_FILE} DESTINATION ${RobartsVTK_EXECUTABLE_OUTPUT_PATH}/Debug)
        ENDIF()
      ELSE()
        IF( PlusLib_DEBUG_FILE EQUAL PlusLib_RELEASE_FILE AND EXISTS ${PlusLib_RELEASE_FILE})
          FILE(COPY ${PlusLib_RELEASE_FILE} DESTINATION ${RobartsVTK_EXECUTABLE_OUTPUT_PATH})
        ELSE()
          IF( EXISTS ${PlusLib_RELEASE_FILE} )
            FILE(COPY ${PlusLib_RELEASE_FILE} DESTINATION ${RobartsVTK_EXECUTABLE_OUTPUT_PATH})
          ENDIF()
          IF( EXISTS ${PlusLib_DEBUG_FILE} )
            FILE(COPY ${PlusLib_DEBUG_FILE} DESTINATION ${RobartsVTK_EXECUTABLE_OUTPUT_PATH})
          ENDIF()
        ENDIF()
      ENDIF()
    ENDFOREACH()
  
    INCLUDE(${PlusLib_USE_FILE})
  ENDIF()
ENDIF()

#-----------------------------------------------------------------------------
# Enable inclusion of global configuration files in all projects (RobartsVTKConfigure.h)
#-----------------------------------------------------------------------------
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})

#-----------------------------------------------------------------------------
# Start listing the RobartsVTK modules
#-----------------------------------------------------------------------------
SET(RobartsVTK_LIBRARIES CACHE INTERNAL "")
SET(RobartsVTK_INCLUDE_DIRS ${CMAKE_CURRENT_BINARY_DIR} CACHE INTERNAL "")

#-----------------------------------------------------------------------------
# vtkRobartsCommon - contains common utilities
#-----------------------------------------------------------------------------
OPTION (RobartsVTK_USE_COMMON "Build the vtkRobartsCommon module" ON)
IF (RobartsVTK_USE_COMMON)
  ADD_SUBDIRECTORY (Common/vtkRobartsCommon)
  SET(RobartsVTK_INCLUDE_DIRS ${RobartsVTK_INCLUDE_DIRS} ${vtkRobartsCommon_INCLUDE_DIRS} CACHE INTERNAL "" )
  SET(RobartsVTK_LIBRARIES ${RobartsVTK_LIBRARIES} vtkRobartsCommon CACHE INTERNAL "")
ENDIF()

#-----------------------------------------------------------------------------
# vtkRobartsRegistration
#-----------------------------------------------------------------------------
OPTION (RobartsVTK_USE_REGISTRATION "Build the vtkRobartsRegistration module" OFF)
IF (RobartsVTK_USE_REGISTRATION)
  OPTION(RobartsVTK_Include_Outdated_Registration "Include classes that still use VTK4 pipeline style." OFF)
  ADD_SUBDIRECTORY(ImageProcessing/vtkRobartsRegistration)
  SET(RobartsVTK_INCLUDE_DIRS ${RobartsVTK_INCLUDE_DIRS} ${vtkRobartsRegistration_INCLUDE_DIRS} CACHE INTERNAL "" )
  SET(RobartsVTK_LIBRARIES ${RobartsVTK_LIBRARIES} vtkRobartsRegistration CACHE INTERNAL "")
ENDIF ()

#-----------------------------------------------------------------------------
# CUDA Dependencies
#-----------------------------------------------------------------------------
OPTION (RobartsVTK_USE_CUDA "Build the CUDA modules" OFF)
IF (RobartsVTK_USE_CUDA)
  # Look for CUDA facilities
  FIND_PACKAGE(CUDA REQUIRED)

  # Add Robart's CUDA libraries
  ADD_SUBDIRECTORY(Common/vtkCudaCommon)
  
  SET(RobartsVTK_INCLUDE_DIRS ${RobartsVTK_INCLUDE_DIRS} ${vtkCudaCommon_INCLUDE_DIRS} CACHE INTERNAL "" )
  SET(RobartsVTK_LIBRARIES ${RobartsVTK_LIBRARIES} vtkCudaCommon CACHE INTERNAL "")
ENDIF ()

#-----------------------------------------------------------------------------
# Add the CUDA Visualization Libraries
#-----------------------------------------------------------------------------
IF (RobartsVTK_USE_CUDA)
  OPTION (RobartsVTK_USE_CUDA_VISUALIZATION "Build the CUDA visualization modules" ON)
  IF (RobartsVTK_USE_CUDA_VISUALIZATION)
    ADD_SUBDIRECTORY(Visualization/vtkCudaVisualization)
    SET(RobartsVTK_INCLUDE_DIRS ${RobartsVTK_INCLUDE_DIRS} ${vtkCudaVisualization_INCLUDE_DIRS} CACHE INTERNAL "" )
    SET(RobartsVTK_LIBRARIES ${RobartsVTK_LIBRARIES} vtkCudaVisualization CACHE INTERNAL "")
  ENDIF()
ENDIF()

#-----------------------------------------------------------------------------
# Add the CUDA Analytics Libraries
#-----------------------------------------------------------------------------
IF (RobartsVTK_USE_CUDA AND RobartsVTK_USE_COMMON)
  OPTION (RobartsVTK_USE_CUDA_ANALYTICS "Build the CUDA image analytics modules" ON)
  IF (RobartsVTK_USE_CUDA_ANALYTICS)
    ADD_SUBDIRECTORY(ImageProcessing/vtkCudaImageAnalytics)
    ADD_SUBDIRECTORY(Utilities/MaxFlow)
    SET(RobartsVTK_INCLUDE_DIRS ${RobartsVTK_INCLUDE_DIRS} ${vtkCudaImageAnalytics_INCLUDE_DIRS} CACHE INTERNAL "" )
    SET(RobartsVTK_LIBRARIES ${RobartsVTK_LIBRARIES} vtkCudaImageAnalytics CACHE INTERNAL "")
  ENDIF()
ENDIF()

#-----------------------------------------------------------------------------
# Examples
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
# Perform the actual python wrapping
#-----------------------------------------------------------------------------
IF(VTK_WRAP_PYTHON)
  # Use our clone of the VTK environment to do our python wrapping
  ADD_SUBDIRECTORY(Wrapping/Python)
ENDIF(VTK_WRAP_PYTHON)

#-----------------------------------------------------------------------------
# Generate build configurations
#-----------------------------------------------------------------------------
CONFIGURE_FILE(
  ${CMAKE_CURRENT_SOURCE_DIR}/RobartsVTKConfigure.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/RobartsVTKConfigure.h
  )

#-----------------------------------------------------------------------------
# Exports
#-----------------------------------------------------------------------------
SET(RobartsVTK_USE_FILE ${CMAKE_CURRENT_SOURCE_DIR}/UseRobartsVTK.cmake)

SET(RobartsVTK_TARGETS_FILE ${CMAKE_CURRENT_BINARY_DIR}/RobartsVTKTargets.cmake)
export(TARGETS ${RobartsVTK_LIBRARIES}
  FILE ${RobartsVTK_TARGETS_FILE}
  )
  
CONFIGURE_FILE(
  ${CMAKE_CURRENT_SOURCE_DIR}/RobartsVTKConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/RobartsVTKConfig.cmake)

