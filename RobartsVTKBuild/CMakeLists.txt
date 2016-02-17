PROJECT(RobartsVTKBuild)

# todo python wrapping
# todo module setup for cuda entries
# todo module export doesn't have dllexport/import - build_shared_libs?

# CMake version requirements
CMAKE_MINIMUM_REQUIRED(VERSION 3.2)
cmake_policy(SET CMP0003 NEW)

SET(RobartsVTKLib_PROJECT_ARGS)
SET(RobartsVTKLib_DEPENDENCIES)

# RobartsVTK Options
OPTION(RobartsVTK_USE_QT "Build with QT enabled" OFF)
OPTION(RobartsVTK_USE_ITK "Use ITK in RobartsITK" OFF)
OPTION(RobartsVTK_USE_LIBXML2 "Use LibXml2 in RobartsVTK" ON)
OPTION(RobartsVTK_USE_PLUS "Use the PLUSToolkit for data collection." OFF)
OPTION(RobartsVTK_USE_REGISTRATION "Build the vtkRobartsRegistration module" OFF)
IF (RobartsVTK_USE_REGISTRATION)
  OPTION(RobartsVTK_Include_Outdated_Registration "Include classes that still use VTK4 pipeline style." OFF)
ENDIF ()
OPTION(RobartsVTK_USE_COMMON "Build the vtkRobartsCommon module" ON)
OPTION(RobartsVTK_USE_CUDA "Build the CUDA modules" OFF)
IF (RobartsVTK_USE_CUDA)
  OPTION(RobartsVTK_USE_CUDA_VISUALIZATION "Build the CUDA visualization modules" ON)
ENDIF()
IF (RobartsVTK_USE_CUDA AND RobartsVTK_USE_COMMON)
  OPTION(RobartsVTK_USE_CUDA_ANALYTICS "Build the CUDA image analytics modules" ON)
ENDIF()
OPTION(RobartsVTK_BUILD_EXAMPLES "Build RobartsVTK examples." OFF)
OPTION(RobartsVTK_WRAP_PYTHON "Wrap the RobartsVTK libs in Python." ON)

#-----------------------------------------------------------------------------
# Preparatory setup
#-----------------------------------------------------------------------------
SET(Plus_DEPENDENCIES)
INCLUDE(ExternalProject)

#-----------------------------------------------------------------------------
# SVN - Let's check if a valid version of SVN is available
#-----------------------------------------------------------------------------
# Need subversion for the automatic update of the repository
MARK_AS_ADVANCED(CLEAR Subversion_SVN_EXECUTABLE)
FIND_FILE(Subversion_SVN_EXECUTABLE svn${CMAKE_EXECUTABLE_SUFFIX} 
  PATHS 
  "c:/Program Files/TortoiseSVN/bin/"
  "c:/Program Files/SlikSvn/bin/"
  )
FIND_PACKAGE(Subversion REQUIRED)

#-----------------------------------------------------------------------------
# ASSEMBLA - Set username and password for checking out Plus 
# Anonymous user name for Plus assembla server: https://subversion.assembla.com/svn/plus/
#-----------------------------------------------------------------------------
SET(PLUS_ASSEMBLA_USERNAME "perklab_anonymous" )
SET(PLUS_ASSEMBLA_PASSWORD "anonymous" )

#-----------------------------------------------------------------------------
# GIT - Let's check if a valid version of GIT is available
#-----------------------------------------------------------------------------
SET(GIT_PROTOCOL "git")

MARK_AS_ADVANCED(CLEAR GIT_EXECUTABLE)
FIND_FILE(GIT_EXECUTABLE git${CMAKE_EXECUTABLE_SUFFIX} 
  PATHS 
    "c:/Program Files/Git/bin/"
    "c:/Program Files (x86)/Git/bin/"
  )
FIND_PACKAGE(Git)
IF(NOT GIT_FOUND)
  MESSAGE(FATAL_ERROR "ERROR: Install Git and try to re-configure")
ENDIF()

#-----------------------------------------------------------------------------
# Set up python choices to pass to VTK
#-----------------------------------------------------------------------------
SET(RobartsVTK_VTK_PYTHON_ARGS)
IF( RobartsVTK_WRAP_PYTHON )
  FIND_PACKAGE(PythonLibs REQUIRED)
  
  MARK_AS_ADVANCED(CLEAR PYTHON_INCLUDE_DIR)
  MARK_AS_ADVANCED(CLEAR PYTHON_LIBRARY)
  IF( PythonLibs_FOUND )
    MARK_AS_ADVANCED(CLEAR PYTHON_INCLUDE_DIR)
    MARK_AS_ADVANCED(CLEAR PYTHON_LIBRARY)
    GET_FILENAME_COMPONENT(PYTHON_DIR ${PYTHON_INCLUDE_DIR} DIRECTORY)
    FIND_FILE(PYTHON_EXECUTABLE python${CMAKE_EXECUTABLE_SUFFIX} 
      PATHS
      ${PYTHON_DIR}
      NO_SYSTEM_ENVIRONMENT_PATH
    )

    SET(RobartsVTK_VTK_PYTHON_ARGS ${RobartsVTK_VTK_PYTHON_ARGS} -DPYTHON_INCLUDE_DIR:PATH=${PYTHON_INCLUDE_DIR} -DPYTHON_LIBRARY:FILEPATH=${PYTHON_LIBRARY} -DPYTHON_EXECUTABLE:FILEPATH=${PYTHON_EXECUTABLE})
  ENDIF()
ENDIF()

#-----------------------------------------------------------------------------
# Qt comes first so we know which version to pass to VTK
#-----------------------------------------------------------------------------
IF(RobartsVTK_USE_QT)
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
    IF(EXISTS ${QT_QMAKE_EXECUTABLE} AND NOT Qt5_DIR)
      GET_FILENAME_COMPONENT(ROBARTS_QT_BIN_DIR ${QT_QMAKE_EXECUTABLE} DIRECTORY)
      GET_FILENAME_COMPONENT(ROBARTS_QT_TOP_DIR ${ROBARTS_QT_BIN_DIR} DIRECTORY)
      GET_FILENAME_COMPONENT(CMAKE_PREFIX_PATH ${ROBARTS_QT_BIN_DIR} DIRECTORY)
      SET(Qt5_DIR ${ROBARTS_QT_TOP_DIR}/lib/cmake/Qt5)
    ENDIF()

    FIND_PACKAGE(Qt5 QUIET COMPONENTS Widgets NO_MODULE)

    IF( ${QT_QMAKE_EXECUTABLE} STREQUAL "NOTFOUND" AND NOT ${Qt5_DIR} STREQUAL "Qt5_DIR-NOTFOUND")
      GET_FILENAME_COMPONENT(ROBARTS_QT_CMAKE_DIR ${Qt5_DIR} DIRECTORY)
      GET_FILENAME_COMPONENT(ROBARTS_QT_LIB_DIR ${ROBARTS_QT_CMAKE_DIR} DIRECTORY)
      GET_FILENAME_COMPONENT(ROBARTS_QT_TOP_DIR ${ROBARTS_QT_LIB_DIR} DIRECTORY)
      SET(QT_QMAKE_EXECUTABLE ${ROBARTS_QT_TOP_DIR}/bin/qmake${CMAKE_EXECUTABLE_SUFFIX} CACHE FILEPATH "Location of Qt qmake executable." FORCE)
    ENDIF()

    IF ( NOT Qt5_FOUND )
      MESSAGE( FATAL_ERROR "This project requires Qt4 or Qt5 for building PlusApp. These components are missing. Please verify configuration by selecting QT_QMAKE_EXECUTABLE (Qt4) or Qt5_DIR (Qt5) or turn off RobartsVTK_USE_QT option.")
    ENDIF()
    SET(QT_VERSION_MAJOR ${Qt5_VERSION_MAJOR})
  ENDIF()

  SET(RobartsVTKLib_PROJECT_ARGS ${RobartsVTKLib_PROJECT_ARGS} -DQT_QMAKE_EXECUTABLE:FILEPATH=${QT_QMAKE_EXECUTABLE})
  IF( QT4_FOUND )
    SET(RobartsVTKLib_DEPENDENCIES ${RobartsVTKLib_DEPENDENCIES} Qt4::QtCore)
  ELSE()
    SET(RobartsVTKLib_DEPENDENCIES ${RobartsVTKLib_DEPENDENCIES} Qt5::Widgets)
  ENDIF()
ENDIF ()

#-----------------------------------------------------------------------------
# External project build flags
#-----------------------------------------------------------------------------
SET(ADDITIONAL_C_FLAGS "" CACHE STRING "Additional C Flags")
MARK_AS_ADVANCED(ADDITIONAL_C_FLAGS)

SET(ADDITIONAL_CXX_FLAGS "/MP2" CACHE STRING "Additional CXX Flags")
MARK_AS_ADVANCED(ADDITIONAL_CXX_FLAGS)

SET(ep_common_c_flags "${CMAKE_C_FLAGS_INIT} ${ADDITIONAL_C_FLAGS}")
SET(ep_common_cxx_flags "${CMAKE_CXX_FLAGS_INIT} ${ADDITIONAL_CXX_FLAGS}")
SET(ep_dependency_DIR ${CMAKE_BINARY_DIR}/Deps)

#-----------------------------------------------------------------------------
# VTK
#-----------------------------------------------------------------------------
INCLUDE(External_VTK.cmake)

IF( VTK_DIR )
  SET(RobartsVTKLib_PROJECT_ARGS ${RobartsVTKLib_PROJECT_ARGS} -DVTK_DIR:PATH=${VTK_DIR})
ELSE()
  IF( RobartsVTK_USE_PLUS )
    SET(Plus_DEPENDENCIES ${Plus_DEPENDENCIES} vtk)
    SET(VTK_DIR ${RobartsVTK_VTK_DIR}) # to send to PLUS
  ENDIF()
  SET(RobartsVTKLib_DEPENDENCIES ${RobartsVTKLib_DEPENDENCIES} vtk)
  SET(RobartsVTKLib_PROJECT_ARGS ${RobartsVTKLib_PROJECT_ARGS} -DVTK_DIR:PATH=${RobartsVTK_VTK_DIR})
ENDIF()

#-----------------------------------------------------------------------------
# Look for an ITK build directory (optional)
#-----------------------------------------------------------------------------
OPTION(RobartsVTK_USE_ITK "Use ITK in RobartsITK" OFF)
IF(RobartsVTK_USE_ITK)
  INCLUDE(External_ITK.cmake)
  
  IF( ITK_DIR )
    SET(RobartsVTKLib_PROJECT_ARGS ${RobartsVTKLib_PROJECT_ARGS} -DITK_DIR:PATH=${ITK_DIR})
  ELSE()
    IF( RobartsVTK_USE_PLUS )
      SET(Plus_DEPENDENCIES ${Plus_DEPENDENCIES} itk)
      SET(ITK_DIR ${RobartsVTK_ITK_DIR}) # to send to PLUS
    ENDIF()

    SET(RobartsVTKLib_PROJECT_ARGS ${RobartsVTKLib_PROJECT_ARGS} -DITK_DIR:PATH=${RobartsVTK_ITK_DIR})
    SET(RobartsVTKLib_DEPENDENCIES ${RobartsVTKLib_DEPENDENCIES} itk)
  ENDIF()
ENDIF()

#-----------------------------------------------------------------------------
# Look for a LibXml2 (optional)
#-----------------------------------------------------------------------------
OPTION(RobartsVTK_USE_LIBXML2 "Use LibXml2 in RobartsVTK" ON)
IF(RobartsVTK_USE_LIBXML2)
  FIND_PACKAGE(LibXml2 QUIET)
  
  IF( NOT LibXml2_FOUND )
    MESSAGE(STATUS "LibXml2 requested but was not found. Please set LibXml2_DIR and try again.")
  ELSE()
    SET(RobartsVTKLib_PROJECT_ARGS ${RobartsVTKLib_PROJECT_ARGS} -DLibXml2_DIR:PATH=${LibXml2_DIR})
    SET(RobartsVTKLib_DEPENDENCIES ${RobartsVTKLib_DEPENDENCIES} LibXml2)
  ENDIF()
ENDIF()

#-----------------------------------------------------------------------------
# Choose static or shared libraries.
#-----------------------------------------------------------------------------
OPTION(BUILD_SHARED_LIBS "Build shared libraries." OFF)
SET(RobartsVTKLib_PROJECT_ARGS ${RobartsVTKLib_PROJECT_ARGS} -DBUILD_SHARED_LIBS:BOOL=${BUILD_SHARED_LIBS})

#-----------------------------------------------------------------------------
# Build testing
#-----------------------------------------------------------------------------
OPTION(BUILD_TESTING "Build testing." OFF)
SET(RobartsVTKLib_PROJECT_ARGS ${RobartsVTKLib_PROJECT_ARGS} -DBUILD_TESTING:BOOL=${BUILD_TESTING})

#-----------------------------------------------------------------------------
# Build documentation
#-----------------------------------------------------------------------------
OPTION(BUILD_DOCUMENTATION "Build Documentation" OFF)
SET(RobartsVTKLib_PROJECT_ARGS ${RobartsVTKLib_PROJECT_ARGS} -DBUILD_DOCUMENTATION:BOOL=${BUILD_DOCUMENTATION})

#-----------------------------------------------------------------------------
# PLUS for data collection
#-----------------------------------------------------------------------------
IF(RobartsVTK_USE_PLUS)

  IF ( NOT VTK_DIR )
    # VTK_DIR is not supplied, so it is built inside Plus, therefore we need to specify dependencies to make sure it is built early enough
    SET(VTK_DEPENDENCIES)
    LIST(APPEND PlusLib_DEPENDENCIES vtk)   
  ENDIF()

  IF ( NOT ITK_DIR )
    # ITK_DIR is not supplied, so it is built inside Plus, therefore we need to specify dependencies to make sure it is built early enough
    SET(ITK_DEPENDENCIES)
    LIST(APPEND PlusLib_DEPENDENCIES itk)   
  ENDIF()

  INCLUDE(External_PlusBuild.cmake)
  
  IF( PlusLib_DIR )
    SET(RobartsVTKLib_PROJECT_ARGS ${RobartsVTKLib_PROJECT_ARGS} -DPlusLib_DIR:PATH=${PlusLib_DIR})
    SET(RobartsVTKLib_DEPENDENCIES ${RobartsVTKLib_DEPENDENCIES} PlusLib)
  ELSE()
    SET(RobartsVTKLib_PROJECT_ARGS ${RobartsVTKLib_PROJECT_ARGS} -DPlusLib_DIR:PATH=${PLUS_PLUSBUILD_BIN_DIR}/PlusLib-bin/src)
    SET(RobartsVTKLib_DEPENDENCIES ${RobartsVTKLib_DEPENDENCIES} Plus)
  ENDIF()
ENDIF()

IF (RobartsVTK_USE_CUDA)
  FIND_PACKAGE(CUDA REQUIRED)

  SET(RobartsVTKLib_PROJECT_ARGS ${RobartsVTKLib_PROJECT_ARGS} -DCUDA_TOOLKIT_ROOT_DIR:PATH=${CUDA_TOOLKIT_ROOT_DIR})
ENDIF()

#-----------------------------------------------------------------------------
# RobartsVTKLib download
#-----------------------------------------------------------------------------
  SET (RobartsVTK_SRC_DIR "${CMAKE_BINARY_DIR}/RobartsVTK")
  SET (RobartsVTK_BIN_DIR "${CMAKE_BINARY_DIR}/RobartsVTK-bin" CACHE INTERNAL "Path to store RobartsVTK binaries")
  ExternalProject_Add( RobartsVTK
    PREFIX "${CMAKE_BINARY_DIR}/RobartsVTK-prefix"
    SOURCE_DIR "${RobartsVTK_SRC_DIR}"
    BINARY_DIR "${RobartsVTK_BIN_DIR}"
    #--Download step--------------
    GIT_REPOSITORY git://git.imaging.robarts.ca/RobartsVTK.git
    GIT_TAG vtk6Migration
    #--Configure step-------------
    CMAKE_ARGS 
    ${ep_common_args}
    ${RobartsVTKLib_PROJECT_ARGS}
    #--Build step-----------------
    #--Install step-----------------
    INSTALL_COMMAND "" # Do not install
    DEPENDS ${RobartsVTKLib_DEPENDENCIES}
    )