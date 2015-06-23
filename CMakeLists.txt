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

# Legacy kits (don't enable if working in VTK6+)

# vtkAnis
OPTION (RobartsVTK_USE_ANIS "Build the vtkAnis module" OFF)
IF (RobartsVTK_USE_ANIS)
  IF(${VTK_MAJOR_VERSION} GREATER 5)
    MESSAGE(FATAL_ERROR "vtkAnis can't be built using VTK 6 and higher. Please downgrade to VTK 5.10 or disable RobartsVTK_USE_ANIS.")
  ENDIF()
  ADD_SUBDIRECTORY(vtkAnis)
  SET(RobartsVTK_INCLUDE_DIRS ${RobartsVTK_INCLUDE_DIRS} ${RobartsVTK_SOURCE_DIR}/vtkAnis )
  SET(RobartsVTK_LIBRARIES ${RobartsVTK_LIBRARIES} vtkAnis)
ENDIF (RobartsVTK_USE_ANIS)

# vtkRobartsECG
OPTION (RobartsVTK_USE_ROBARTSECG "Build the vtkRobartsECG module" OFF)
IF (RobartsVTK_USE_ROBARTSECG)
  IF(${VTK_MAJOR_VERSION} GREATER 5)
    MESSAGE(FATAL_ERROR "vtkRobartsECG can't be built using VTK 6 and higher. Please downgrade to VTK 5.10 or disable RobartsVTK_USE_ROBARTSECG.")
  ENDIF()
  ADD_SUBDIRECTORY(vtkRobartsECG)
  SET(RobartsVTK_INCLUDE_DIRS ${RobartsVTK_INCLUDE_DIRS} ${RobartsVTK_SOURCE_DIR}/vtkRobartsECG )
  SET(RobartsVTK_LIBRARIES ${RobartsVTK_LIBRARIES} vtkRobartsECG)
ENDIF (RobartsVTK_USE_ROBARTSECG)

# vtkRobartsHaptics
OPTION (RobartsVTK_USE_ROBARTSHAPTICS "Build the vtkRobartsHaptics module" OFF)
IF (RobartsVTK_USE_ROBARTSHAPTICS)
  IF(${VTK_MAJOR_VERSION} GREATER 5)
    MESSAGE(FATAL_ERROR "vtkRobartsHaptics can't be built using VTK 6 and higher. Please downgrade to VTK 5.10 or disable RobartsVTK_USE_ROBARTSHAPTICS.")
  ENDIF()
  ADD_SUBDIRECTORY(vtkRobartsHaptics)
  SET(RobartsVTK_INCLUDE_DIRS ${RobartsVTK_INCLUDE_DIRS} ${RobartsVTK_SOURCE_DIR}/vtkRobartsHaptics )
  SET(RobartsVTK_LIBRARIES ${RobartsVTK_LIBRARIES} vtkRobartsHaptics)
ENDIF (RobartsVTK_USE_ROBARTSHAPTICS)

# vtkRobartsLegacy
OPTION (RobartsVTK_USE_ROBARTSLEGACY "Build the vtkRobartsLegacy module" OFF)
IF (RobartsVTK_USE_ROBARTSLEGACY)
  IF(${VTK_MAJOR_VERSION} GREATER 5)
    MESSAGE(FATAL_ERROR "vtkRobartsLegacy can't be built using VTK 6 and higher. Please downgrade VTK or disable RobartsVTK_USE_ROBARTSLEGACY.")
  ENDIF()
  ADD_SUBDIRECTORY(vtkRobartsLegacy)
  SET(RobartsVTK_INCLUDE_DIRS ${RobartsVTK_INCLUDE_DIRS} ${RobartsVTK_SOURCE_DIR}/vtkRobartsLegacy )
  SET(RobartsVTK_LIBRARIES ${RobartsVTK_LIBRARIES} vtkRobartsLegacy)
ENDIF (RobartsVTK_USE_ROBARTSLEGACY)

# vtkRobartsUltrasound - contains new freehand ultrasound code
OPTION (RobartsVTK_USE_ROBARTSULTRASOUND "Build the vtkRobartsUltrasound module" OFF)
IF (RobartsVTK_USE_ROBARTSULTRASOUND)
  IF(${VTK_MAJOR_VERSION} GREATER 5)
    MESSAGE(FATAL_ERROR "vtkRobartsUltrasound can't be built using VTK 6 and higher. Please downgrade to VTK 5.10 or disable RobartsVTK_USE_ROBARTSULTRASOUND.")
  ENDIF()
  ADD_SUBDIRECTORY (vtkRobartsUltrasound)
  SET(RobartsVTK_INCLUDE_DIRS ${RobartsVTK_INCLUDE_DIRS} ${RobartsVTK_SOURCE_DIR}/vtkRobartsUltrasound)
  SET(RobartsVTK_LIBRARIES ${RobartsVTK_LIBRARIES} vtkRobartsUltrasound)
ENDIF (RobartsVTK_USE_ROBARTSULTRASOUND)

# vtkRobartsVideo
OPTION (RobartsVTK_USE_ROBARTSVIDEO "Build the vtkRobartsVideo module" OFF)
IF (RobartsVTK_USE_ROBARTSVIDEO)
  IF(${VTK_MAJOR_VERSION} GREATER 5)
    MESSAGE(FATAL_ERROR "vtkRobartsVideo can't be built using VTK 6 and higher. Please use PlusLib instead.")
  ENDIF()
  SUBDIRS(vtkRobartsVideo)
  SET(RobartsVTK_INCLUDE_DIRS ${RobartsVTK_INCLUDE_DIRS} ${RobartsVTK_SOURCE_DIR}/vtkRobartsVideo )
  SET(RobartsVTK_LIBRARIES ${RobartsVTK_LIBRARIES} vtkRobartsVideo)
ENDIF (RobartsVTK_USE_ROBARTSVIDEO)

# End of legacy kits

# vtkMwierz
OPTION (RobartsVTK_USE_MWIERZ "Build the vtkMwierz module" OFF)
IF (RobartsVTK_USE_MWIERZ)
  ADD_SUBDIRECTORY(vtkMwierz)
  SET(RobartsVTK_INCLUDE_DIRS ${RobartsVTK_INCLUDE_DIRS} ${RobartsVTK_SOURCE_DIR}/vtkMwierz )
  SET(RobartsVTK_LIBRARIES ${RobartsVTK_LIBRARIES} vtkMwierz)
ENDIF (RobartsVTK_USE_MWIERZ)

# vtkRobartsUtilities - contains random utilities like a Read-Write lock and Image Pipe over TCI/IP
OPTION (RobartsVTK_USE_ROBARTSUTILITIES "Build the vtkRobartsUltrasound module" OFF)
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

# ---------------------------------------------------------------
# PlusLib (replacing RobartsVideo and AIGS)

OPTION (RobartsVTK_USE_PLUSLIB "Download and build the PLUS library" OFF)
IF (RobartsVTK_USE_PLUSLIB)
  # Make sure we have all the components needed (temporary)

  IF (NOT QT_FOUND)
    MESSAGE( FATAL_ERROR "PlusLib requires Qt. One of these components is missing. Please verify configuration")
  ENDIF (NOT QT_FOUND)

  SET(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/Plus ${CMAKE_MODULE_PATH})
  # -----
  # GIT
  # Need Git for the automatic update of the repository
  # -----
  OPTION(PLUS_USE_GIT_PROTOCOL "If behind a firewall turn this off to use http instead." ON)
  SET(GIT_PROTOCOL "git")
  IF(NOT PLUS_USE_GIT_PROTOCOL)
    SET(GIT_PROTOCOL "http")
  ENDIF()

  MARK_AS_ADVANCED(CLEAR GIT_EXECUTABLE)
  FIND_FILE(GIT_EXECUTABLE git${CMAKE_EXECUTABLE_SUFFIX}
    PATHS
      "c:/Program Files/Git/bin/"
      "c:/Program Files (x86)/Git/bin/"
    )
  FIND_PACKAGE(Git)
  IF(NOT GIT_FOUND)
    MESSAGE(FATAL_ERROR "error: Install Git and try to re-configure")
  ENDIF()

  # -----
  # SVN - Let's check if a valid version of SVN is available
  # -----
  # Need subversion for the automatic update of the repository
  MARK_AS_ADVANCED(CLEAR Subversion_SVN_EXECUTABLE)
  FIND_FILE(Subversion_SVN_EXECUTABLE svn${CMAKE_EXECUTABLE_SUFFIX}
    PATHS
      "c:/Program Files/TortoiseSVN/bin/"
      "c:/Program Files/SlikSvn/bin/"
    )
  FIND_PACKAGE(Subversion REQUIRED)

  # -----
  # ASSEMBLA - Set username and password for assembla
  # Anonymous user name for Plus assembla server: https://subversion.assembla.com/svn/plus/
  # -----
  SET(PLUSBUILD_ASSEMBLA_USERNAME "perklab_anonymous" )
  SET(PLUSBUILD_ASSEMBLA_PASSWORD "anonymous" )

  SET(PLUS_SVN_REVISION 0 CACHE STRING "Set Plus desired SVN revision number (0 means latest)")

  # -----
  # Options to control build process
  # -----

  # Determine the operating system to set default values accordingly
  SET (ENABLED_BY_DEFAULT_ON_WINDOWS_ONLY OFF)
  SET (ENABLED_BY_DEFAULT_ON_WINDOWS32_ONLY OFF)
  if (CMAKE_HOST_WIN32)
    SET (ENABLED_BY_DEFAULT_ON_WINDOWS_ONLY ON)
    if (NOT CMAKE_CL_64)
      SET (ENABLED_BY_DEFAULT_ON_WINDOWS32_ONLY ON)
    endif ()
  endif (CMAKE_HOST_WIN32)

  OPTION(PLUS_USE_OpenIGTLink "Use OpenIGTLink" ON)
  OPTION(PLUS_BUILD_SHARED_LIBS "Build shared libraries for PLUS." ON)

  # Documentation
  OPTION(PLUS_DOCUMENTATION "Build Plus documentation (Doxygen)." OFF)
  IF(PLUS_DOCUMENTATION)
    # Try to detect GraphViz path (CMake's Doxygen package finder only tries some obsolete paths on Windows)
    # Parentheses is not permitted due to CMP0053
    SET(PROGRAMFILESX86 "ProgramFiles(x86)")
    find_program(DOXYGEN_DOT_EXECUTABLE
      NAMES dot
      PATHS
        "$ENV{ProgramFiles}/Graphviz2.38/bin"
        "$ENV{${PROGRAMFILESX86}}/Graphviz2.38/bin"
        "$ENV{ProgramFiles}/Graphviz2.34/bin"
        "$ENV{${PROGRAMFILESX86}}/Graphviz2.34/bin"
      DOC "Graphviz Dot tool for using Doxygen"
      NO_SYSTEM_ENVIRONMENT_PATH
    )
    find_package(Doxygen REQUIRED)

    IF(NOT DOXYGEN_FOUND)
      message( FATAL_ERROR "Documentation: Doxygen not found. Either specify location of doxygen or disable PLUS_DOCUMENTATION." )
    ENDIF(NOT DOXYGEN_FOUND)
    IF(NOT DOXYGEN_DOT_FOUND)
      message( FATAL_ERROR "Documentation: Graphviz dot tool not found (http://www.graphviz.org/Download.php, required by Doxygen for diagram generation). Either specify location of dot or disable PLUSBUILD_DOCUMENTATION." )
    ENDIF(NOT DOXYGEN_DOT_FOUND)
    OPTION(PLUS_DOCUMENTATION_SEARCH_SERVER_INDEXED "Search index for documentation is generated by th web server. Provides full-text search but only works on web servers." OFF)
  ENDIF(PLUS_DOCUMENTATION)

  # Imaging hardware

  # determine current OS
  if("$ENV{PROCESSOR_ARCHITEW6432}" STREQUAL "")
    if("$ENV{PROCESSOR_ARCHITECTURE}" STREQUAL "x86")
      SET( TEMP_OS_ARCH "x86" )
    else()
      SET( TEMP_OS_ARCH "x64" )
    endif()
  else()
    SET( TEMP_OS_ARCH "x64" )
  endif()

  OPTION(PLUS_USE_ULTRASONIX_VIDEO "Provide support for the Ultrasonix ultrasound systems" OFF)
  SET (PLUS_ULTRASONIX_SDK_MAJOR_VERSION 5 CACHE STRING "Set Ultrasonix SDK major version (version: [major].[minor].[patch])")
  SET (PLUS_ULTRASONIX_SDK_MINOR_VERSION 7 CACHE STRING "Set Ultrasonix SDK minor version (version: [major].[minor].[patch])")
  SET (PLUS_ULTRASONIX_SDK_PATCH_VERSION 4 CACHE STRING "Set Ultrasonix SDK patch version (version: [major].[minor].[patch])")
  OPTION(PLUS_TEST_ULTRASONIX "Enable testing of acquisition from Ultrasonix ultrasound systems. Enable this only if an Ultrasonix device accessible from this computer. " OFF)
  IF (PLUS_TEST_ULTRASONIX)
    SET (PLUS_TEST_ULTRASONIX_IP_ADDRESS "130.15.7.24" CACHE STRING "IP address of the Ultrasonix scanner that is used during testing")
  ENDIF(PLUS_TEST_ULTRASONIX)

  OPTION(PLUS_USE_BKPROFOCUS_VIDEO "Provide support for BK ProFocus ultrasound systems through the OEM (TCP/IP) interface" OFF)
  IF (PLUS_USE_BKPROFOCUS_VIDEO)
    OPTION(PLUS_USE_BKPROFOCUS_CAMERALINK "Enable acquisition from BK ProFocus ultrasound systems through CameraLink interface" OFF)
    OPTION(PLUS_TEST_BKPROFOCUS "Enable testing of acquisition from BK ProFocus ultrasound systems. Enable this only if a BK ProFocus device is connected to this computer. " OFF)
  ENDIF (PLUS_USE_BKPROFOCUS_VIDEO)
  IF ( (NOT ${CMAKE_GENERATOR} MATCHES "Win64") AND TEMP_OS_ARCH MATCHES "x64" AND PLUS_USE_BKPROFOCUS_CAMERALINK)
    # warning regarding cross compilation of bkprofocus
    MESSAGE( "BK ProFocus support on a 64-bit OS requires 64-bit Plus build. A 64-bit OS and a 32-bit Plus build configuration is detected. Compilation will be successful, but the resulting executables will fail to start." )
  ENDIF ()
  IF ( PLUS_USE_BKPROFOCUS_CAMERALINK AND (NOT PLUS_USE_BKPROFOCUS_VIDEO) )
    MESSAGE(FATAL_ERROR "error: PLUS_USE_BKPROFOCUS_VIDEO must be enabled if the PLUS_USE_BKPROFOCUS_CAMERALINK option is enabled" )
  ENDIF ()

  OPTION(PLUS_USE_ICCAPTURING_VIDEO "Provide support for the IC framegrabber device" OFF)
  OPTION(PLUS_USE_VFW_VIDEO "Provide support for the Video-for-Windows video digitizer (legacy, use Microsoft Media Foundation instead)" OFF)
  OPTION(PLUS_USE_MMF_VIDEO "Provide support for the Microsoft Media Foundation video digitizers (requires installation of Windows Platform SDK 7.1 or later)" OFF)
  IF(PLUS_USE_MMF_VIDEO)
    OPTION(PLUS_TEST_MMF_VIDEO "Enable testing of acquisition from MMF video device (webcam). Enable this only if an MMF device is connected to this computer." OFF)
  ENDIF(PLUS_USE_MMF_VIDEO)
  OPTION(PLUS_USE_EPIPHAN "Provide support for the Epiphan framegrabber device" OFF)
  OPTION(PLUS_USE_INTERSON_VIDEO "Provide support for the Interson USB ultrasound probes" OFF)
  OPTION(PLUS_USE_INTERSONSDKCXX_VIDEO "Provide support for the Interson SDK 1.X with C++ Wrapper USB ultrasound probes" OFF)
  OPTION(PLUS_USE_TELEMED_VIDEO "Provide support for the Telemed ultrasound probes" OFF)
  OPTION(PLUS_USE_THORLABS_VIDEO "Provide support for the ThorLabs Compact Spectrometers" OFF)

  # Tracking hardware

  OPTION(PLUS_USE_POLARIS "Provide support for the NDI POLARIS and AURORA" OFF)
  OPTION(PLUS_USE_CERTUS "Provide support for the NDI Certus" OFF)
  OPTION(PLUS_USE_MICRONTRACKER "Provide support for the Claron MicronTracker" OFF)
  OPTION(PLUS_USE_BRACHY_TRACKER "Provide support for the Brachy Steppers" ${ENABLED_BY_DEFAULT_ON_WINDOWS32_ONLY})
  OPTION(PLUS_USE_Ascension3DG "Provide support for the Ascension 3DG Tracker" ${ENABLED_BY_DEFAULT_ON_WINDOWS32_ONLY})
  OPTION(PLUS_USE_Ascension3DGm "Provide support for the Ascension MedSafe Tracker" OFF)
  IF (PLUS_USE_Ascension3DG AND PLUS_USE_Ascension3DGm)
    MESSAGE(FATAL_ERROR "PLUS_USE_Ascension3DGm and PLUS_USE_Ascension3DGm options cannot be enabled at the same time. See more details at https://www.assembla.com/spaces/plus/tickets/851")
  ENDIF()

  OPTION(PLUS_USE_PHIDGET_SPATIAL_TRACKER "Provide support for the Phidget Spatial accelerometer" OFF)
  OPTION(PLUS_USE_3dConnexion_TRACKER "Provide support for the 3dConnexion 3d mouse" OFF)
  OPTION(PLUS_USE_STEALTHLINK "Provide support for the Medtronick StealthLink Server" OFF)
  OPTION(PLUS_USE_IntuitiveDaVinci "Provide support for the da Vinci Surgical System" OFF)
  OPTION(PLUS_USE_PHILIPS_3D_ULTRASOUND "Provide support for the Philips ie33 3D ultrasound probe" OFF)
  IF(PLUS_USE_PHILIPS_3D_ULTRASOUND)
    OPTION(PLUS_TEST_PHILIPS_3D_ULTRASOUND "Enable testing of acquisition from Philips 3D ultrasound systems. Enable this only if a Philips device is accessible from this computer. " OFF)
  ENDIF(PLUS_USE_PHILIPS_3D_ULTRASOUND)
  IF (PLUS_TEST_PHILIPS_3D_ULTRASOUND)
    SET (PLUS_TEST_PHILIPS_3D_ULTRASOUND_IP_ADDRESS "129.100.44.8" CACHE STRING "IP address of the Philips scanner that is used during testing")
  ENDIF(PLUS_TEST_PHILIPS_3D_ULTRASOUND)

  # -----
  # Plus executable output path
  # -----
  SET (PLUS_EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}/bin") #CHECK THIS!!!!!

  # -----
  # Specify common external project properties
  # -----
  INCLUDE(${CMAKE_ROOT}/Modules/ExternalProject.cmake)
  INCLUDE(CTest)

  SET(ep_base "${CMAKE_BINARY_DIR}")

  SET(ep_common_args
    #-DCMAKE_INSTALL_PREFIX:PATH=${ep_install_dir}
    )
  IF(NOT ${CMAKE_GENERATOR} MATCHES "Visual Studio")
    SET(ep_common_args
      ${ep_common_args}
      -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
      )
  ENDIF()

  SET(ep_common_c_flags "${CMAKE_C_FLAGS_INIT} ${ADDITIONAL_C_FLAGS}")
  SET(ep_common_cxx_flags "${CMAKE_CXX_FLAGS_INIT} ${ADDITIONAL_CXX_FLAGS}")

  # Compute -G arg for configuring external projects with the same CMake generator:
  IF(CMAKE_EXTRA_GENERATOR)
    SET(gen "${CMAKE_EXTRA_GENERATOR} - ${CMAKE_GENERATOR}")
  ELSE()
    SET(gen "${CMAKE_GENERATOR}")
  ENDIF()

  # -----
  # Specify external projects
  # -----
  #UNSET(VTK_DIR CACHE)
  #UNSET(ITK_DIR CACHE)
  INCLUDE(Plus/External_VTK.cmake)
  INCLUDE(Plus/External_ITK.cmake)

  IF(PLUS_USE_OpenIGTLink)
    INCLUDE(Plus/External_OpenIGTLink.cmake)
  ENDIF(PLUS_USE_OpenIGTLink)

  IF ( PLUS_USE_BKPROFOCUS_VIDEO )
    INCLUDE(Plus/External_GrabbieLib.cmake)
  ENDIF()

  IF(PLUS_USE_CERTUS)
    FIND_PACKAGE (NDIOAPI)
    IF (NOT NDIOAPI_FOUND)
      MESSAGE( FATAL_ERROR "This project requires NDI Oapi for CERTUS tracking. One of the components is missing. Please verify configuration or turn off PLUS_USE_CERTUS.")
    ENDIF()
  ENDIF(PLUS_USE_CERTUS)

  IF (PLUS_USE_ULTRASONIX_VIDEO)
    FIND_PACKAGE (ULTRASONIX_SDK)
    IF (NOT ULTRASONIX_SDK_FOUND)
      MESSAGE( FATAL_ERROR "This project requires Ultrasonix SDK ${ULTRASONIX_SDK_VERSION} for Ultrasonix video. One of the components is missing. Please verify configuration or turn off PLUS_USE_ULTRASONIX_VIDEO.")
    ENDIF()
  ENDIF(PLUS_USE_ULTRASONIX_VIDEO)

  IF ( PLUS_USE_MICRONTRACKER )
    FIND_PACKAGE (MicronTracker)
    IF ( NOT MICRONTRACKER_FOUND)
      MESSAGE( FATAL_ERROR "This project requires Claron MicronTracker SDK for supporting the MicronTracker tracking device. One of the components is missing. Please verify configuration or turn off PLUS_USE_MICRONTRACKER.")
    ENDIF()
  ENDIF(PLUS_USE_MICRONTRACKER)

  IF ( PLUS_USE_ICCAPTURING_VIDEO )
   FIND_PACKAGE (ICCAPTURING)
    IF ( NOT ICCAPTURING_FOUND)
      MESSAGE( FATAL_ERROR "This project requires IC Capturing SDK for supporting the Imaging Source USB frame grabber. One of the components is missing. Please verify configuration or turn off PLUS_USE_ICCAPTURING_VIDEO.")
    ENDIF()
  ENDIF(PLUS_USE_ICCAPTURING_VIDEO)

  IF ( PLUS_USE_STEALTHLINK )
    FIND_PACKAGE (STEALTHLINK)
    IF ( NOT STEALTHLINK_FOUND)
      MESSAGE( FATAL_ERROR "This project requires Stealthlink2 SDK for supporting communication with Medtronic StealthStation. Please verify configuration or turn off PLUS_USE_STEALTHLINK.")
    ENDIF()
  ENDIF()

  IF ( PLUS_USE_INTERSON_VIDEO )
   FIND_PACKAGE (INTERSON)
    IF ( NOT INTERSON_FOUND)
      MESSAGE( FATAL_ERROR "This project requires Interson iSDK for supporting the Interson USB ultrasound probes. One of the components is missing. Please verify configuration or turn off PLUS_USE_INTERSON_VIDEO.")
    ENDIF()
  ENDIF()

  IF( PLUS_USE_IntuitiveDaVinci)
    FIND_PACKAGE (IntuitiveDaVinci)
    IF( NOT IntuitiveDaVinci_FOUND )
      MESSAGE( FATAL_ERROR "This project requires headers and library provided by Intuitive. One of the components is missing. Please verify configuration or turn off PLUS_USE_IntuitiveDaVinci.")
    ENDIF()
  ENDIF()

  IF ( PLUS_USE_TELEMED_VIDEO )
    FIND_PACKAGE (Telemed)
    IF ( NOT TELEMED_FOUND )
      MESSAGE( FATAL_ERROR "This project requires Telemed SDK for supporting the Telemed ultrasound probes. One of the components is missing. Please verify configuration or turn off PLUS_USE_TELEMED_VIDEO.")
    ENDIF()
  ENDIF()

  IF ( PLUS_USE_THORLABS_VIDEO )
    FIND_PACKAGE (ThorLabs)
    IF ( NOT THORLABS_FOUND )
      MESSAGE( FATAL_ERROR "This project requires ThorLabs CCS VISA  SDK for supporting the ThorLabs devices. One of the components is missing. Please verify configuration or turn off PLUS_USE_THORLABS_VIDEO.")
    ENDIF()
  ENDIF()

  IF (PLUS_USE_PHILIPS_3D_ULTRASOUND)
    IF(NOT CMAKE_HOST_WIN32)
    # Philips is windows only
    MESSAGE( FATAL_ERROR "Philips SDK is only available for Windows.")
    ENDIF(NOT CMAKE_HOST_WIN32)
    IF(PLUS_USE_ULTRASONIX_VIDEO)
    # Ultrasonix contains its own libmmd.dll which does not support all the functionality needed for the philips probe
    # They both cannot be enabled at the same time.
    MESSAGE( FATAL_ERROR "Plus cannot enable both Ultrasonix and Philips devices due to .dll conflicts in their respective SDK packages.")
    ELSE(PLUS_USE_ULTRASONIX_VIDEO)
      SET (PLUS_Philips_MAJOR_VERSION 1 CACHE STRING "Set Philips library major version (version: [major].[minor].[patch])")
      SET (PLUS_Philips_MINOR_VERSION 0 CACHE STRING "Set Philips library minor version (version: [major].[minor].[patch])")
      SET (PLUS_Philips_PATCH_VERSION 0 CACHE STRING "Set Philips library patch version (version: [major].[minor].[patch])")
      FIND_PACKAGE(Philips)
      IF( NOT PHILIPS_FOUND )
        MESSAGE( FATAL_ERROR "In order to use the Philips ie33 ultrasound system, the requisite DLLs must be made available. Please verify configuration or turn off PLUS_USE_PHILIPS_3D_ULTRASOUND.")
      ENDIF(NOT PHILIPS_FOUND)
    message( "To use the Philips devices you must:" )
    message( "1. Register '${CMAKE_CURRENT_BINARY_DIR}/bin/Debug|Release/Stream3d.dll'. If you are using a 32-bit OS, this can be done by running 'regsvr32 Stream3d.dll' in command line. If you are using a 64-bit OS, this can be done by the following procedure. Open the command line in administrator mode (Right click on cmd, choose 'Run as administrator'). Go to directory 'C:/Windows/SysWow64'. Run the regsvr32 command in SysWow64." )
    ENDIF(PLUS_USE_ULTRASONIX_VIDEO)
  ENDIF(PLUS_USE_PHILIPS_3D_ULTRASOUND)

  # -----
  # Specify target dependencies
  # -----
  SET(PlusLib_DEPENDENCIES)

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

  IF(PLUS_USE_OpenIGTLink AND NOT OpenIGTLink_DIR)
    # OpenIGTLink_DIR is not supplied, so it is built inside Plus, therefore we need to specify dependencies to make sure it is built early enough
    SET(OpenIGTLink_DEPENDENCIES)
    LIST(APPEND PlusLib_DEPENDENCIES OpenIGTLink)
  ENDIF()

  IF ( PLUS_USE_BKPROFOCUS_VIDEO )
    SET(GrabbieLib_DEPENDENCIES)
    LIST(APPEND PlusLib_DEPENDENCIES GrabbieLib)
  ENDIF()

  IF( PLUS_USE_INTERSONSDKCXX_VIDEO AND NOT IntersonSDKCxx_DIR )
    SET(IntersonSDKCxx_DEPENDENCIES)
    LIST(APPEND PlusLib_DEPENDENCIES IntersonSDKCxx)
    INCLUDE(Plus/External_IntersonSDKCxx.cmake)
  ENDIF()

  INCLUDE(Plus/External_PlusLib.cmake)

  SET(PLUSLIB_DIR ${CMAKE_BINARY_DIR}/PlusLib-bin CACHE PATH "The directory containing PlusLib binaries" FORCE)
  SET(RobartsVTK_INCLUDE_DIRS ${RobartsVTK_INCLUDE_DIRS} ${PLUSLIB_INCLUDE_DIRS} )
  SET(RobartsVTK_LIBRARIES ${RobartsVTK_LIBRARIES} ${PLUSLIB_LIBRARIES})
ENDIF (RobartsVTK_USE_PLUSLIB)


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
  IF(RobartsVTK_USE_ROBARTSVIDEO AND RobartsVTK_USE_DIRECTSHOW)
    ADD_SUBDIRECTORY(DirectShowExamples)
  ENDIF(RobartsVTK_USE_ROBARTSVIDEO AND RobartsVTK_USE_DIRECTSHOW)
ENDIF(RobartsVTK_BUILD_EXAMPLES)

#-----------------------------------------------------------------------------
# The commands in this directory are intended to be executed as
# the end of the whole configuration process, as a "last step".
# This directory is typically the last ADD_SUBDIRECTORY in the main CMakeLists.txt.
ADD_SUBDIRECTORY(CMake/LastConfigureStep)
