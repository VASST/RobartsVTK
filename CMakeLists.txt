PROJECT(RobartsVTK)

#-----------------------------------------------------------------------------
# CMake version requirements
#-----------------------------------------------------------------------------
CMAKE_MINIMUM_REQUIRED(VERSION 3.2)
cmake_policy(SET CMP0003 NEW)

SET(RobartsVTK_MAJOR_VERSION 2)
SET(RobartsVTK_MINOR_VERSION 0)
SET(RobartsVTK_PATCH_VERSION 0)
SET(RobartsVTK_VERSION ${RobartsVTK_MAJOR_VERSION}.${RobartsVTK_MINOR_VERSION}.${RobartsVTK_PATCH_VERSION})

# Use solution folders.
IF(MSVC OR ${CMAKE_GENERATOR} MATCHES "Xcode")
  SET_PROPERTY(GLOBAL PROPERTY USE_FOLDERS ON)
  SET_PROPERTY(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "CMakeTargets")
ENDIF()

INCLUDE(${CMAKE_SOURCE_DIR}/CMake/RobartsCommon.cmake)
LIST(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/CMake)

#-----------------------------------------------------------------------------
# Options
#-----------------------------------------------------------------------------
OPTION(RobartsVTK_USE_QT "Build with QT enabled" OFF)
OPTION(RobartsVTK_USE_ITK "Use ITK in RobartsITK" OFF)
OPTION(RobartsVTK_USE_PLUS "Use the PLUSToolkit for data collection." OFF)
OPTION(RobartsVTK_USE_REGISTRATION "Build the vtkRobartsRegistration module" OFF)
OPTION(RobartsVTK_USE_COMMON "Build the vtkRobartsCommon module" ON)
OPTION(RobartsVTK_USE_CUDA "Build the CUDA modules" OFF)
IF(RobartsVTK_USE_CUDA)
  OPTION(RobartsVTK_USE_CUDA_VISUALIZATION "Build the CUDA visualization modules" ON)
ENDIF()
IF(RobartsVTK_USE_CUDA AND RobartsVTK_USE_COMMON)
  OPTION(RobartsVTK_USE_CUDA_ANALYTICS "Build the CUDA image analytics modules" ON)
ENDIF()
OPTION(RobartsVTK_USE_OpenCL "Build the OpenCL modules" OFF)
OPTION(RobartsVTK_USE_VISUALIZATION  "Build the visualization modules" ON)
OPTION(RobartsVTK_BUILD_APPS "Build RobartsVTK applications." OFF)
OPTION(BUILD_SHARED_LIBS "Build shared libraries." ON)
OPTION(BUILD_DOCUMENTATION "Build Documentation" OFF)

#-----------------------------------------------------------------------------
# Compiler options
#-----------------------------------------------------------------------------
IF(MSVC)
  OPTION(CMAKE_CXX_MP_FLAG "Build with /MP flag enabled" OFF)
ENDIF()

#-----------------------------------------------------------------------------
# Testing
#-----------------------------------------------------------------------------
OPTION(BUILD_TESTING "Build test programs" ON)
INCLUDE(CTest)

#-----------------------------------------------------------------------------
# RobartsVTK executable output path 
#-----------------------------------------------------------------------------
# Set up our directory structure for output libraries and binaries
IF(NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY)
  SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY "bin")
ENDIF()
IF(NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)
  SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY "lib")
ENDIF()
IF(NOT CMAKE_ARCHIVE_OUTPUT_DIRECTORY)
  SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "lib")
ENDIF()
if(NOT RobartsVTK_INCLUDE_INSTALL)
  set(RobartsVTK_INCLUDE_INSTALL "include")
endif()
MARK_AS_ADVANCED(CMAKE_RUNTIME_OUTPUT_DIRECTORY
  CMAKE_LIBRARY_OUTPUT_DIRECTORY
  CMAKE_ARCHIVE_OUTPUT_DIRECTORY
  RobartsVTK_INCLUDE_INSTALL
  )

#-----------------------------------------------------------------------------
# Add configure file to included directories
#-----------------------------------------------------------------------------
INCLUDE_DIRECTORIES(${CMAKE_BINARY_DIR})

#-----------------------------------------------------------------------------
# Find OpenMP
#-----------------------------------------------------------------------------
FIND_PACKAGE(OpenMP REQUIRED)

#-----------------------------------------------------------------------------
# Look for a VTK build directory
#-----------------------------------------------------------------------------
FIND_PACKAGE(VTK REQUIRED NO_MODULE)
IF( ${VTK_VERSION} VERSION_LESS "7" )
  MESSAGE( SEND_ERROR "VTK must be version 7 or higher.")
ENDIF()
MESSAGE(STATUS "Using VTK available at: ${VTK_DIR}")

#-----------------------------------------------------------------------------
# Confirm that VTK was built with Qt IF requested
#-----------------------------------------------------------------------------
OPTION(RobartsVTK_USE_QT "Build with QT enabled" OFF)
IF(RobartsVTK_USE_QT)
  IF(NOT TARGET vtkGUISupportQt)
    MESSAGE( SEND_ERROR "You have to build VTK with VTK_USE_QT flag ON IF you need to use RobartsVTK_USE_QT.")
  ENDIF()
ENDIF()

#-----------------------------------------------------------------------------
# Qt
#-----------------------------------------------------------------------------
IF(RobartsVTK_USE_QT)
  IF( (RobartsVTK_USE_VISUALIZATION OR RobartsVTK_USE_CUDA_ANALYTICS) AND RobartsVTK_BUILD_APPS)
    SET(RobartsVTK_VIS_QT_COMPONENTS Test Concurrent Xml)
  ENDIF()

  FIND_PACKAGE(Qt5 REQUIRED COMPONENTS Widgets ${RobartsVTK_VIS_QT_COMPONENTS})

  SET(QT_VERSION_MAJOR ${Qt5Widgets_VERSION_MAJOR})
  SET(CMAKE_AUTOMOC ON)
  SET(CMAKE_AUTOUIC ON)
  SET(CMAKE_AUTORCC ON)

  GET_FILENAME_COMPONENT(QT_CMAKE_DIR ${Qt5_DIR} DIRECTORY)
  GET_FILENAME_COMPONENT(QT_LIB_DIR ${QT_CMAKE_DIR} DIRECTORY)
  GET_FILENAME_COMPONENT(QT_ROOT_DIR ${QT_LIB_DIR} DIRECTORY)
  SET(QT_BINARY_DIR ${QT_ROOT_DIR}/bin)

  # --------------------------------------------------------------------------
  # Copy Qt binaries to CMAKE_RUNTIME_OUTPUT_DIRECTORY
  IF( ${CMAKE_GENERATOR} MATCHES "Visual Studio" )
    # Determine shared library extension without the dot (dll instead of .dll)
    STRING(SUBSTRING ${CMAKE_SHARED_LIBRARY_SUFFIX} 1 -1 CMAKE_SHARED_LIBRARY_SUFFIX_NO_SEPARATOR)

    # Get all Qt shared library names
    SET(RELEASE_REGEX_PATTERN .t5.*[^d][.]${CMAKE_SHARED_LIBRARY_SUFFIX_NO_SEPARATOR})
    SET(DEBUG_REGEX_PATTERN .t5.*d[.]${CMAKE_SHARED_LIBRARY_SUFFIX_NO_SEPARATOR})

    # Copy shared libraries to bin directory to allow running Plus applications in the build tree  
    FILE(COPY "${QT_BINARY_DIR}/"
      DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Release
      FILES_MATCHING REGEX ${RELEASE_REGEX_PATTERN}
      )
    FILE(COPY "${QT_BINARY_DIR}/"
      DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Debug
      FILES_MATCHING REGEX ${DEBUG_REGEX_PATTERN}
      )
  ENDIF()
ENDIF()

#-----------------------------------------------------------------------------
# Look for an ITK build directory (optional)
#-----------------------------------------------------------------------------
OPTION(RobartsVTK_USE_ITK "Use ITK in RobartsITK" OFF)
IF(RobartsVTK_USE_ITK)
  # ITK has been built already
  FIND_PACKAGE(ITK REQUIRED NO_MODULE)

  IF(ITK_FOUND)
    MESSAGE(STATUS "Using ITK available at: ${ITK_DIR}")

    SET (RobartsVTK_ITK_DIR "${ITK_DIR}" CACHE INTERNAL "Path to store itk binaries")
  ENDIF()
ENDIF()

#-----------------------------------------------------------------------------
# Choose static or shared libraries.
#-----------------------------------------------------------------------------
OPTION(BUILD_SHARED_LIBS "Build shared libraries." ${BUILD_SHARED_LIBS})
SET (RobartsVTK_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
IF(BUILD_SHARED_LIBS)
  SET (BUILD_SHARED_LIBS_CONFIG 1)
ELSE (BUILD_SHARED_LIBS)
  SET (BUILD_SHARED_LIBS_CONFIG 0)
ENDIF(BUILD_SHARED_LIBS)

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
IF(NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)
  SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin CACHE INTERNAL "Single output directory for building all libraries.")
ENDIF()
IF(NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY)
  SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin CACHE INTERNAL "Single output directory for building all executables.")
ENDIF()
IF(NOT CMAKE_ARCHIVE_OUTPUT_DIRECTORY)
  SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib CACHE INTERNAL "Single output directory for building all archives.")
ENDIF()

SET(RobartsVTK_Export_Template ${CMAKE_CURRENT_SOURCE_DIR}/RobartsVTKExport.h.in)

#-----------------------------------------------------------------------------
# PLUS configuration (optional)
#-----------------------------------------------------------------------------
IF(RobartsVTK_USE_PLUS)
    # PlusLib has been built already
  FIND_PACKAGE(PlusLib REQUIRED NO_MODULE)

  IF(PlusLib_FOUND)
    MESSAGE(STATUS "Using PlusLib available at: ${PlusLib_DIR}")
  ENDIF()
ENDIF()

#-----------------------------------------------------------------------------
# Start listing the RobartsVTK modules
#-----------------------------------------------------------------------------
SET(RobartsVTK_LIBRARIES)
SET(RobartsVTK_INCLUDE_DIRS ${CMAKE_CURRENT_BINARY_DIR} CACHE INTERNAL "")

#-----------------------------------------------------------------------------
# vtkRobartsCommon - contains common utilities
#-----------------------------------------------------------------------------
OPTION(RobartsVTK_USE_COMMON "Build the vtkRobartsCommon module" ON)
IF(RobartsVTK_USE_COMMON)
  ADD_SUBDIRECTORY (Common/vtkRobartsCommon)

  LIST(APPEND RobartsVTK_LIBRARIES vtkRobartsCommon)
ENDIF()

#-----------------------------------------------------------------------------
# vtkRobartsRegistration
#-----------------------------------------------------------------------------
OPTION(RobartsVTK_USE_REGISTRATION "Build the vtkRobartsRegistration module" OFF)
IF(RobartsVTK_USE_REGISTRATION)
  ADD_SUBDIRECTORY(ImageProcessing/vtkRobartsRegistration)

  LIST(APPEND RobartsVTK_LIBRARIES vtkRobartsRegistration)
ENDIF()

#-----------------------------------------------------------------------------
# CUDA Dependencies
#-----------------------------------------------------------------------------
OPTION(RobartsVTK_USE_CUDA "Build the CUDA modules" OFF)
IF(RobartsVTK_USE_CUDA)
  # Look for CUDA facilities
  FIND_PACKAGE(CUDA REQUIRED)

  # Allow the user to define the CUDA build architectures
  IF(${CUDA_VERSION} VERSION_LESS "7.0.0")
    SET(_generations "Fermi" "Kepler" "Maxwell")
  ELSE()
    SET(_generations "Fermi" "Kepler" "Maxwell" "Pascal")
  ENDIF()

  SET(CUDA_GENERATION "" CACHE STRING "Build CUDA device code only for specific GPU architecture. Leave empty to build for all architectures.")
  IF(CMAKE_VERSION VERSION_GREATER "2.8" )
    SET_PROPERTY( CACHE CUDA_GENERATION PROPERTY STRINGS "" ${_generations} )
  ENDIF()

  IF(CUDA_GENERATION)
    IF(NOT ";${_generations};" MATCHES ";${CUDA_GENERATION};")
      STRING(REPLACE ";" ", " _generations "${_generations}")
      message(FATAL_ERROR "ERROR: ${_generations} Generations are suppered.")
    ENDIF()
    UNSET(CUDA_ARCH_BIN CACHE)
    UNSET(CUDA_ARCH_PTX CACHE)
  ENDIF()

  SET(__cuda_arch_ptx "")
  IF(CUDA_GENERATION STREQUAL "Fermi")
    SET(__cuda_arch_bin "2.0")
  ELSEIF(CUDA_GENERATION STREQUAL "Kepler")
    SET(__cuda_arch_bin "3.0 3.5 3.7")
  ELSEIF(CUDA_GENERATION STREQUAL "Maxwell")
    SET(__cuda_arch_bin "5.0")
  ELSEIF(CUDA_GENERATION STREQUAL "Pascal")
    SET(__cuda_arch_bin "6.0")
  ENDIF()

  IF(NOT DEFINED __cuda_arch_bin)
    IF(ARM)
      SET(__cuda_arch_bin "3.2")
      SET(__cuda_arch_ptx "")
    ELSEIF(AARCH64)
      SET(__cuda_arch_bin "5.3")
      SET(__cuda_arch_ptx "")
    else()
      IF(${CUDA_VERSION} VERSION_LESS "8.0")
        SET(__cuda_arch_bin "2.0 3.0 3.5 3.7 5.0")
      else()
        SET(__cuda_arch_bin "2.0 3.0 3.5 3.7 5.0 6.0")
      ENDIF()
      SET(__cuda_arch_ptx "")
    ENDIF()
  ENDIF()

  SET(CUDA_ARCH_BIN ${__cuda_arch_bin} CACHE STRING "Specify 'real' GPU architectures to build binaries for, BIN(PTX) format is supported")
  SET(CUDA_ARCH_PTX ${__cuda_arch_ptx} CACHE STRING "Specify 'virtual' PTX architectures to build PTX intermediate code for")

  STRING(REGEX REPLACE "\\." "" ARCH_BIN_NO_POINTS "${CUDA_ARCH_BIN}")
  STRING(REGEX REPLACE "\\." "" ARCH_PTX_NO_POINTS "${CUDA_ARCH_PTX}")
  
  # Ckeck if user specified 1.0 compute capability: we don't support it
  STRING(REGEX MATCH "1.0" HAS_ARCH_10 "${CUDA_ARCH_BIN} ${CUDA_ARCH_PTX}")
  SET(CUDA_ARCH_BIN_OR_PTX_10 0)
  IF(NOT ${HAS_ARCH_10} STREQUAL "")
    SET(CUDA_ARCH_BIN_OR_PTX_10 1)
  ENDIF()

  # NVCC flags to be SET
  SET(NVCC_FLAGS_EXTRA "")

  # These vars will be passed into the templates
  SET(RobartsVTK_CUDA_ARCH_BIN "")
  SET(RobartsVTK_CUDA_ARCH_PTX "")
  SET(RobartsVTK_CUDA_ARCH_FEATURES "")

  # Tell NVCC to add binaries for the specified GPUs
  STRING(REGEX MATCHALL "[0-9()]+" ARCH_LIST "${ARCH_BIN_NO_POINTS}")
  FOREACH(ARCH IN LISTS ARCH_LIST)
    IF(ARCH MATCHES "([0-9]+)\\(([0-9]+)\\)")
      # User explicitly specIFied PTX for the concrete BIN
      SET(NVCC_FLAGS_EXTRA ${NVCC_FLAGS_EXTRA} -gencode arch=compute_${CMAKE_MATCH_2},code=sm_${CMAKE_MATCH_1})
      SET(RobartsVTK_CUDA_ARCH_BIN "${RobartsVTK_CUDA_ARCH_BIN} ${CMAKE_MATCH_1}")
      SET(RobartsVTK_CUDA_ARCH_FEATURES "${RobartsVTK_CUDA_ARCH_FEATURES} ${CMAKE_MATCH_2}")
    else()
      # User didn't explicitly specIFy PTX for the concrete BIN, we assume PTX=BIN
      SET(NVCC_FLAGS_EXTRA ${NVCC_FLAGS_EXTRA} -gencode arch=compute_${ARCH},code=sm_${ARCH})
      SET(RobartsVTK_CUDA_ARCH_BIN "${RobartsVTK_CUDA_ARCH_BIN} ${ARCH}")
      SET(RobartsVTK_CUDA_ARCH_FEATURES "${RobartsVTK_CUDA_ARCH_FEATURES} ${ARCH}")
    ENDIF()
  ENDFOREACH()
  SET(NVCC_FLAGS_EXTRA ${NVCC_FLAGS_EXTRA} -D_FORCE_INLINES)

  # Tell NVCC to add PTX intermediate code for the specIFied architectures
  STRING(REGEX MATCHALL "[0-9]+" ARCH_LIST "${ARCH_PTX_NO_POINTS}")
  FOREACH(ARCH IN LISTS ARCH_LIST)
    SET(NVCC_FLAGS_EXTRA ${NVCC_FLAGS_EXTRA} -gencode arch=compute_${ARCH},code=compute_${ARCH})
    SET(RobartsVTK_CUDA_ARCH_PTX "${RobartsVTK_CUDA_ARCH_PTX} ${ARCH}")
    SET(RobartsVTK_CUDA_ARCH_FEATURES "${RobartsVTK_CUDA_ARCH_FEATURES} ${ARCH}")
  ENDFOREACH()

  # These vars will be processed in other scripts
  SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ${NVCC_FLAGS_EXTRA})
  SET(RobartsVTK_CUDA_CC "${NVCC_FLAGS_EXTRA}")

  MESSAGE(STATUS "CUDA NVCC target flags: ${CUDA_NVCC_FLAGS}")

  # Look for OpenGL
  FIND_PACKAGE(OpenGL REQUIRED)

  # Add Robart's CUDA libraries
  ADD_SUBDIRECTORY(Common/vtkCudaCommon)

  LIST(APPEND RobartsVTK_LIBRARIES vtkCudaCommon)

  ADD_SUBDIRECTORY(Common/vtkCudaCommonStatic)

  LIST(APPEND RobartsVTK_LIBRARIES vtkCudaCommonStatic)
ENDIF()

#-----------------------------------------------------------------------------
# Add the CUDA Visualization Libraries
#-----------------------------------------------------------------------------
IF(RobartsVTK_USE_CUDA)
  OPTION(RobartsVTK_USE_CUDA_VISUALIZATION "Build the CUDA visualization modules" ON)
  IF(RobartsVTK_USE_CUDA_VISUALIZATION)
    FIND_PACKAGE(OpenGL REQUIRED)

    ADD_SUBDIRECTORY(Visualization/vtkCudaVisualization)

    LIST(APPEND RobartsVTK_LIBRARIES vtkCudaVisualization)
  ENDIF()
ENDIF()

#-----------------------------------------------------------------------------
# Add the Robarts Visualization Libraries
#-----------------------------------------------------------------------------
OPTION(RobartsVTK_USE_VISUALIZATION "Build the Robarts Visualization modules" OFF)
IF(RobartsVTK_USE_VISUALIZATION)
  FIND_PACKAGE(OpenGL REQUIRED)

  ADD_SUBDIRECTORY(Visualization/vtkRobartsVisualization)

  LIST(APPEND RobartsVTK_LIBRARIES vtkRobartsVisualization)
ENDIF()

#-----------------------------------------------------------------------------
# Add the CUDA Analytics Libraries
#-----------------------------------------------------------------------------
IF(RobartsVTK_USE_CUDA AND RobartsVTK_USE_COMMON)
  OPTION(RobartsVTK_USE_CUDA_ANALYTICS "Build the CUDA image analytics modules" ON)
  IF(RobartsVTK_USE_CUDA_ANALYTICS)
    ADD_SUBDIRECTORY(ImageProcessing/vtkCudaImageAnalytics)

    LIST(APPEND RobartsVTK_LIBRARIES vtkCudaImageAnalytics)
  ENDIF()
ENDIF()

#-----------------------------------------------------------------------------
# OpenCL Dependencies
#-----------------------------------------------------------------------------
IF(RobartsVTK_USE_OpenCL)
  FIND_PACKAGE(OpenCL REQUIRED)

  ADD_LIBRARY(OpenCL IMPORTED SHARED)

  IF(${OpenCL_LIBRARY} MATCHES "${CMAKE_SHARED_LIBRARY_SUFFIX}")
    SET_TARGET_PROPERTIES(OpenCL PROPERTIES IMPORTED_LOCATION ${OpenCL_LIBRARY})
  ELSEIF(${OpenCL_LIBRARY} MATCHES "${CMAKE_STATIC_LIBRARY_SUFFIX}")
    SET_TARGET_PROPERTIES(OpenCL PROPERTIES IMPORTED_IMPLIB ${OpenCL_LIBRARY})
  ENDIF()

  OPTION(RobartsVTK_USE_CL_VOLUMERECONSTRUCTION "Build the OpenCL Volume Reconstruction modules" ON)
  IF(RobartsVTK_USE_CL_VOLUMERECONSTRUCTION)
    ADD_SUBDIRECTORY(ImageProcessing/vtkCLVolumeReconstruction)

    LIST(APPEND RobartsVTK_LIBRARIES vtkCLVolumeReconstruction)
  ENDIF()
ENDIF()

#-----------------------------------------------------------------------------
# Applications
#-----------------------------------------------------------------------------
OPTION(RobartsVTK_BUILD_APPS "Build RobartsVTK applications." OFF)
IF(RobartsVTK_BUILD_APPS)
  IF(RobartsVTK_USE_CUDA)
    ADD_SUBDIRECTORY(Applications/Cuda2DTFVolumeRendering)
    SET_TARGET_PROPERTIES(Cuda2DTFVolumeRendering PROPERTIES FOLDER Applications)
    IF(RobartsVTK_USE_QT AND RobartsVTK_USE_CUDA_VISUALIZATION)
      ADD_SUBDIRECTORY(Applications/FluoroPredViz)
      SET_TARGET_PROPERTIES(FluoroPredViz PROPERTIES FOLDER Applications)
      ADD_SUBDIRECTORY(Applications/DRRExample)
      SET_TARGET_PROPERTIES(DRRExample PROPERTIES FOLDER Applications)
      ADD_SUBDIRECTORY(Applications/TransferFunctionGUI)
      SET_TARGET_PROPERTIES(TFUIDual PROPERTIES FOLDER Applications)
      SET_TARGET_PROPERTIES(TFUIGradient PROPERTIES FOLDER Applications)
      SET_TARGET_PROPERTIES(TFUIKohonen PROPERTIES FOLDER Applications)
    ENDIF()

    IF(RobartsVTK_USE_QT AND RobartsVTK_USE_CUDA_ANALYTICS)
      ADD_SUBDIRECTORY(Applications/SEGUE)
      SET_TARGET_PROPERTIES(SEGUE PROPERTIES FOLDER Applications)
    ENDIF()

    IF(RobartsVTK_USE_OPENCV)
      IF(RobartsVTK_USE_VISUALIZATION)
        IF(RobartsVTK_USE_CUDA_VISUALIZATION)
          ADD_SUBDIRECTORY(Applications/StreamingUSVis)
          SET_TARGET_PROPERTIES(StreamingUSVis PROPERTIES FOLDER Applications)
          ADD_SUBDIRECTORY(Applications/Cuda1DTFVolumeRendering)
          SET_TARGET_PROPERTIES(Cuda1DTFVolumeRendering PROPERTIES FOLDER Applications)
          ADD_SUBDIRECTORY(Applications/CLReconstruction)
          SET_TARGET_PROPERTIES(CLReconstruction PROPERTIES FOLDER Applications)
          ADD_SUBDIRECTORY(Applications/CLReconstructionWithVisualization)
          SET_TARGET_PROPERTIES(CLReconstructionWithVisualization PROPERTIES FOLDER Applications)
        ENDIF()

        ADD_SUBDIRECTORY(Applications/KeyholeRendering)
        ADD_SUBDIRECTORY(Applications/KeyholeRenderingMultipleViewPorts2)
        ADD_SUBDIRECTORY(Applications/KeyholeRenderingWithVideo)
        ADD_SUBDIRECTORY(Applications/KeyholeRenderingWithStereoVideo)
        ADD_SUBDIRECTORY(Applications/KeyholeRenderingMultipleViewPorts)
        SET_TARGET_PROPERTIES(KeyholeRenderingWithStereoVideo
                              KeyholeRenderingWithVideo
                              KeyholeRenderingMultipleViewPorts2
                              KeyholeRendering
                              KeyholeRenderingMultipleViewPorts
                              PROPERTIES FOLDER Applications)
        
      ENDIF()
    ENDIF()
  ENDIF()

  IF(RobartsVTK_USE_COMMON)
    ADD_SUBDIRECTORY(Applications/MaxFlow)
    SET_TARGET_PROPERTIES(MaxFlow GHMFSegment PROPERTIES FOLDER Applications)
    IF(RobartsVTK_USE_CUDA AND RobartsVTK_USE_CUDA_ANALYTICS)
      ADD_SUBDIRECTORY(Applications/CudaMaxFlow)
      SET_TARGET_PROPERTIES(CudaMaxFlow CUDAGHMFSegment KSOMTrain KSOMApply PROPERTIES FOLDER Applications)
    ENDIF()
  ENDIF()

  IF(RobartsVTK_USE_PLUS AND RobartsVTK_USE_QT AND RobartsVTK_USE_OPENCV)
    MESSAGE(STATUS "Using PlusApp available at: ${PlusApp_DIR}")

    ADD_SUBDIRECTORY(Applications/CameraCalibration)
    ADD_SUBDIRECTORY(Applications/PLUSOpenCVExample)
    IF(TARGET CameraCalibration)
      SET_TARGET_PROPERTIES(CameraCalibration PROPERTIES FOLDER Applications)
    ENDIF()
    IF(TARGET PLUSOpenCVExample)
      SET_TARGET_PROPERTIES(PLUSOpenCVExample PROPERTIES FOLDER Applications)
    ENDIF()
  ENDIF()

  IF(OpenCV_FOUND)
    MESSAGE(STATUS "Using OpenCV available at: ${OpenCV_DIR}")
  ENDIF()
ENDIF()

SET(RobartsVTK_INCLUDE_DIRS ${CMAKE_BINARY_DIR})
LIST(REMOVE_DUPLICATES RobartsVTK_LIBRARIES)

#-----------------------------------------------------------------------------
# Populate export variables
#-----------------------------------------------------------------------------
FOREACH(lib ${RobartsVTK_LIBRARIES})
  GET_TARGET_PROPERTY(lib_DIRS ${lib} INTERFACE_INCLUDE_DIRECTORIES)
  LIST(APPEND RobartsVTK_INCLUDE_DIRS ${lib_DIRS})
ENDFOREACH()
LIST(REMOVE_DUPLICATES RobartsVTK_INCLUDE_DIRS)

#-----------------------------------------------------------------------------
# Generate build configurations
#-----------------------------------------------------------------------------
CONFIGURE_FILE(
  ${CMAKE_CURRENT_SOURCE_DIR}/RobartsVTKConfigure.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/RobartsVTKConfigure.h
  )
IF(WIN32)
  CONFIGURE_FILE(
    ${CMAKE_CURRENT_SOURCE_DIR}/BuildAndTest.bat.in
    ${CMAKE_CURRENT_BINARY_DIR}/BuildAndTest.bat
    )
ELSE()
  CONFIGURE_FILE(
    ${CMAKE_CURRENT_SOURCE_DIR}/BuildAndTest.sh.in
    ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/BuildAndTest.sh
    )
ENDIF()

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
  ${CMAKE_CURRENT_BINARY_DIR}/RobartsVTKConfig.cmake
  )
CONFIGURE_FILE(
  ${CMAKE_CURRENT_SOURCE_DIR}/RobartsVTKConfigVersion.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/RobartsVTKConfigVersion.cmake 
  @ONLY)