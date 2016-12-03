PROJECT(RobartsVTK)

# CMake version requirements
CMAKE_MINIMUM_REQUIRED(VERSION 3.2)
cmake_policy(SET CMP0003 NEW)

SET(RobartsVTK_MAJOR_VERSION 2)
SET(RobartsVTK_MINOR_VERSION 0)
SET(RobartsVTK_PATCH_VERSION 0)
SET(RobartsVTK_VERSION ${RobartsVTK_MAJOR_VERSION}.${RobartsVTK_MINOR_VERSION}.${RobartsVTK_PATCH_VERSION})

# Use solution folders.
IF(MSVC OR ${CMAKE_GENERATOR} MATCHES "Xcode")
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)
  set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "CMakeTargets")
ENDIF()

INCLUDE(${CMAKE_SOURCE_DIR}/CMake/RobartsCommon.cmake)
LIST(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/CMake)

# Options
OPTION(RobartsVTK_USE_QT "Build with QT enabled" OFF)
OPTION(RobartsVTK_USE_ITK "Use ITK in RobartsITK" OFF)
OPTION(RobartsVTK_USE_PLUS "Use the PLUSToolkit for data collection." OFF)
OPTION(RobartsVTK_USE_REGISTRATION "Build the vtkRobartsRegistration module" OFF)
OPTION(RobartsVTK_USE_COMMON "Build the vtkRobartsCommon module" ON)
OPTION(RobartsVTK_USE_CUDA "Build the CUDA modules" OFF)
IF (RobartsVTK_USE_CUDA)
  OPTION(RobartsVTK_USE_CUDA_VISUALIZATION "Build the CUDA visualization modules" ON)
ENDIF()
IF (RobartsVTK_USE_CUDA AND RobartsVTK_USE_COMMON)
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
  add_compile_options($<$<CXX_COMPILER_ID:MSVC>:/MP>)
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

SET(RobartsVTK_DATA_DIR ${CMAKE_SOURCE_DIR}/Data CACHE PATH "Location of the data directory.")

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
FIND_PACKAGE(VTK 7 REQUIRED NO_MODULE)

MESSAGE(STATUS "Using VTK available at: ${VTK_DIR}")

#-----------------------------------------------------------------------------
# Confirm that VTK was built with Qt if requested
#-----------------------------------------------------------------------------
IF(RobartsVTK_USE_QT)
  IF (NOT TARGET vtkGUISupportQt)
    MESSAGE( SEND_ERROR "You have to build VTK with VTK_USE_QT flag ON if you need to use RobartsVTK_USE_QT.")
  ENDIF()
ENDIF()

#-----------------------------------------------------------------------------
# Qt
#-----------------------------------------------------------------------------
OPTION (RobartsVTK_USE_QT "Build with QT enabled" OFF)
IF (RobartsVTK_USE_QT)
  IF(RobartsVTK_USE_VISUALIZATION AND RobartsVTK_BUILD_APPS)
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
  IF ( ${CMAKE_GENERATOR} MATCHES "Visual Studio" )
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
OPTION(BUILD_SHARED_LIBS "Build shared libraries." ${BUILD_SHARED_LIBS} )
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
IF(NOT LIBRARY_OUTPUT_PATH)
  SET(LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR}/lib CACHE INTERNAL "Single output directory for building all libraries.")
ENDIF()
IF(NOT EXECUTABLE_OUTPUT_PATH)
  SET(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR}/bin CACHE INTERNAL "Single output directory for building all executables.")
ENDIF()

SET(RobartsVTK_Export_Template ${CMAKE_CURRENT_SOURCE_DIR}/RobartsVTKExport.h.in)

#-----------------------------------------------------------------------------
# PLUS configuration (optional)
#-----------------------------------------------------------------------------
OPTION(RobartsVTK_USE_PLUS "Use the PLUSToolkit for data collection." OFF)
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
SET(RobartsVTK_LIBRARIES CACHE INTERNAL "")
SET(RobartsVTK_INCLUDE_DIRS ${CMAKE_CURRENT_BINARY_DIR} CACHE INTERNAL "")

#-----------------------------------------------------------------------------
# vtkRobartsCommon - contains common utilities
#-----------------------------------------------------------------------------
OPTION (RobartsVTK_USE_COMMON "Build the vtkRobartsCommon module" ON)
IF (RobartsVTK_USE_COMMON)
  ADD_SUBDIRECTORY (Common/vtkRobartsCommon)

  SET(RobartsVTK_LIBRARIES ${RobartsVTK_LIBRARIES} vtkRobartsCommon CACHE INTERNAL "")
ENDIF()

#-----------------------------------------------------------------------------
# vtkRobartsRegistration
#-----------------------------------------------------------------------------
OPTION (RobartsVTK_USE_REGISTRATION "Build the vtkRobartsRegistration module" OFF)
IF (RobartsVTK_USE_REGISTRATION)
  ADD_SUBDIRECTORY(ImageProcessing/vtkRobartsRegistration)

  SET(RobartsVTK_LIBRARIES ${RobartsVTK_LIBRARIES} vtkRobartsRegistration CACHE INTERNAL "")
ENDIF ()

#-----------------------------------------------------------------------------
# CUDA Dependencies
#-----------------------------------------------------------------------------
OPTION (RobartsVTK_USE_CUDA "Build the CUDA modules" OFF)
IF (RobartsVTK_USE_CUDA)
  # Look for CUDA facilities
  FIND_PACKAGE(CUDA REQUIRED)

  # Allow the user to define the CUDA build architectures
  IF(${CUDA_VERSION} VERSION_LESS "7.0.0")
    set(_generations "Fermi" "Kepler" "Maxwell")
  ELSE()
    set(_generations "Fermi" "Kepler" "Maxwell" "Pascal")
  ENDIF()

  set(CUDA_GENERATION "" CACHE STRING "Build CUDA device code only for specific GPU architecture. Leave empty to build for all architectures.")
  if( CMAKE_VERSION VERSION_GREATER "2.8" )
    set_property( CACHE CUDA_GENERATION PROPERTY STRINGS "" ${_generations} )
  endif()

  if(CUDA_GENERATION)
    if(NOT ";${_generations};" MATCHES ";${CUDA_GENERATION};")
      string(REPLACE ";" ", " _generations "${_generations}")
      message(FATAL_ERROR "ERROR: ${_generations} Generations are suppered.")
    endif()
    unset(CUDA_ARCH_BIN CACHE)
    unset(CUDA_ARCH_PTX CACHE)
  endif()

  set(__cuda_arch_ptx "")
  if(CUDA_GENERATION STREQUAL "Fermi")
    set(__cuda_arch_bin "2.0")
  elseif(CUDA_GENERATION STREQUAL "Kepler")
    set(__cuda_arch_bin "3.0 3.5 3.7")
  elseif(CUDA_GENERATION STREQUAL "Maxwell")
    set(__cuda_arch_bin "5.0")
  elseif(CUDA_GENERATION STREQUAL "Pascal")
    set(__cuda_arch_bin "6.0")
  endif()

  if(NOT DEFINED __cuda_arch_bin)
    if(ARM)
      set(__cuda_arch_bin "3.2")
      set(__cuda_arch_ptx "")
    elseif(AARCH64)
      set(__cuda_arch_bin "5.3")
      set(__cuda_arch_ptx "")
    else()
      if(${CUDA_VERSION} VERSION_LESS "8.0")
        set(__cuda_arch_bin "2.0 3.0 3.5 3.7 5.0")
      else()
        set(__cuda_arch_bin "2.0 3.0 3.5 3.7 5.0 6.0")
      endif()
      set(__cuda_arch_ptx "")
    endif()
  endif()

  set(CUDA_ARCH_BIN ${__cuda_arch_bin} CACHE STRING "Specify 'real' GPU architectures to build binaries for, BIN(PTX) format is supported")
  set(CUDA_ARCH_PTX ${__cuda_arch_ptx} CACHE STRING "Specify 'virtual' PTX architectures to build PTX intermediate code for")

  string(REGEX REPLACE "\\." "" ARCH_BIN_NO_POINTS "${CUDA_ARCH_BIN}")
  string(REGEX REPLACE "\\." "" ARCH_PTX_NO_POINTS "${CUDA_ARCH_PTX}")
  
  # Ckeck if user specified 1.0 compute capability: we don't support it
  string(REGEX MATCH "1.0" HAS_ARCH_10 "${CUDA_ARCH_BIN} ${CUDA_ARCH_PTX}")
  set(CUDA_ARCH_BIN_OR_PTX_10 0)
  if(NOT ${HAS_ARCH_10} STREQUAL "")
    set(CUDA_ARCH_BIN_OR_PTX_10 1)
  endif()

  # NVCC flags to be set
  set(NVCC_FLAGS_EXTRA "")

  # These vars will be passed into the templates
  set(RobartsVTK_CUDA_ARCH_BIN "")
  set(RobartsVTK_CUDA_ARCH_PTX "")
  set(RobartsVTK_CUDA_ARCH_FEATURES "")

  # Tell NVCC to add binaries for the specified GPUs
  string(REGEX MATCHALL "[0-9()]+" ARCH_LIST "${ARCH_BIN_NO_POINTS}")
  foreach(ARCH IN LISTS ARCH_LIST)
    if(ARCH MATCHES "([0-9]+)\\(([0-9]+)\\)")
      # User explicitly specified PTX for the concrete BIN
      set(NVCC_FLAGS_EXTRA ${NVCC_FLAGS_EXTRA} -gencode arch=compute_${CMAKE_MATCH_2},code=sm_${CMAKE_MATCH_1})
      set(RobartsVTK_CUDA_ARCH_BIN "${RobartsVTK_CUDA_ARCH_BIN} ${CMAKE_MATCH_1}")
      set(RobartsVTK_CUDA_ARCH_FEATURES "${RobartsVTK_CUDA_ARCH_FEATURES} ${CMAKE_MATCH_2}")
    else()
      # User didn't explicitly specify PTX for the concrete BIN, we assume PTX=BIN
      set(NVCC_FLAGS_EXTRA ${NVCC_FLAGS_EXTRA} -gencode arch=compute_${ARCH},code=sm_${ARCH})
      set(RobartsVTK_CUDA_ARCH_BIN "${RobartsVTK_CUDA_ARCH_BIN} ${ARCH}")
      set(RobartsVTK_CUDA_ARCH_FEATURES "${RobartsVTK_CUDA_ARCH_FEATURES} ${ARCH}")
    endif()
  endforeach()
  set(NVCC_FLAGS_EXTRA ${NVCC_FLAGS_EXTRA} -D_FORCE_INLINES)

  # Tell NVCC to add PTX intermediate code for the specified architectures
  string(REGEX MATCHALL "[0-9]+" ARCH_LIST "${ARCH_PTX_NO_POINTS}")
  foreach(ARCH IN LISTS ARCH_LIST)
    set(NVCC_FLAGS_EXTRA ${NVCC_FLAGS_EXTRA} -gencode arch=compute_${ARCH},code=compute_${ARCH})
    set(RobartsVTK_CUDA_ARCH_PTX "${RobartsVTK_CUDA_ARCH_PTX} ${ARCH}")
    set(RobartsVTK_CUDA_ARCH_FEATURES "${RobartsVTK_CUDA_ARCH_FEATURES} ${ARCH}")
  endforeach()

  # These vars will be processed in other scripts
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ${NVCC_FLAGS_EXTRA})
  set(RobartsVTK_CUDA_CC "${NVCC_FLAGS_EXTRA}")

  message(STATUS "CUDA NVCC target flags: ${CUDA_NVCC_FLAGS}")

  # Look for OpenGL
  FIND_PACKAGE(OpenGL REQUIRED)

  # Add Robart's CUDA libraries
  ADD_SUBDIRECTORY(Common/vtkCudaCommon)

  SET(RobartsVTK_LIBRARIES ${RobartsVTK_LIBRARIES} vtkCudaCommon CACHE INTERNAL "")

  ADD_SUBDIRECTORY(Common/vtkCudaCommonStatic)

  SET(RobartsVTK_LIBRARIES ${RobartsVTK_LIBRARIES} vtkCudaCommonStatic CACHE INTERNAL "")
ENDIF ()

#-----------------------------------------------------------------------------
# Add the CUDA Visualization Libraries
#-----------------------------------------------------------------------------
IF (RobartsVTK_USE_CUDA)
  OPTION (RobartsVTK_USE_CUDA_VISUALIZATION "Build the CUDA visualization modules" ON)
  IF (RobartsVTK_USE_CUDA_VISUALIZATION)
    FIND_PACKAGE(OpenGL REQUIRED)

    ADD_SUBDIRECTORY(Visualization/vtkCudaVisualization)

    SET(RobartsVTK_LIBRARIES ${RobartsVTK_LIBRARIES} vtkCudaVisualization CACHE INTERNAL "")
  ENDIF()
ENDIF()

#-----------------------------------------------------------------------------
# Add the Robarts Visualization Libraries
#-----------------------------------------------------------------------------
OPTION (RobartsVTK_USE_VISUALIZATION "Build the Robarts Visualization modules" OFF)
IF(RobartsVTK_USE_VISUALIZATION)
  FIND_PACKAGE(OpenGL REQUIRED)
  FIND_PACKAGE(FFTW REQUIRED)
  MESSAGE(STATUS "Using FFTW available at: ${FFTW_ROOT_DIR}")

  IF(NOT PNG_CMAKE_FILE)
    MESSAGE(FATAL_ERROR "libpng include file not specified. Unable to find PNG library. Please set PNG_CMAKE_FILE:FILEPATH to the location of libpng16.cmake")
  ENDIF()
  INCLUDE(${PNG_CMAKE_FILE})
  
  MESSAGE(STATUS "Using libpng available at: ${PNG_CMAKE_FILE}")
  MESSAGE(STATUS "Using zlib available at: ${ZLIB_ROOT}")

  ADD_SUBDIRECTORY(Visualization/vtkRobartsVisualization)

  SET(RobartsVTK_LIBRARIES ${RobartsVTK_LIBRARIES} vtkRobartsVisualization CACHE INTERNAL "")
ENDIF()

#-----------------------------------------------------------------------------
# Add the CUDA Analytics Libraries
#-----------------------------------------------------------------------------
IF (RobartsVTK_USE_CUDA AND RobartsVTK_USE_COMMON)
  OPTION (RobartsVTK_USE_CUDA_ANALYTICS "Build the CUDA image analytics modules" ON)
  IF (RobartsVTK_USE_CUDA_ANALYTICS)
    ADD_SUBDIRECTORY(ImageProcessing/vtkCudaImageAnalytics)

    SET(RobartsVTK_LIBRARIES ${RobartsVTK_LIBRARIES} vtkCudaImageAnalytics CACHE INTERNAL "")
  ENDIF()
ENDIF()

#-----------------------------------------------------------------------------
# OpenCL Dependencies
#-----------------------------------------------------------------------------
IF (RobartsVTK_USE_OpenCL)
  FIND_PACKAGE(OpenCL REQUIRED)

  ADD_LIBRARY(OpenCL IMPORTED SHARED)

  IF(${OpenCL_LIBRARY} MATCHES "${CMAKE_SHARED_LIBRARY_SUFFIX}")
    SET_TARGET_PROPERTIES( OpenCL PROPERTIES IMPORTED_LOCATION ${OpenCL_LIBRARY} )
  ELSEIF(${OpenCL_LIBRARY} MATCHES "${CMAKE_STATIC_LIBRARY_SUFFIX}")
    SET_TARGET_PROPERTIES( OpenCL PROPERTIES IMPORTED_IMPLIB ${OpenCL_LIBRARY} )
  ENDIF()  

  OPTION (RobartsVTK_USE_CL_VOLUMERECONSTRUCTION "Build the OpenCL Volume Reconstruction modules" ON)
  IF (RobartsVTK_USE_CL_VOLUMERECONSTRUCTION)
    ADD_SUBDIRECTORY(ImageProcessing/vtkCLVolumeReconstruction)

    SET(RobartsVTK_LIBRARIES ${RobartsVTK_LIBRARIES} vtkCLVolumeReconstruction CACHE INTERNAL "")
  ENDIF()
ENDIF()

#-----------------------------------------------------------------------------
# Applications
#-----------------------------------------------------------------------------
OPTION(RobartsVTK_BUILD_APPS "Build RobartsVTK applications." OFF)
IF(RobartsVTK_BUILD_APPS)
  IF(RobartsVTK_USE_CUDA)
    ADD_SUBDIRECTORY(Applications/Cuda2DTFVolumeRendering)
    set_target_properties(Cuda2DTFVolumeRendering PROPERTIES FOLDER Applications)
    IF(RobartsVTK_USE_QT)
      ADD_SUBDIRECTORY(Applications/FluoroPredViz)
      set_target_properties(FluoroPredViz PROPERTIES FOLDER Applications)
      ADD_SUBDIRECTORY(Applications/TransferFunctionGUI)
      set_target_properties(TFUIDual PROPERTIES FOLDER Applications)
      set_target_properties(TFUIGradient PROPERTIES FOLDER Applications)
      set_target_properties(TFUIKohonen PROPERTIES FOLDER Applications)
    ENDIF()
  ENDIF()
  
  IF(RobartsVTK_USE_CL_VOLUMERECONSTRUCTION AND RobartsVTK_USE_PLUS)
    #-------------------------------------
    # Find OpenCV
    #-------------------------------------
    FIND_PACKAGE(OpenCV 3.1.0 REQUIRED NO_MODULE)

    IF(RobartsVTK_USE_VISUALIZATION)
      IF(RobartsVTK_USE_CUDA_VISUALIZATION)
        ADD_SUBDIRECTORY(Applications/StreamingUSVis)
        set_target_properties(StreamingUSVis PROPERTIES FOLDER Applications)
		ADD_SUBDIRECTORY(Applications/Cuda1DTFVolumeRendering)
        set_target_properties(Cuda1DTFVolumeRendering PROPERTIES FOLDER Applications)
		ADD_SUBDIRECTORY(Applications/CLReconstruction)
		set_target_properties(CLReconstruction PROPERTIES FOLDER Applications)
		ADD_SUBDIRECTORY(Applications/CLReconstructionWithVisualization)
		set_target_properties(CLReconstructionWithVisualization PROPERTIES FOLDER Applications)
      ENDIF()
      ADD_SUBDIRECTORY(Applications/KeyholeRendering)
      set_target_properties(KeyholeRendering PROPERTIES FOLDER Applications)
    ENDIF()
  ENDIF()

  IF(RobartsVTK_USE_COMMON AND RobartsVTK_USE_CUDA AND RobartsVTK_USE_CUDA_ANALYTICS)
    ADD_SUBDIRECTORY(Applications/MaxFlow)
    set_target_properties(MaxFlow GHMFSegment KSOMTrain KSOMApply PROPERTIES FOLDER Applications)
  ENDIF()
  
  IF(RobartsVTK_USE_PLUS AND RobartsVTK_USE_QT)
    #-------------------------------------
    # Find OpenCV
    #-------------------------------------
    FIND_PACKAGE(OpenCV 3.1.0 REQUIRED NO_MODULE)

    FIND_PACKAGE(PlusApp REQUIRED)
    MESSAGE(STATUS "Using PlusApp available at: ${PlusApp_DIR}")

    ADD_SUBDIRECTORY(Applications/CameraCalibration)
    set_target_properties(CameraCalibration PROPERTIES FOLDER Applications)
  ENDIF()
  
  IF(OpenCV_FOUND)
    MESSAGE(STATUS "Using OpenCV available at: ${OpenCV_DIR}")
  ENDIF()
ENDIF()

SET(RobartsVTK_INCLUDE_DIRS ${CMAKE_BINARY_DIR})
#-----------------------------------------------------------------------------
# Populate export variables
#-----------------------------------------------------------------------------
foreach( lib ${RobartsVTK_LIBRARIES} )
  GET_TARGET_PROPERTY(lib_DIRS ${lib} INTERFACE_INCLUDE_DIRECTORIES)
  LIST(APPEND RobartsVTK_INCLUDE_DIRS ${lib_DIRS})
endforeach()
LIST(REMOVE_DUPLICATES RobartsVTK_INCLUDE_DIRS)

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
  ${CMAKE_CURRENT_BINARY_DIR}/RobartsVTKConfig.cmake
  )
CONFIGURE_FILE(
  ${CMAKE_CURRENT_SOURCE_DIR}/RobartsVTKConfigVersion.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/RobartsVTKConfigVersion.cmake 
  @ONLY)