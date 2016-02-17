IF(ITK_DIR)
  # ITK has been built already
  FIND_PACKAGE(ITK REQUIRED NO_MODULE PATHS ${ITK_DIR} NO_DEFAULT_PATH)
  
  MESSAGE(STATUS "Using ITK available at: ${ITK_DIR}")
ELSE(ITK_DIR)
  # ITK has not been built yet, so download and build it as an external project
  SET (ITKv4_REPOSITORY ${GIT_PROTOCOL}://itk.org/ITK.git)
  SET (ITKv4_GIT_TAG 8d58556089399c11d51795d46d6b17c355af95dc) #v4.7.2 from 2015-04-30
  
  MESSAGE(STATUS "Downloading and building ITK from: ${GIT_PROTOCOL}://itk.org/ITK.git")

  SET (RobartsVTK_ITK_SRC_DIR "${ep_dependency_DIR}/itk")
  SET (RobartsVTK_ITK_DIR "${ep_dependency_DIR}/itk-bin" CACHE INTERNAL "Path to store itk binaries")
  ExternalProject_Add( itk
    PREFIX "${ep_dependency_DIR}/itk-prefix"
    SOURCE_DIR "${RobartsVTK_ITK_SRC_DIR}"
    BINARY_DIR "${RobartsVTK_ITK_DIR}"
    #--Download step--------------
    GIT_REPOSITORY "${ITKv4_REPOSITORY}"
    GIT_TAG "${ITKv4_GIT_TAG}"
    #--Configure step-------------
    CMAKE_ARGS 
      ${ep_common_args}
      -DCMAKE_RUNTIME_OUTPUT_DIRECTORY:STRING=${RobartsVTK_EXECUTABLE_OUTPUT_PATH}
      -DBUILD_SHARED_LIBS:BOOL=ON
      -DBUILD_TESTING:BOOL=OFF
      -DBUILD_EXAMPLES:BOOL=OFF
      -DKWSYS_USE_MD5:BOOL=ON
      -DITK_USE_REVIEW:BOOL=ON
      -DCMAKE_CXX_FLAGS:STRING=${ep_common_cxx_flags}
      -DCMAKE_C_FLAGS:STRING=${ep_common_c_flags}
      -DITK_WRAP_PYTHON:BOOL=OFF
      -DITK_LEGACY_REMOVE:BOOL=ON
      -DITK_LEGACY_SILENT:BOOL=ON
      -DKWSYS_USE_MD5:BOOL=ON
    #--Build step-----------------
    #--Install step-----------------
    INSTALL_COMMAND ""
    DEPENDS ${ITK_DEPENDENCIES}
    )

ENDIF(ITK_DIR)