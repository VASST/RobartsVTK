macro(REMOVE_VTK_DEFINITIONS)
  # Remove vtk definitions
  # This is used for CUDA targets, because nvcc does not like VTK 6+ definitions style.

  get_directory_property(_dir_defs DIRECTORY ${CMAKE_SOURCE_DIR} COMPILE_DEFINITIONS)
  set(_vtk_definitions)
  foreach(_item ${_dir_defs})
      if(_item MATCHES "vtk*")
          list(APPEND _vtk_definitions -D${_item})
      endif()
  endforeach()
  remove_definitions(${_vtk_definitions})
endmacro(REMOVE_VTK_DEFINITIONS)

###############################################################################
###############################################################################
# ADD LIBRARY
###############################################################################
###############################################################################
macro(cuda_vtk_add_library cuda_target)
  REMOVE_VTK_DEFINITIONS()

  CUDA_ADD_CUDA_INCLUDE_ONCE()

  # Separate the sources from the options
  CUDA_GET_SOURCES_AND_OPTIONS(_sources _cmake_options _options ${ARGN})
  CUDA_BUILD_SHARED_LIBRARY(_cuda_shared_flag ${ARGN})
  # Create custom commands and targets for each file.
  CUDA_WRAP_SRCS( ${cuda_target} OBJ _generated_files ${_sources}
    ${_cmake_options} ${_cuda_shared_flag}
    OPTIONS ${_options} )

  # Compute the file name of the intermedate link file used for separable
  # compilation.
  CUDA_COMPUTE_SEPARABLE_COMPILATION_OBJECT_FILE_NAME(link_file ${cuda_target} "${${cuda_target}_SEPARABLE_COMPILATION_OBJECTS}")

  # Add the library.
  vtk_module_library(${cuda_target} ${_cmake_options}
    ${_generated_files}
    ${_sources}
    ${link_file}
    )

  # Add a link phase for the separable compilation if it has been enabled.  If
  # it has been enabled then the ${cuda_target}_SEPARABLE_COMPILATION_OBJECTS
  # variable will have been defined.
  CUDA_LINK_SEPARABLE_COMPILATION_OBJECTS("${link_file}" ${cuda_target} "${_options}" "${${cuda_target}_SEPARABLE_COMPILATION_OBJECTS}")

  vtk_module_link_libraries(${cuda_target} LINK_PRIVATE
    ${CUDA_LIBRARIES}
    )

  # We need to set the linker language based on what the expected generated file
  # would be. CUDA_C_OR_CXX is computed based on CUDA_HOST_COMPILATION_CPP.
  set_target_properties(${cuda_target}
    PROPERTIES
    LINKER_LANGUAGE ${CUDA_C_OR_CXX}
    )

endmacro()

###############################################################################
###############################################################################
# ADD EXECUTABLE
###############################################################################
###############################################################################
macro(cuda_vtk_add_executable cuda_target)
  REMOVE_VTK_DEFINITIONS()

  CUDA_ADD_CUDA_INCLUDE_ONCE()

  # Separate the sources from the options
  CUDA_GET_SOURCES_AND_OPTIONS(_sources _cmake_options _options ${ARGN})
  # Create custom commands and targets for each file.
  CUDA_WRAP_SRCS( ${cuda_target} OBJ _generated_files ${_sources} OPTIONS ${_options} )

  # Compute the file name of the intermedate link file used for separable
  # compilation.
  CUDA_COMPUTE_SEPARABLE_COMPILATION_OBJECT_FILE_NAME(link_file ${cuda_target} "${${cuda_target}_SEPARABLE_COMPILATION_OBJECTS}")

  # Add the library.
  vtk_add_executable(${cuda_target} ${_cmake_options}
    ${_generated_files}
    ${_sources}
    ${link_file}
    )

  # Add a link phase for the separable compilation if it has been enabled.  If
  # it has been enabled then the ${cuda_target}_SEPARABLE_COMPILATION_OBJECTS
  # variable will have been defined.
  CUDA_LINK_SEPARABLE_COMPILATION_OBJECTS("${link_file}" ${cuda_target} "${_options}" "${${cuda_target}_SEPARABLE_COMPILATION_OBJECTS}")

  vtk_module_link_libraries(${cuda_target} LINK_PRIVATE
    ${CUDA_LIBRARIES}
    )

  # We need to set the linker language based on what the expected generated file
  # would be. CUDA_C_OR_CXX is computed based on CUDA_HOST_COMPILATION_CPP.
  set_target_properties(${cuda_target}
    PROPERTIES
    LINKER_LANGUAGE ${CUDA_C_OR_CXX}
    )

endmacro()