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