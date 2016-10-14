# --------------------------------------------------------------------------
# Library export directive file generation

# This macro generates a ...Export.h file that specifies platform-specific DLL export directives,
# for example on Windows: __declspec( dllexport )
MACRO(GENERATE_EXPORT_DIRECTIVE_FILE LIBRARY_NAME)
  SET (MY_LIBNAME ${LIBRARY_NAME})
  SET (MY_EXPORT_HEADER_PREFIX ${MY_LIBNAME})
  SET (MY_LIBRARY_EXPORT_DIRECTIVE "${MY_LIBNAME}Export")
  CONFIGURE_FILE(
    ${RobartsVTK_Export_Template}
    ${CMAKE_CURRENT_BINARY_DIR}/${MY_EXPORT_HEADER_PREFIX}Export.h
    )
ENDMACRO()

macro(REMOVE_VTK_DEFINITIONS)
  get_directory_property(_dir_defs DIRECTORY ${CMAKE_SOURCE_DIR} COMPILE_DEFINITIONS)
  set(_vtk_definitions)
  foreach(_item ${_dir_defs})
      if(_item MATCHES "vtk*")
          list(APPEND _vtk_definitions -D${_item})
      endif()
  endforeach()
  remove_definitions(${_vtk_definitions})
endmacro()