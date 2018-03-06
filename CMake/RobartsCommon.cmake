# --------------------------------------------------------------------------
# Library export directive file generation

# This macro generates a ...Export.h file that specifies platform-specific DLL export directives,
# for example on Windows: __declspec( dllexport )
macro(GENERATE_EXPORT_DIRECTIVE_FILE LIBRARY_NAME)
  set (MY_LIBNAME ${LIBRARY_NAME})
  set (MY_EXPORT_HEADER_PREFIX ${MY_LIBNAME})
  set (MY_LIBRARY_EXPORT_DIRECTIVE "${MY_LIBNAME}Export")
  configure_file(
    ${RobartsVTK_Export_Template}
    ${CMAKE_CURRENT_BINARY_DIR}/${MY_EXPORT_HEADER_PREFIX}Export.h
    )
endmacro()

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


macro(RobartsVTKInstallLibrary _target_name _variable_root)
  INSTALL(TARGETS ${_target_name} EXPORT RobartsVTK
    RUNTIME DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY} COMPONENT RuntimeLibraries
    LIBRARY DESTINATION ${CMAKE_LIBRARY_OUTPUT_DIRECTORY} COMPONENT RuntimeLibraries
    ARCHIVE DESTINATION ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY} COMPONENT Development
    )
  INSTALL(FILES ${${_variable_root}_HDRS}
    DESTINATION ${RobartsVTK_INCLUDE_INSTALL} COMPONENT Development
    )
  GET_TARGET_PROPERTY(_library_type ${_target_name} TYPE)
  IF(${_library_type} STREQUAL SHARED_LIBRARY AND MSVC)
    INSTALL(FILES "$<TARGET_PDB_FILE:${_target_name}>" OPTIONAL
      DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY} COMPONENT RuntimeLibraries
      )
  ENDIF()
endmacro()