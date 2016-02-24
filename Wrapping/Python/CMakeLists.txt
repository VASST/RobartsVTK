# This is where we actually wrap the modules that have requested it.
include(vtkPythonWrapping)

get_property(VTK_PYTHON_MODULES GLOBAL PROPERTY VTK_PYTHON_WRAPPED)

# Get the include directories for the module and all its dependencies.
macro(vtk_include_recurse module)
  if(TARGET ${module})
    get_target_property(${module}_USE_CUDA ${module} module_uses_cuda)
    if(${module}_USE_CUDA)
      include_directories(${CUDA_INCLUDE_DIRS})
    endif()
  endif()
  _vtk_module_config_recurse("${module}_PYTHON" ${module})
  include_directories(${${module}_PYTHON_INCLUDE_DIRS})
endmacro()

# Loop through all modules that should be wrapped, and wrap them.
foreach(module ${VTK_PYTHON_MODULES})
  vtk_module_load(${module})
  vtk_module_headers_load(${module})
  vtk_include_recurse(${module})
  vtk_add_python_wrapping(${module})
endforeach()

# Handle out-of-source builds correctly.
#
#  1. Create a list of Python files to be installed/copied.
#  2. Copy them to VTK_BINARY_DIR if it is different from VTK_SOURCE_DIR.
#  3. Use Python's compileall to compile the copied files.
#
# *** Step 1 has to be done carefully to avoid missing out files ***

if(PYTHON_EXECUTABLE)
  # Now create a list of Python files.

  # Wrapping/Python/robartsvtk/*.py
  unset(VTK_PYTHON_FILES)
  unset(VTK_PYTHON_IMPORT_ALL)
  foreach(module ${VTK_PYTHON_MODULES})
    set(VTK_PYTHON_IMPORT_ALL "${VTK_PYTHON_IMPORT_ALL}from ${module} import *\n")
    configure_file(robartsvtk/module.py.in robartsvtk/${module}.py @ONLY)
  endforeach()
  configure_file(robartsvtk/__init__.py.in robartsvtk/__init__.py @ONLY)

  # Now copy these files if necessary.
  unset(VTK_PYTHON_SOURCE_FILES)
  unset(VTK_PYTHON_OUTPUT_FILES)
  foreach(file ${VTK_PYTHON_FILES})
    set(src "${CMAKE_CURRENT_BINARY_DIR}/${file}.py")
    set(VTK_PYTHON_SOURCE_FILES ${VTK_PYTHON_SOURCE_FILES} ${src})
  endforeach()

  if(NOT "${VTK_BINARY_DIR}" MATCHES "^${VTK_SOURCE_DIR}$")
    foreach(file ${VTK_PYTHON_FILES})
      set(src "${CMAKE_CURRENT_SOURCE_DIR}/${file}.py")
      set(tgt "${CMAKE_CURRENT_BINARY_DIR}/${file}.py")
      add_custom_command(DEPENDS ${src}
        COMMAND ${CMAKE_COMMAND} -E copy ${src} ${tgt}
        OUTPUT ${tgt}
        COMMENT "source copy")
    endforeach()
  endif()

  # Byte compile the Python files.
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/compile_all_vtk.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/compile_all_vtk.py
    @ONLY)
  add_custom_command(
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/compile_all_vtk.py
    DEPENDS ${VTK_PYTHON_SOURCE_FILES} ${CMAKE_CURRENT_BINARY_DIR}/compile_all_vtk.py
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/vtk_compile_complete"
    )

  # Create a target on which custom commands can depend.  When this
  # is up-to-date, it is safe to run VTK python code.  Therefore
  # this should also depend on the vtkpython executable.
  add_custom_target(vtkpython_pyc ALL
    DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/vtk_compile_complete")
  add_dependencies(vtkpython_pyc
    vtkpython
    )

  # Install python modules
  install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/robartsvtk"
    DESTINATION ${VTK_INSTALL_PYTHON_MODULE_DIR} COMPONENT RuntimeLibraries
    USE_SOURCE_PERMISSIONS)

  # Install python extension library that backs the modules
  if (BUILD_SHARED_LIBS)
    foreach(module ${VTK_PYTHON_MODULES})
      install(TARGETS ${module}Python
        RUNTIME DESTINATION ${VTK_INSTALL_RUNTIME_DIR} COMPONENT RuntimeLibraries
        LIBRARY DESTINATION ${VTK_INSTALL_PYTHON_MODULE_DIR}/robartsvtk COMPONENT RuntimeLibraries
        ARCHIVE DESTINATION ${VTK_INSTALL_ARCHIVE_DIR} COMPONENT Development
        )
    endforeach()
  endif()
endif()
