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

set(VTK_PYTHON_MODULES_AND_KITS ${VTK_PYTHON_MODULES})
set(_vtk_python_modules_only ${VTK_PYTHON_MODULES})

if(VTK_ENABLE_KITS)
  set(_vtk_kits_with_suffix)
  set(VTK_KIT_SUFFIX "Kit") # Required to avoid conflict with targets like vtkFiltersPython
  # Create list of module that do not belong to any kits
  foreach(kit IN LISTS vtk_kits)
    # XXX Variable '_${kit}_modules' is set in vtkModuleTop.cmake
    list(REMOVE_ITEM _vtk_python_modules_only ${_${kit}_modules})
    list(APPEND _vtk_kits_with_suffix ${kit}${VTK_KIT_SUFFIX})
  endforeach()
  set(VTK_PYTHON_MODULES_AND_KITS ${_vtk_kits_with_suffix} ${_vtk_python_modules_only})
  # Loop through all kits that should wrapped, and wrap them.
  foreach(kit IN LISTS vtk_kits)
    set(kit_srcs)
    set(_module_depends)
    foreach(module IN LISTS _${kit}_modules)
      vtk_module_load(${module})
      vtk_module_headers_load(${module})
      vtk_include_recurse(${module})
      list(APPEND _module_depends ${${module}_DEPENDS})
    endforeach()
    vtk_add_python_wrapping("${_${kit}_modules}" kit_srcs ${kit}${VTK_KIT_SUFFIX})
    # Compose kit python module dependency list.
    set(_kit_depends)
    foreach(dep IN LISTS _module_depends)
      if(${dep}_KIT)
        list(APPEND _kit_depends ${${dep}_KIT}${VTK_KIT_SUFFIX})
      else()
        list(APPEND _kit_depends ${dep})
      endif()
    endforeach()
    list(REMOVE_DUPLICATES _kit_depends)
    set(${kit}${VTK_KIT_SUFFIX}_DEPENDS ${_kit_depends})
    # Wrap kit.
    vtk_add_python_wrapping_library(${kit}${VTK_KIT_SUFFIX} kit_srcs ${kit})
  endforeach()
endif()

# Loop through all modules that should be wrapped, and wrap them.
foreach(module IN LISTS _vtk_python_modules_only)
  vtk_module_load(${module})
  vtk_module_headers_load(${module})
  vtk_include_recurse(${module})
  vtk_add_python_wrapping(${module} module_srcs)
  set(_saved_${module}_DEPENDS ${${module}_DEPENDS})
  # Update module dependency to consider dependent kits if any.
  if(VTK_ENABLE_KITS)
    set(_module_depends_with_kits)
    foreach(dep IN LISTS ${module}_DEPENDS)
      if(${dep}_KIT)
        list(APPEND _module_depends_with_kits ${${dep}_KIT}${VTK_KIT_SUFFIX})
      else()
        list(APPEND _module_depends_with_kits ${dep})
      endif()
    endforeach()
    list(REMOVE_DUPLICATES _module_depends_with_kits)
    set(${module}_DEPENDS ${_module_depends_with_kits})
  endif()
  vtk_add_python_wrapping_library(${module} module_srcs ${module})
  
  # TODO : understand kit system so that these links aren't hard coded
  target_link_libraries(${module}PythonD LINK_PUBLIC vtkCommonKitPythonD vtkRenderingKitPythonD )

  # Restore regular list of dependency so that it can be used later.
  set(${module}_DEPENDS ${_saved_${module}_DEPENDS})
  unset(_saved_${module}_DEPENDS)
endforeach()

# Handle out-of-source builds correctly.
#
#  1. Create a list of Python files to be installed/copied.
#  2. Copy them to VTK_BINARY_DIR if it is different from VTK_SOURCE_DIR.
#  3. Use Python's compileall to compile the copied files.
#
# *** Step 1 has to be done carefully to avoid missing out files ***

if(PYTHON_EXECUTABLE)
  # Wrapping/Python/robartsvtk/*.py
  unset(VTK_PYTHON_FILES)
  unset(VTK_PYTHON_IMPORT_ALL)
  foreach(module IN LISTS VTK_PYTHON_MODULES_AND_KITS)
    set(VTK_PYTHON_IMPORT_ALL "${VTK_PYTHON_IMPORT_ALL}from ${module} import *\n")
    configure_file(robartsvtk/module.py.in robartsvtk/${module}.py @ONLY)
    list(APPEND VTK_PYTHON_FILES
      "${CMAKE_CURRENT_BINARY_DIR}/robartsvtk/${module}.py")
  endforeach()
  configure_file(robartsvtk/__init__.py.in robartsvtk/__init__.py @ONLY)
  list(APPEND VTK_PYTHON_FILES
    "${CMAKE_CURRENT_BINARY_DIR}/robartsvtk/__init__.py")

  # Now copy these files if necessary.
  unset(VTK_PYTHON_SOURCE_FILES)
  unset(VTK_PYTHON_OUTPUT_FILES)
  foreach(file ${VTK_PYTHON_FILES})
    if (NOT IS_ABSOLUTE "${file}")
      set(src "${CMAKE_CURRENT_BINARY_DIR}/${file}.py")
      list(APPEND VTK_PYTHON_SOURCE_FILES "${src}")
    endif ()
  endforeach()

  if(NOT "${VTK_BINARY_DIR}" MATCHES "^${VTK_SOURCE_DIR}$")
    foreach(file ${VTK_PYTHON_FILES})
      if (NOT IS_ABSOLUTE "${file}")
        set(src "${CMAKE_CURRENT_SOURCE_DIR}/${file}.py")
        set(tgt "${CMAKE_CURRENT_BINARY_DIR}/${file}.py")
        add_custom_command(DEPENDS ${src}
          COMMAND ${CMAKE_COMMAND} -E copy ${src} ${tgt}
          OUTPUT ${tgt}
          COMMENT "source copy")
      endif ()
    endforeach()
  endif()

  # Byte compile the Python files.
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/compile_all_vtk.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/compile_all_vtk.py
    @ONLY)
  add_custom_command(
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/compile_all_vtk.py
    DEPENDS ${VTK_PYTHON_SOURCE_FILES} ${CMAKE_CURRENT_BINARY_DIR}/compile_all_vtk.py
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/robartsvtk_compile_complete"
    )

  # Create a target on which custom commands can depend.  When this
  # is up-to-date, it is safe to run VTK python code.  Therefore
  # this should also depend on the vtkpython executable.
  add_custom_target(vtkpython_pyc ALL
    DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/robartsvtk_compile_complete")
  add_dependencies(vtkpython_pyc
    vtkpython
    )

  # If no runtime is to be installed then do not install python modules.
  if(NOT VTK_INSTALL_NO_RUNTIME)

    # Install python modules
    install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/robartsvtk"
      DESTINATION ${VTK_INSTALL_PYTHON_MODULE_DIR} COMPONENT RuntimeLibraries
      USE_SOURCE_PERMISSIONS)

    # Install python extension library that backs the modules
    if (BUILD_SHARED_LIBS AND NOT VTK_INSTALL_NO_LIBRARIES)
      foreach(module IN LISTS VTK_PYTHON_MODULES_AND_KITS)
        install(TARGETS ${module}Python
          RUNTIME DESTINATION ${VTK_INSTALL_RUNTIME_DIR} COMPONENT RuntimeLibraries
          LIBRARY DESTINATION ${VTK_INSTALL_PYTHON_MODULE_DIR}/robartsvtk COMPONENT RuntimeLibraries
          ARCHIVE DESTINATION ${VTK_INSTALL_ARCHIVE_DIR} COMPONENT Development
          )
      endforeach()
    endif()

    # Install the conveniently configured python interpretters
    if(NOT VTK_INSTALL_NO_PYTHON_EXES AND VTK_ENABLE_VTKPYTHON)
      # Install the vtkpython executable
      install(TARGETS vtkpython
        DESTINATION ${VTK_INSTALL_RUNTIME_DIR})

      if(PVTKPYTHON_EXECUTABLE)
        # Install the mpi enabled vtkpython executable
        install(TARGETS pvtkpython
          DESTINATION ${VTK_INSTALL_RUNTIME_DIR})
      endif()
    endif()

  endif()

endif()
