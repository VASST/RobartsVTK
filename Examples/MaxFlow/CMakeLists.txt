PROJECT( MaxFlowExample )

vtk_module_impl()

SET ( Module_SRCS 
  MaxFlowExample.cxx
)

# -----------------------------------------------------------------
# Build the library
vtk_add_executable(${PROJECT_NAME} ${Module_SRCS})
foreach(dep IN LISTS ${PROJECT_NAME}_DEPENDS)
  vtk_module_link_libraries(${PROJECT_NAME} LINK_PUBLIC ${dep})
endforeach()
vtk_module_link_libraries(${PROJECT_NAME} LINK_PUBLIC ${${PROJECT_NAME}_APPEND_LIBS})
vtk_target_install(${PROJECT_NAME})

# -----------------------------------------------------------------
LIST(APPEND VTK_MODULES_ENABLED ${PROJECT_NAME})
SET(VTK_MODULES_ENABLED ${VTK_MODULES_ENABLED} PARENT_SCOPE)