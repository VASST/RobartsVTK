PROJECT( MaxFlow )

vtk_module_impl()

# -----------------------------------------------------------------
# Build the GHMF_Segment executable
SET ( Module_SRCS 
  GHMF_Segment.cxx
)

vtk_add_executable(GHMF_Segment ${Module_SRCS})
foreach(dep IN LISTS ${PROJECT_NAME}_DEPENDS)
  vtk_module_link_libraries(GHMF_Segment LINK_PUBLIC ${dep})
endforeach()
vtk_target_install(GHMF_Segment)

# -----------------------------------------------------------------
# Build the KSOM_train executable
SET ( Module_SRCS 
  KSOM_train.cxx
)

vtk_add_executable(KSOM_train ${Module_SRCS})
foreach(dep IN LISTS ${PROJECT_NAME}_DEPENDS)
  vtk_module_link_libraries(KSOM_train LINK_PUBLIC ${dep})
endforeach()
vtk_target_install(KSOM_train)

# -----------------------------------------------------------------
# Build the KSOM_apply executable
SET ( Module_SRCS 
  KSOM_apply.cxx
)

vtk_add_executable(KSOM_apply ${Module_SRCS})
foreach(dep IN LISTS ${PROJECT_NAME}_DEPENDS)
  vtk_module_link_libraries(KSOM_apply LINK_PUBLIC ${dep})
endforeach()
vtk_target_install(KSOM_apply)
# -----------------------------------------------------------------
LIST(APPEND VTK_MODULES_ENABLED ${PROJECT_NAME})
SET(VTK_MODULES_ENABLED ${VTK_MODULES_ENABLED} PARENT_SCOPE)