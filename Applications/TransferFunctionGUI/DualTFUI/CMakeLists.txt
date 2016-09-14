PROJECT(TFUIDual)

include(module.cmake)
vtk_module_impl()

SET ( Module_SRCS 
  qTransferFunctionWindowWidget.cxx
  qHistogramHolderLabel.cxx
  qTransferFunctionDefinitionWidget.cxx
  qSegmentationWidget.cxx
  main.cxx
)

LIST(APPEND ${PROJECT_NAME}_INCLUDE_DIRS ${Qt5Widgets_INCLUDE_DIRS})
SET(${PROJECT_NAME}_APPEND_LIBS Qt5::Widgets ${_Qt5Widgets_LIB_DEPENDENCIES})
LIST(APPEND ${PROJECT_NAME}_LIBS ${${PROJECT_NAME}_APPEND_LIBS})

SET(${PROJECT_NAME}_INCLUDE_DIRS ${VTK_INCLUDE_DIRS})

# -----------------------------------------------------------------
# Build the executable
vtk_add_executable(${PROJECT_NAME} ${Module_SRCS})
foreach(dep IN LISTS ${PROJECT_NAME}_DEPENDS)
  vtk_module_link_libraries(${PROJECT_NAME} LINK_PUBLIC ${dep})
endforeach()
vtk_module_link_libraries(${PROJECT_NAME} LINK_PUBLIC ${${PROJECT_NAME}_LIBS})
vtk_target_install(${PROJECT_NAME})
vtk_module_export(${Module_SRCS})

# -----------------------------------------------------------------
LIST(APPEND VTK_MODULES_ENABLED ${PROJECT_NAME})
SET(VTK_MODULES_ENABLED ${VTK_MODULES_ENABLED} PARENT_SCOPE)