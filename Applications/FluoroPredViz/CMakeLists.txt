PROJECT( FluoroPredViz )

SET ( ${PROJECT_NAME}_SRCS
  main.cxx
  FluoroPredViz.cxx
  ResizableQVTKWidget.cxx
)

IF( MSVC OR ${CMAKE_GENERATOR} MATCHES "Xcode")
  SET ( ${PROJECT_NAME}_HDRS 
    FluoroPredViz.h
    ResizableQVTKWidget.h
    )
ENDIF()

# -----------------------------------------------------------------
# Build the executable
ADD_EXECUTABLE(${PROJECT_NAME} ${${PROJECT_NAME}_HDRS} ${${PROJECT_NAME}_SRCS})
target_include_directories(${PROJECT_NAME} PUBLIC ${VTK_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PUBLIC
  Qt5::Widgets
  vtkCudaCommon
  vtkCudaVisualization
  vtkCudaImageAnalytics
  vtkRobartsCommon
  vtkRenderingVolume${VTK_RENDERING_BACKEND}
  vtkRenderingCore 
  vtkFiltersCore 
  vtkImagingCore 
  vtkInteractionWidgets 
  vtkGUISupportQt 
  vtkIOCore 
  vtkCommonCore 
  vtkIOMINC 
  )