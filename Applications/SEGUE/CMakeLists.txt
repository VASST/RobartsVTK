PROJECT(SEGUE)

SET ( ${PROJECT_NAME}_SRCS 
  LURange.cpp
  main.cxx
  qcolortablewidget.cpp
  QHierarchyWidget.cxx
  QLabellingWidget.cxx
  QSmoothnessScalarWidget.cxx
  qtcolortriangle.cpp
  SEGUEMainWindow.cxx
  SliceViewer.cpp
  vtkImageLuminance2.cpp
)

IF( MSVC OR ${CMAKE_GENERATOR} MATCHES "Xcode")
  SET ( ${PROJECT_NAME}_HDRS 
    LURange.h
    qcolortablewidget.h
    QHierarchyWidget.h
    QLabellingWidget.h
    QSmoothnessScalarWidget.h
    qtcolortriangle.h
    SEGUEMainWindow.h
    SliceViewer.h
    vtkImageLuminance2.h
    QNumberAction.h
  )
ENDIF()

# -----------------------------------------------------------------
# Build the library
ADD_EXECUTABLE(${PROJECT_NAME} ${${PROJECT_NAME}_HDRS} ${${PROJECT_NAME}_SRCS})
include(${VTK_USE_FILE})
target_include_directories(${PROJECT_NAME} PUBLIC 
  ${VTK_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR} 
  ${${PROJECT_NAME}_INCLUDE_DIRS}
  )
target_link_libraries(${PROJECT_NAME} PUBLIC
  Qt5::Widgets
  vtkCudaCommon
  vtkCudaImageAnalytics
  vtkRobartsCommon
  vtkRenderingVolume${VTK_RENDERING_BACKEND}
  vtkRenderingGL2PSOpenGL2
  vtkRenderingVolume 
  vtkRenderingCore 
  vtkFiltersCore 
  vtkImagingCore 
  vtkIOImage 
  vtkCommonCore 
  vtkCommonDataModel
  vtkInteractionWidgets
  vtksys
  vtkIOLegacy
  vtkIOMINC
  vtkIOImage 
  vtkImagingMath
  vtkGUISupportQt
  vtkParallelCore
  )