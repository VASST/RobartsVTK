PROJECT(vtkRobartsRegistration)

SET ( ${PROJECT_NAME}_SRCS
  vtkShapeBasedInterpolation.cxx
  vtkTPSRegistration.cxx
  vtkPrincipalComponentAnalysis.cxx
  vtkPowellMinimizer.cxx
  vtkPolyDataSurfaceArea.cxx
  vtkPolyDataNormals2.cxx
  vtkPolyDataCorrespondence.cxx
  vtkMeshSmootheness.cxx
  vtkImageTMIManipulator.cxx
  vtkImageSMIPVIManipulator.cxx
  vtkImageSMIManipulator.cxx
  vtkImageSMIManipulator2.cxx
  vtkImageSDManipulator.cxx
  vtkImageRMIManipulator.cxx
  vtkImagePatternIntensity.cxx
  vtkImageNormalizedCrossCorrelation.cxx
  vtkImageNMIManipulator.cxx
  vtkImageNCCManipulator.cxx
  vtkImageMIManipulator.cxx
  vtkImageECRManipulator.cxx
  vtkImageADManipulator.cxx
  vtkImageAbsoluteDifference.cxx
  vtkImage3DNoiseSource.cxx
  vtkGBEFixed.cxx
  vtkGBE.cxx
  vtkFunctionMinimizer.cxx
  vtkCompactSupportRBFTransform.cxx
)

IF( MSVC OR ${CMAKE_GENERATOR} MATCHES "Xcode")
  SET ( ${PROJECT_NAME}_HDRS 
    vtkShapeBasedInterpolation.h
    vtkTPSRegistration.h
    vtkPrincipalComponentAnalysis.h
    vtkPowellMinimizer.h
    vtkPolyDataSurfaceArea.h
    vtkPolyDataNormals2.h
    vtkPolyDataCorrespondence.h
    vtkMeshSmootheness.h
    vtkImageTMIManipulator.h
    vtkImageSMIPVIManipulator.h
    vtkImageSMIManipulator.h
    vtkImageSMIManipulator2.h
    vtkImageSDManipulator.h
    vtkImageRMIManipulator.h
    vtkImagePatternIntensity.h
    vtkImageNormalizedCrossCorrelation.h
    vtkImageNMIManipulator.h
    vtkImageNCCManipulator.h
    vtkImageMIManipulator.h
    vtkImageECRManipulator.h
    vtkImageADManipulator.h
    vtkImageAbsoluteDifference.h
    vtkImage3DNoiseSource.h
    vtkGBEFixed.h
    vtkGBE.h
    vtkFunctionMinimizer.h
    vtkCompactSupportRBFTransform.h
  )
ENDIF()

# -----------------------------------------------------------------
# Build the library
ADD_LIBRARY(${PROJECT_NAME} ${${PROJECT_NAME}_HDRS} ${${PROJECT_NAME}_SRCS})
target_include_directories(${PROJECT_NAME} PUBLIC 
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR} 
  ${VTK_INCLUDE_DIRS}
  )
target_link_libraries(${PROJECT_NAME} PUBLIC
  vtkPlusCommon
  vtkImagingHybrid 
  vtkImagingCore
  vtkImagingStatistics
  vtkFiltersGeneral
  vtkFiltersCore
  vtkImagingMath
  vtkCommonDataModel
  vtkFiltersGeneral
  )
GENERATE_EXPORT_DIRECTIVE_FILE(${PROJECT_NAME})