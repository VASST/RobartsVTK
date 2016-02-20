PROJECT(vtkRobartsRegistration)

include(module.cmake)

SET(${PROJECT_NAME}_INCLUDE_DIRS ${${PROJECT_NAME}_BINARY_DIR})
SET(${PROJECT_NAME}_PYTHON_INCLUDE_DIRS ${${PROJECT_NAME}_INCLUDE_DIRS})

# Add entries to project details
SET(${PROJECT_NAME}_INCLUDE_DIRS ${${PROJECT_NAME}_SOURCE_DIR} ${${PROJECT_NAME}_BINARY_DIR} CACHE INTERNAL "" FORCE)

SET ( Module_SRCS
  vtkShapeBasedInterpolation.cxx
  vtkTPSRegistration.cxx
  vtkPrincipalComponentAnalysis.cxx
  vtkPowellMinimizer.cxx
  vtkPolyDataSurfaceArea.cxx
  vtkPolyDataNormals2.cxx
  vtkPolyDataCorrespondence.cxx
  vtkMeshSmootheness.cxx
  vtkImageTMIManipulator.cxx
  vtkImageSquaredDifference.cxx
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
  vtkDeformedVolume.cxx
  vtkCompactSupportRBFTransform.cxx
  vtkCISGSubdivideGridTransform.cxx
)

if( MSVC )
  SET( ${vtk-module}_HDRS ${${vtk-module}_HDRS}
    vtkCISGSubdivideGridTransform.h
    vtkCompactSupportRBFTransform.h
    vtkDeformedVolume.h
    vtkFunctionMinimizer.h
    vtkGBE.h
    vtkGBEFixed.h
    vtkImage3DNoiseSource.h
    vtkImageAbsoluteDifference.h
    vtkImageADManipulator.h
    vtkImageECRManipulator.h
    vtkImageMIManipulator.h
    vtkImageNCCManipulator.h
    vtkImageNMIManipulator.h
    vtkImageNormalizedCrossCorrelation.h
    vtkImagePatternIntensity.h
    vtkImageRMIManipulator.h
    vtkImageSDManipulator.h
    vtkImageSMIManipulator.h
    vtkImageSMIManipulator2.h
    vtkImageSMIPVIManipulator.h
    vtkImageSquaredDifference.h
    vtkImageTMIManipulator.h
    vtkMeshSmootheness.h
    vtkPolyDataCorrespondence.h
    vtkPolyDataNormals2.h
    vtkPolyDataSurfaceArea.h
    vtkPowellMinimizer.h
    vtkPrincipalComponentAnalysis.h
    vtkShapeBasedInterpolation.h
    vtkTPSRegistration.h
    vtkTriangleMeshQuality.h
  )
endif()

# The following classes need re-writing to move from VTK4 to VTK6 design
if( ${RobartsVTK_Include_Outdated_Registration} )
  SET ( Module_SRCS ${Module_SRCS} 
    vtkImageTsallisMutualInformation.cxx
    vtkImageSquaredDifference.cxx
    vtkImageShannonMutualInformation.cxx
    vtkImageResliceBSpline.cxx
    vtkImageRenyiMutualInformation.cxx
    vtkImageProbeSum.cxx
    vtkImageNormalizedMutualInformation.cxx
    vtkImageGridStatistics.cxx
    vtkImageDataStatistics.cxx
    vtkImageCrossCorrelation.cxx
    vtkImageAmoebaGrid.cxx
    vtkGridTransformBSpline.cxx
    vtkCalcCrossCorrelation.cxx
    vtkCalcCentroid.cxx
    vtkImageMean3D.cxx
  )
  if( MSVC )
    SET( ${vtk-module}_HDRS ${${vtk-module}_HDRS} 
      vtkImageTsallisMutualInformation.h
      vtkImageSquaredDifference.h
      vtkImageShannonMutualInformation.h
      vtkImageResliceBSpline.h
      vtkImageRenyiMutualInformation.h
      vtkImageProbeSum.h
      vtkImageNormalizedMutualInformation.h
      vtkImageGridStatistics.h
      vtkImageDataStatistics.h
      vtkImageCrossCorrelation.h
      vtkImageAmoebaGrid.h
      vtkGridTransformBSpline.h
      vtkCalcCrossCorrelation.h
      vtkCalcCentroid.h
      vtkImageMean3D.h
    )
  endif()
endif()

# -----------------------------------------------------------------
# Build the library
include_directories(${${PROJECT_NAME}_INCLUDE_DIRS})
vtk_module_library(${PROJECT_NAME} ${Module_SRCS})

# -----------------------------------------------------------------
# Build the python wrapping if enabled
IF( RobartsVTK_WRAP_PYTHON AND VTK_WRAP_PYTHON )
  vtk_module_load(${PROJECT_NAME})
  vtk_module_headers_load(${PROJECT_NAME})
  vtk_include_recurse(${PROJECT_NAME})
  vtk_add_python_wrapping(${PROJECT_NAME})
ENDIF()

LIST(APPEND VTK_MODULES_ENABLED ${PROJECT_NAME})
SET(VTK_MODULES_ENABLED ${VTK_MODULES_ENABLED} PARENT_SCOPE)