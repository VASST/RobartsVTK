PROJECT(vtkRobartsRegistration)

include(module.cmake)

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
  vtkCompactSupportRBFTransform.cxx
)

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
    vtkDeformedVolume.cxx # depends on vtkGridTransformBSpline.cxx
    vtkCISGSubdivideGridTransform.cxx # depends on vtkGridTransformBSpline.cxx
    vtkCalcCrossCorrelation.cxx
    vtkCalcCentroid.cxx
    vtkImageMean3D.cxx
  )
endif()

# -----------------------------------------------------------------
# Build the library
vtk_module_library(${PROJECT_NAME} ${Module_SRCS})

# -----------------------------------------------------------------
LIST(APPEND VTK_MODULES_ENABLED ${PROJECT_NAME})
SET(VTK_MODULES_ENABLED ${VTK_MODULES_ENABLED} PARENT_SCOPE)