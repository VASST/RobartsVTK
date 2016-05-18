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

SET(${PROJECT_NAME}_INCLUDE_DIRS ${VTK_INCLUDE_DIRS})

# -----------------------------------------------------------------
# Build the library
vtk_module_library(${PROJECT_NAME} ${Module_SRCS})

# -----------------------------------------------------------------
LIST(APPEND VTK_MODULES_ENABLED ${PROJECT_NAME})
SET(VTK_MODULES_ENABLED ${VTK_MODULES_ENABLED} PARENT_SCOPE)