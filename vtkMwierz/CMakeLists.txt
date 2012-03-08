PROJECT(Mwierz)

SET(KIT Mwierz)
SET(UKIT Mwierz)
SET(KIT_LIBS vtkHybrid vtkImaging)
SET(KIT_EXTRA_LIBS)

SET ( Kit_SRCS
  #vtkTriangleMeshQuality.cxx
  vtkTPSRegistration.cxx
  vtkShapeBasedInterpolation.cxx
  vtkPrincipalComponentAnalysis.cxx
  vtkPowellMinimizer.cxx
  vtkPolyDataSurfaceArea.cxx
  vtkPolyDataNormals2.cxx
  vtkPolyDataCorrespondence.cxx
  vtkMeshSmootheness.cxx
  vtkImageTsallisMutualInformation.cxx
  vtkImageTMIManipulator.cxx
  vtkImageSquaredDifference.cxx
  vtkImageSMIPVIManipulator.cxx
  vtkImageSMIManipulator.cxx
  vtkImageSMIManipulator2.cxx
  vtkImageShannonMutualInformation.cxx
  vtkImageSDManipulator.cxx
  vtkImageRMIManipulator.cxx
  vtkImageResliceBSpline.cxx
  vtkImageRenyiMutualInformation.cxx
  vtkImageProbeSum.cxx
  vtkImagePatternIntensity.cxx
  vtkImageNormalizedMutualInformation.cxx
  vtkImageNormalizedCrossCorrelation.cxx
  vtkImageNMIManipulator.cxx
  vtkImageNCCManipulator.cxx
  vtkImageMIManipulator.cxx
##  vtkImageMean3D.cxx
  vtkImageGridStatistics.cxx
  vtkImageECRManipulator.cxx
  vtkImageDataStatistics.cxx
  vtkImageCrossCorrelation.cxx
  vtkImageAmoebaGrid.cxx
  vtkImageADManipulator.cxx
  vtkImageAbsoluteDifference.cxx
  vtkImage3DNoiseSource.cxx
  vtkGridTransformBSpline.cxx
  vtkGBEFixed.cxx
  vtkGBE.cxx
  vtkFunctionMinimizer.cxx
  vtkDeformedVolume.cxx
  vtkCompactSupportRBFTransform.cxx
  vtkCISGSubdivideGridTransform.cxx
  vtkCalcCrossCorrelation.cxx
  vtkCalcCentroid.cxx
)


SET(Kit_EXTRA_SRCS)
SET(Kit_EXTRA_CMDS)
SET(Kit_TCL_EXTRA_SRCS)
SET(Kit_PYTHON_EXTRA_SRCS)
SET(Kit_JAVA_EXTRA_SRCS)

SET (KIT_EXTRA_LIBS ${KIT_EXTRA_LIBS} vtkImaging)
SET (KIT_EXTRA_LIBS ${KIT_EXTRA_LIBS} vtkImagingPython)

# -----------------------------------------------------------------

# Include the common setting file
INCLUDE(${RobartsVTK_SOURCE_DIR}/CMake/KitCommonBlock.cmake)

