/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkFusionRenderer.h,v $
  Language:  C++
  Date:      $Date: 2008/07/17 15:33:39 $
  Version:   $Revision: 1.3 $

  Copyright (c) 1993-2002 Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
// .NAME vtkFusionRenderer - Fusion renderer
// .SECTION Description
// vtkFusionRenderer is a concrete implementation of the abstract class
// vtkRenderer. vtkFusionRenderer merges ultrasound video, or other objects
// with endoscopic video.

#ifndef __vtkFusionRenderer_h
#define __vtkFusionRenderer_h

#include <GL/glew.h>

#include <vector>
#include <cv.h>
#include <cxcore.h>

#include "vtkOpenGLRenderer.h"

#include "GLSLShader.h"
#include <vtkVersion.h> //for VTK_MAJOR_VERSION

class vtkImageData;
class vtkEndoscope;
class vtkVideoSource;
class vtkTrackerTool;
class vtkPNGReader;
class vtkPNGWriter;
class vtkPolyData;

class VTK_EXPORT vtkFusionRenderer : public vtkOpenGLRenderer
{
public:
  static vtkFusionRenderer *New();
#if (VTK_MAJOR_VERSION <= 5)
  vtkTypeRevisionMacro(vtkFusionRenderer,vtkRenderer);
#else
  vtkTypeMacro(vtkFusionRenderer,vtkRenderer);
#endif
  void PrintSelf(ostream& os, vtkIndent indent);

  // Description:
  // Concrete open gl render method.
  void DeviceRender(void);
  void Clear(void);

  virtual void SetEndoscope(vtkEndoscope *);
  vtkGetObjectMacro(Endoscope,vtkEndoscope);
  virtual void SetUS(vtkVideoSource *);
  vtkGetObjectMacro(US,vtkVideoSource);
  virtual void SetUSTool(vtkTrackerTool *);
  vtkGetObjectMacro(USTool,vtkTrackerTool);
  virtual void SetMaskReader(vtkPNGReader *);
  vtkGetObjectMacro(MaskReader,vtkPNGReader);

  virtual void SetDebugTool(vtkTrackerTool *);
  vtkGetObjectMacro(DebugTool,vtkTrackerTool);


  void LoadTextures(void);
  void Initialize(void);
  void LoadUltrasoundCalibration(void);
  int AddPolyData(vtkPolyData *);
  void AddPolyData(int, vtkPolyData *);

  void SaveImages(void);

  void SetTranslucency(double);

protected:
  vtkFusionRenderer();
  ~vtkFusionRenderer();

  vtkEndoscope *Endoscope;
  GLenum EndoscopeFormat;
  vtkVideoSource *US;
  GLenum USFormat;
  vtkTrackerTool *USTool;
  vtkTrackerTool *DebugTool;

  vtkPNGReader *MaskReader;

  IplImage  *SourceImage;
  IplImage  *DestImage;
  IplImage  *DestImageGray;

  float *mUndistortImg;

  bool bInitialized;

//BTX
  vtkMatrix4x4 *OpenCVtoOpenGL;
  vtkMatrix4x4 *TrackingToOpenGL;

  GLuint mFrame, mUSFrame, mGLUndistortTex, mMaskID;

  GLSLShader *undistortShader, *maskShader, *surfShader;

  struct
  {
    int bFlipX, bFlipY;
    double OrigX, OrigY;
    double SpacX, SpacY;
  } UScalib;

  std::vector<vtkPolyData*> mMeshes;
  float mScale[2], mCenter[2];
  int *mPts;

//ETX

  int   mImgNo;
  bool  mbSave;

  double Translucency;

private:
  vtkFusionRenderer(const vtkFusionRenderer&);  // Not implemented.
  void operator=(const vtkFusionRenderer&);  // Not implemented.
};

#endif
