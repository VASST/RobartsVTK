/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkFusionRenderer.cxx,v $
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
#include "vtkFusionRenderer.h"

#include "vtkCuller.h"
#include "vtkObjectFactory.h"
#include "vtkRenderWindow.h"
#include "vtkEndoscope.h"
#include "vtkVideoSource.h"
#include "vtkTrackerTool.h"
#include "vtkTrackerBuffer.h"
#include "vtkPNGReader.h"
#include "vtkPNGWriter.h"
#include "vtkMatrix4x4.h"
#include "vtkTransform.h"
#include "vtkImageData.h"
#include "vtkCellArray.h"
#include "vtkPointData.h"
#include "vtkPolyData.h"

#ifndef VTK_IMPLEMENT_MESA_CXX
#if defined(__APPLE__) && (defined(VTK_USE_CARBON) || defined(VTK_USE_COCOA))
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#endif
#endif

#include "OpenCV/otherlibs/highgui/highgui.h"

#include <math.h>

// Screen dimensions (these should be variables, I'd say)
#define WIDTH 400
#define HEIGHT 400

////////////////////////// NOTE!
// The opengl contexts will swap and there's a good chance you will lose your textures


#ifndef VTK_IMPLEMENT_MESA_CXX
vtkCxxRevisionMacro(vtkFusionRenderer, "$Revision: 1.3 $");
vtkStandardNewMacro(vtkFusionRenderer);
#endif

vtkCxxSetObjectMacro(vtkFusionRenderer,Endoscope,vtkEndoscope);
vtkCxxSetObjectMacro(vtkFusionRenderer,US,vtkVideoSource);
vtkCxxSetObjectMacro(vtkFusionRenderer,USTool,vtkTrackerTool);
vtkCxxSetObjectMacro(vtkFusionRenderer,MaskReader,vtkPNGReader);
vtkCxxSetObjectMacro(vtkFusionRenderer,DebugTool,vtkTrackerTool);

vtkFusionRenderer::vtkFusionRenderer()
{
  Endoscope = NULL;
  US = NULL;
  USTool = NULL;
  DebugTool = NULL;
  MaskReader = NULL;
  bInitialized = false;
  mCenter[0] = mCenter[1] = 0.0;        // TODO: Grab this from the endoscope's ImageParams
  mScale[0] = mScale[1] = 1.0;          // TODO: Grab this from the endoscope's ImageParams (and use it)
  TrackingToOpenGL = vtkMatrix4x4::New();
  OpenCVtoOpenGL = vtkMatrix4x4::New();
  OpenCVtoOpenGL->Identity();
  OpenCVtoOpenGL->SetElement(1, 1, -1.f);
  OpenCVtoOpenGL->SetElement(2, 2, -1.f);
  mPts = new int[6];
  mImgNo = 0;
  mbSave = false;
  Translucency = 1.0f;
}

vtkFusionRenderer::~vtkFusionRenderer()
{
  delete[] mPts;
  TrackingToOpenGL->Delete();
  OpenCVtoOpenGL->Delete();
}

void vtkFusionRenderer::SaveImages(void)
{
  mbSave = true;
}

void vtkFusionRenderer::SetTranslucency(double t)
{
  Translucency = t;
}

void vtkFusionRenderer::Initialize(void)
{
    this->RenderWindow->MakeCurrent();
	  GLenum err = glewInit();
	  if (GLEW_OK != err)
	  {
	    vtkErrorMacro(<< "GLEW failed to initialize");
	  }
	  if (!GLEW_VERSION_2_0)
	  {
	    vtkErrorMacro(<< "OpenGL 2.0 not supported");
	  }
	  if (!GLEW_ARB_shading_language_100)
	  {
	    vtkErrorMacro(<< "GLSL not supported");
	  }

    // Load the shaders
    undistortShader = new GLSLShader;
    if(!undistortShader->LoadVertexShader("E:/shaders/simpleVS.vs"))
    {
      printf(undistortShader->GetError().c_str());
      exit(-1);
    }
    if(!undistortShader->LoadFragmentShader("E:/shaders/undistortFS.fs"))
    {
      printf(undistortShader->GetError().c_str());
      exit(-1);
    }
    if(!undistortShader->LinkProgram())
    {
      printf(undistortShader->GetError().c_str());
      exit(-1);
    }

    maskShader = new GLSLShader;
    if(!maskShader->LoadVertexShader("E:/shaders/simpleVS.vs"))
    {
      printf(maskShader->GetError().c_str());
      exit(-1);
    }
    if(!maskShader->LoadFragmentShader("E:/shaders/maskFS.fs"))
    {
      printf(maskShader->GetError().c_str());
      exit(-1);
    }
    if(!maskShader->LinkProgram())
    {
      printf(maskShader->GetError().c_str());
      exit(-1);
    }

    surfShader = new GLSLShader;
    if(!surfShader->LoadVertexShader("E:/shaders/shadeSurfaceVS.vs"))
    {
      printf(surfShader->GetError().c_str());
      exit(-1);
    }
    if(!surfShader->LoadFragmentShader("E:/shaders/shadeSurfaceFS.fs"))
    {
      printf(surfShader->GetError().c_str());
      exit(-1);
    }
    if(!surfShader->LinkProgram())
    {
      printf(surfShader->GetError().c_str());
      exit(-1);
    }

    // Load the textures
    LoadTextures();

    LoadUltrasoundCalibration();

    // Initialize camera stuff
    if(Endoscope)
    {
      Endoscope->GetImageParams(mScale[0], mScale[1], mCenter[0], mCenter[1]);
    }

    bInitialized = true;
}


void vtkFusionRenderer::LoadUltrasoundCalibration(void)
{
    double ignoreZ;

    UScalib.bFlipX = true;  // Apparently I need this
    UScalib.bFlipY = false;

    // Load the pixel spacing information
    US->GetDataSpacing(UScalib.SpacX, UScalib.SpacY, ignoreZ);

    // Load the pixel origin information
    US->GetDataOrigin(UScalib.OrigX, UScalib.OrigY, ignoreZ);
}

void vtkFusionRenderer::LoadTextures(void)
{
  if(!Endoscope) return;
  Endoscope->Update();
  switch(Endoscope->GetOutputFormat())
  {
  case VTK_LUMINANCE:
    EndoscopeFormat = GL_LUMINANCE;
    break;
  case VTK_RGBA:
    EndoscopeFormat = GL_RGBA;
    break;
  case VTK_RGB:
  default:
    EndoscopeFormat = GL_RGB;
    break;
  }
  if(!US) return;
  US->Update();
  switch(US->GetOutputFormat())
  {
  case VTK_LUMINANCE:
    USFormat = GL_LUMINANCE;
    break;
  case VTK_RGBA:
    USFormat = GL_RGBA;
    break;
  case VTK_RGB:
  default:
    USFormat = GL_RGB;
    break;
  }
  if(!USTool) return;
  if(!MaskReader) return;

  vtkMatrix4x4 *endoscopeTransform = vtkMatrix4x4::New();
  void *ptr;
  Endoscope->GetRawImageAndTransform(&ptr,endoscopeTransform);
  endoscopeTransform->Delete();

  glGenTextures(1, &mFrame);
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_RECTANGLE_ARB, mFrame);
  glEnable(GL_TEXTURE_RECTANGLE_ARB);
  glTexImage2D(GL_TEXTURE_RECTANGLE_ARB,
	 0, GL_RGB, 
	 WIDTH, HEIGHT, 0, EndoscopeFormat, 
	 GL_UNSIGNED_BYTE, ptr);

  glGenTextures(1, &mUSFrame);
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_RECTANGLE_ARB, mUSFrame);
  glEnable(GL_TEXTURE_RECTANGLE_ARB);
  glTexImage2D(GL_TEXTURE_RECTANGLE_ARB,
  	 0, GL_RGB, 
     WIDTH, HEIGHT, 0, USFormat, 
  	 GL_UNSIGNED_BYTE, US->GetOutput()->GetScalarPointer());

  Endoscope->CreateUndistortMap(WIDTH, HEIGHT, mUndistortImg);
  glActiveTexture(GL_TEXTURE1);
  glEnable(GL_TEXTURE_RECTANGLE_ARB);
  glGenTextures(1, &mGLUndistortTex);
  glBindTexture(GL_TEXTURE_RECTANGLE_ARB, mGLUndistortTex);
  glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_S, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_T, GL_CLAMP);
  glTexImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, GL_RGBA_FLOAT16_ATI, WIDTH, HEIGHT, 0,
                GL_RGBA, GL_FLOAT, mUndistortImg); //GL_FLOAT_RGBA16_NV

  glGenTextures(1, &mMaskID);
  glActiveTexture(GL_TEXTURE2);
  glEnable(GL_TEXTURE_RECTANGLE_ARB);
  glBindTexture(GL_TEXTURE_RECTANGLE_ARB, mMaskID);
  glTexImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, GL_RGB, WIDTH, HEIGHT, 0, 
                GL_RGB, GL_UNSIGNED_BYTE, MaskReader->GetOutput()->GetScalarPointer());
}

// Concrete open gl render method.
void vtkFusionRenderer::DeviceRender(void)
{
  int i;
  static bFirstTime = true;
  static vtkMatrix4x4 *mat = vtkMatrix4x4::New();
  static vtkMatrix4x4 *endoscopeTransform = vtkMatrix4x4::New();
  void *ptr;

  // Do not remove this MakeCurrent! Due to Start / End methods on
  // some objects which get executed during a pipeline update, 
  // other windows might get rendered since the last time
  // a MakeCurrent was called.
  this->RenderWindow->MakeCurrent();

  // Grab the viewport and set it
  int *orig, *size;
  orig = this->GetOrigin();
  size = this->GetSize();
  int diff = size[0] - (4 * size[1] / 3);
  if(diff > 0)
  {
    // Clear the whole buffer first
    glViewport(orig[0], orig[1], size[0], size[1]);
    Clear();

    // Now set the focussed buffer
    orig[0] = orig[0] + (diff / 2);
    size[0] = 4 * size[1] / 3;
  }
  else if(diff < 0)
  {
    // Clear the whole buffer first
    glViewport(orig[0], orig[1], size[0], size[1]);
    Clear();

    // Now set the focussed buffer
    orig[1] = orig[1] - (diff / 2);
    size[1] = 3 * size[0] / 4;
  }

  glViewport(orig[0], orig[1], size[0], size[1]);

  // Set the perspective transform
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(2.0 * atan(0.5 / this->Aspect[0]) * 180.0 / 3.141592, 
                 this->Aspect[0], 0.9, 2000.0);

  // Enable some features...
  glDisable( GL_CULL_FACE );
  glDisable( GL_DEPTH_TEST );
  glDepthMask( GL_FALSE );
  glDisable(GL_LIGHTING);
  glDisable(GL_LIGHT0);
  glEnable( GL_TEXTURE_RECTANGLE_ARB );
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  // set matrix mode for actors 
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  Clear();

  // If the endoscope feed is present, display it
  if(bInitialized)
  {
    if(Endoscope)
    {
      Endoscope->Update();
	  
      undistortShader->Use();
      int SamplerIdx = glGetUniformLocation(undistortShader->GetProgram(), "baseTex");
      glUniform1i(SamplerIdx, 0);
      SamplerIdx = glGetUniformLocation(undistortShader->GetProgram(), "undistortTex");
      glUniform1i(SamplerIdx, 1);

      // Update the textures
      Endoscope->GetRawImageAndTransform(&ptr, mat);
	    endoscopeTransform->DeepCopy(mat);
      endoscopeTransform->Invert(); // Invert the matrix to get us a transform into endoscope space

      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_RECTANGLE_ARB, mFrame);
      glEnable(GL_TEXTURE_RECTANGLE_ARB);
	    glTexSubImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, 0, 0, WIDTH, HEIGHT, EndoscopeFormat, 
                      GL_UNSIGNED_BYTE, ptr);

      glActiveTexture(GL_TEXTURE1);
      glEnable(GL_TEXTURE_RECTANGLE_ARB);
      glBindTexture(GL_TEXTURE_RECTANGLE_ARB, mGLUndistortTex);

      glColor3f(1.0, 1.0, 1.0);
      glBegin(GL_QUADS);
        glMultiTexCoord2f(GL_TEXTURE0, 0, 0);
        glVertex3f(-0.5, 0.5/this->Aspect[0], -1.0);
        glMultiTexCoord2f(GL_TEXTURE0, WIDTH, 0);
        glVertex3f(0.5, 0.5/this->Aspect[0], -1.0);
        glMultiTexCoord2f(GL_TEXTURE0, WIDTH, HEIGHT);
        glVertex3f(0.5, -0.5/this->Aspect[0], -1.0);
        glMultiTexCoord2f(GL_TEXTURE0, 0, HEIGHT);
        glVertex3f(-0.5, -0.5/this->Aspect[0], -1.0);
      glEnd();
    
	    // Our calibrated endoscope goes from tracking space into the space defined by OpenCV
	    // We need to transform into OpenGL's camera space too
      vtkMatrix4x4::Multiply4x4(OpenCVtoOpenGL, endoscopeTransform, TrackingToOpenGL);
	  }

    // Save the Endoscope-only image and the tracking data
    if(mbSave)
    {
	    char filename[80];

      glFinish();

      // Create the images
      SourceImage = cvCreateImage(cvSize(size[0], size[1]), IPL_DEPTH_8U, 3);
      DestImage = cvCreateImage(cvSize(WIDTH, HEIGHT), IPL_DEPTH_8U, 3);
      DestImageGray = cvCreateImage(cvSize(WIDTH, HEIGHT), IPL_DEPTH_8U, 1);

	    // Save image data
      unsigned char *imgptr;
      cvGetRawData(SourceImage, &imgptr);
	    glReadPixels(orig[0], orig[1], size[0], size[1], GL_RGB, GL_UNSIGNED_BYTE, imgptr);
	    sprintf(filename, "Endoscope%d.png", mImgNo);
      cvResize(SourceImage, DestImage, CV_INTER_AREA);
      cvSaveImage(filename, DestImage);

      // Copy and save the ultrasound image
      if(US->GetOutput()->GetNumberOfScalarComponents() == 1)
      {
        cvGetRawData(DestImageGray, &imgptr);
			  memcpy(imgptr, US->GetOutput()->GetScalarPointer(), WIDTH*HEIGHT);
			  sprintf(filename, "Ultrasound%d.png", mImgNo);
        cvSaveImage(filename, DestImageGray);
      }
      else
      {
        cvGetRawData(DestImage, &imgptr);
			  memcpy(imgptr, US->GetOutput()->GetScalarPointer(), WIDTH*HEIGHT*3);
			  sprintf(filename, "Ultrasound%d.png", mImgNo);
        cvSaveImage(filename, DestImage);
      }

      // Save tracking data
	    sprintf(filename, "WorldToCamera%d.matrix", mImgNo);
	    FILE *fpMat = fopen(filename, "w");
	    if(fpMat)
	    {
		    for(size_t u = 0; u < 4; ++u)
		    {
          for(size_t v = 0; v < 4; ++v)
          {
		        fprintf(fpMat, "%f ", endoscopeTransform->GetElement(u, v));
		      }
		      fprintf(fpMat, "\n");
	      }
	      fclose(fpMat);
	    }
    }

    /////////////////////////////
    // Draw the virtual objects
    glEnable( GL_DEPTH_TEST );
    glDepthMask( GL_TRUE );
    glEnable( GL_SCISSOR_TEST );
    glScissor(orig[0], orig[1], size[0], size[1]);

    // recenter the viewport
    {
      double xScale = double(size[0]) / WIDTH;
      double yScale = double(size[1]) / HEIGHT;

      // Adjust the viewport so that the images are centered correctly
      glViewport( -mCenter[0] * xScale + orig[0], 
                  -mCenter[1] * yScale + orig[1], 
                  size[0], size[1] );
    }


    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glLoadTransposeMatrixd(&TrackingToOpenGL->Element[0][0]);	// Transpose when loading
																// because opengl stores matrices
																// in column major order

    // Draw the ultrasound fan
    if(US && USTool && MaskReader)
    {
      US->Update();
      USTool->GetTracker()->Update();

      if( !Endoscope->GetTrackerTool()->IsOutOfView() &&
          !USTool->IsOutOfView() )
      {
        maskShader->Use();
        int SamplerIdx = glGetUniformLocation(maskShader->GetProgram(), "baseTex");
        glUniform1i(SamplerIdx, 0);
        SamplerIdx = glGetUniformLocation(maskShader->GetProgram(), "maskTex");
        glUniform1i(SamplerIdx, 2);

        int VarIdx = glGetUniformLocation(maskShader->GetProgram(), "translucency");
        glUniform1f(VarIdx, Translucency);

        // Load the ultrasound image into a texture
        glActiveTexture(GL_TEXTURE0);
        glEnable(GL_TEXTURE_RECTANGLE_ARB);
        glBindTexture(GL_TEXTURE_RECTANGLE_ARB, mUSFrame);
        glTexSubImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, 0, 0, WIDTH, HEIGHT, USFormat, 
                        GL_UNSIGNED_BYTE, US->GetOutput()->GetScalarPointer());

        glActiveTexture(GL_TEXTURE2);
        glEnable(GL_TEXTURE_RECTANGLE_ARB);
        glBindTexture(GL_TEXTURE_RECTANGLE_ARB, mMaskID);

        glEnable(GL_BLEND);

        // image endpoints in pixel coordinates
        float USImgCoords[4][2] = {{0,0}, {WIDTH,0}, {WIDTH,HEIGHT}, {0,HEIGHT}};
		
		    // Draw the ultrasoudn plane
		    glBegin(GL_QUADS);
        for(int i = 0; i < 4; i++)
        {
          static vtkMatrix4x4 *USTransform = vtkMatrix4x4::New();

          // Transform from US image space to tracker space
          // Interpolate transform to the endoscope's time frame to make sure the US
          // location is in sync with the endoscopic view
          float usX, usY;
          if(UScalib.bFlipX)
          {
            usX = -(UScalib.OrigX + USImgCoords[i][0] * UScalib.SpacX);
            if((i == 1) || (i == 2)) USImgCoords[i][0] = WIDTH;
            else USImgCoords[i][0] = 0;
          }
          else
          {
            usX = UScalib.OrigX + USImgCoords[i][0] * UScalib.SpacX;
            if((i == 1) || (i == 2)) USImgCoords[i][0] = 0;
            else USImgCoords[i][0] = WIDTH;
          }
          if(UScalib.bFlipY)
          {
            usY = -(UScalib.OrigY + USImgCoords[i][1] * UScalib.SpacY);
            if(i&2) USImgCoords[i][1] = 0;
            else USImgCoords[i][1] = HEIGHT;
          }
          else
          {
            usY = UScalib.OrigY + USImgCoords[i][1] * UScalib.SpacY;
            if(i&2) USImgCoords[i][1] = HEIGHT;
            else USImgCoords[i][1] = 0;
          }
          double USSpace[4] = {usX, usY, 0, 1}, trackingSpace[4];
          USTool->GetBuffer()->Lock();
          USTool->GetBuffer()->GetFlagsAndMatrixFromTime(USTransform, Endoscope->GetFrameTimeStamp());
          USTool->GetBuffer()->Unlock();
          USTransform->MultiplyPoint(USSpace, trackingSpace);

          // Output the vertex and the texture coordinates
          glMultiTexCoord2fv(GL_TEXTURE0, USImgCoords[i]);
          glVertex3dv(trackingSpace);
        }
        glEnd();

        glDisable(GL_BLEND);

        // Save the US and Fused images
        if(mbSave)
        {
		      char filename[80];

          glFinish();

			    // Save the fused image
          unsigned char *imgptr;
          cvGetRawData(SourceImage, &imgptr);
          glReadPixels(orig[0], orig[1], size[0], size[1], GL_RGB, GL_UNSIGNED_BYTE, imgptr);
			    sprintf(filename, "Fused%d.png", mImgNo);
          cvResize(SourceImage, DestImage, CV_INTER_AREA);
          cvSaveImage(filename, DestImage);
        }
      }
      else
      {
        //if( Endoscope->GetTrackerTool()->IsOutOfView() ) printf("Can't see the endoscope\n");
        //if( USTool->IsOutOfView() ) printf("Can't see the ultrasound\n");
        //fflush(stdout);
      }
    }

    // Turn off all the textures	
    glActiveTexture(GL_TEXTURE0);
    glDisable(GL_TEXTURE_RECTANGLE_ARB);
    glBindTexture(GL_TEXTURE_RECTANGLE_ARB, 0);
    glActiveTexture(GL_TEXTURE1);
    glDisable(GL_TEXTURE_RECTANGLE_ARB);
    glBindTexture(GL_TEXTURE_RECTANGLE_ARB, 0);
    glActiveTexture(GL_TEXTURE2);
    glDisable(GL_TEXTURE_RECTANGLE_ARB);
    glBindTexture(GL_TEXTURE_RECTANGLE_ARB, 0);

    surfShader->Use();
    glColor4f(1,0,0,1);

	  for(i = 0; i < mMeshes.size(); i++)
    {
		  int npts;
		  vtkCellArray *polys = mMeshes[i]->GetPolys();
		  vtkDataArray *normals = mMeshes[i]->GetPointData()->GetNormals();
		  vtkPoints *vectors = mMeshes[i]->GetPoints();

		  // Load the mesh
		  polys->InitTraversal();
		  while(polys->GetNextCell(npts, mPts))
		  {
			  glBegin(GL_POLYGON);
			  for(int j = 0; j < npts; j++)
			  {
				  if(normals)
				  {
				    glNormal3dv(normals->GetTuple3(mPts[j]));
				  }
				  glVertex3dv(vectors->GetPoint(mPts[j]));
			  }
			  glEnd();
		  }
    }

    GLSLShader::Release();

    if(DebugTool)
	  {
		  float pos[4];
		  DebugTool->GetTransform()->GetPosition(pos); pos[3] = 1.f;
		  /*printf("Camera Space Position: %f %f %f\n", 
			  TrackingToOpenGL->MultiplyFloatPoint(pos)[0],
			  TrackingToOpenGL->MultiplyFloatPoint(pos)[1],
			  TrackingToOpenGL->MultiplyFloatPoint(pos)[2]);*/

		  glPointSize(5.0);
		  glColor4f(0,1,0,1);
		  glBegin(GL_POINTS);
			  glVertex3fv(pos);
		  glEnd();
	  }

    glLoadIdentity();
  }

  glDisable( GL_SCISSOR_TEST );

  // Turn off save mode and increment the save number
  if(mbSave)
  {
    cvReleaseImage(&SourceImage);
    cvReleaseImage(&DestImage);
    cvReleaseImage(&DestImageGray);
    mbSave = false;
    ++mImgNo;
  }
}


void vtkFusionRenderer::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);
}


void vtkFusionRenderer::Clear(void)
{
  GLbitfield  clear_mask = 0;

  if (! this->Transparent())
    {
    glClearColor( ((GLclampf)(this->Background[0])),
                  ((GLclampf)(this->Background[1])),
                  ((GLclampf)(this->Background[2])),
                  ((GLclampf)(1.0)) );
    clear_mask |= GL_COLOR_BUFFER_BIT;
    }

  glClearDepth( (GLclampd)( 1.0 ) );
  clear_mask |= GL_DEPTH_BUFFER_BIT;

  vtkDebugMacro(<< "glClear\n");
  glClear(clear_mask);
}

int vtkFusionRenderer::AddPolyData(vtkPolyData *pd)
{
    int curMesh = mMeshes.size();
    mMeshes.push_back(vtkPolyData::New());

    // Load the display list
    AddPolyData(curMesh, pd);

    return curMesh;
}

void vtkFusionRenderer::AddPolyData(int mesh, vtkPolyData *pd)
{
    if(mesh >= mMeshes.size())
    {
        return;
    }

	mMeshes[mesh]->DeepCopy(pd);
}
