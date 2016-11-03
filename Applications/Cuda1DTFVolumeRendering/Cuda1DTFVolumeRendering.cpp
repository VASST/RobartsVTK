/*==========================================================================

  Copyright (c) 2016 Uditha L. Jayarathne, ujayarat@robarts.ca

  Use, modification and redistribution of the software, in source or
  binary forms, are permitted provided that the following terms and
  conditions are met:

  1) Redistribution of the source code, in verbatim or modified
  form, must retain the above copyright notice, this license,
  the following disclaimer, and any notices that refer to this
  license and/or the following disclaimer.  

  2) Redistribution in binary form must include the above copyright
  notice, a copy of this license and the following disclaimer
  in the documentation or with other materials provided with the
  distribution.

  3) Modified copies of the source code must be clearly marked as such,
  and must not be misrepresented as verbatim copies of the source code.

  THE COPYRIGHT HOLDERS AND/OR OTHER PARTIES PROVIDE THE SOFTWARE "AS IS"
  WITHOUT EXPRESSED OR IMPLIED WARRANTY INCLUDING, BUT NOT LIMITED TO,
  THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
  PURPOSE.  IN NO EVENT SHALL ANY COPYRIGHT HOLDER OR OTHER PARTY WHO MAY
  MODIFY AND/OR REDISTRIBUTE THE SOFTWARE UNDER THE TERMS OF THIS LICENSE
  BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL OR CONSEQUENTIAL DAMAGES
  (INCLUDING, BUT NOT LIMITED TO, LOSS OF DATA OR DATA BECOMING INACCURATE
  OR LOSS OF PROFIT OR BUSINESS INTERRUPTION) ARISING IN ANY WAY OUT OF
  THE USE OR INABILITY TO USE THE SOFTWARE, EVEN IF ADVISED OF THE
  POSSIBILITY OF SUCH DAMAGES.
  =========================================================================*/

#include <iostream>
#include <string>
// For timing
#include <Windows.h>
#include <stdint.h>

#include <vtkSmartPointer.h>
#include <vtkMetaImageReader.h>
#include <vtkVolumeProperty.h>
#include <vtkColorTransferFunction.h>
#include <vtkPiecewiseFunction.h> 
#include <vtkVolume.h>
#include <vtkRenderer.h> 
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderWindow.h>

// Robarts include
#include <vtkCuda1DVolumeMapper.h>


int main(){

	// Read the volume
	vtkSmartPointer< vtkMetaImageReader > metaReader = vtkSmartPointer< vtkMetaImageReader >::New();
	metaReader->SetFileName("3DUS-output.mhd");
	metaReader->Update();

	// Set Cuda mapper
	vtkSmartPointer< vtkCuda1DVolumeMapper > cudaMapper = vtkSmartPointer< vtkCuda1DVolumeMapper >::New();
	cudaMapper->UseFullVTKCompatibility();
	cudaMapper->SetBlendModeToComposite();
	cudaMapper->SetInputData(metaReader->GetOutput());

	vtkSmartPointer< vtkVolumeProperty > volumeProperty = vtkSmartPointer< vtkVolumeProperty >::New();
	volumeProperty->ShadeOff();
	volumeProperty->SetInterpolationType(VTK_LINEAR_INTERPOLATION);

	vtkSmartPointer< vtkPiecewiseFunction > compositeOpacity = vtkSmartPointer< vtkPiecewiseFunction >::New();
	compositeOpacity->AddPoint(0.0, 0.0);
	compositeOpacity->AddPoint(75.72, 0.079);
	compositeOpacity->AddPoint(176.15, 0.98);
	compositeOpacity->AddPoint(255.0, 1.0);
	volumeProperty->SetScalarOpacity(compositeOpacity); // composite first.

	vtkSmartPointer< vtkColorTransferFunction > colorTransferFun = vtkSmartPointer< vtkColorTransferFunction >::New();
	colorTransferFun->AddRGBPoint(0.0, 0.0, 0.0, 1.0);
	colorTransferFun->AddRGBPoint(40.0, 0.0, 0.1, 0.0);
	colorTransferFun->AddRGBPoint(255.0, 1.0, 0.0, 0.0);
	volumeProperty->SetColor(colorTransferFun);
	
	vtkSmartPointer< vtkVolume > usVolume = vtkSmartPointer< vtkVolume >::New();
	usVolume->SetMapper(cudaMapper);
	usVolume->SetProperty(volumeProperty);

	// Set up renderers
	vtkSmartPointer< vtkRenderer > ren = vtkSmartPointer< vtkRenderer >::New();
	ren->AddViewProp(usVolume);	

	vtkSmartPointer< vtkRenderWindow > renwin = vtkSmartPointer< vtkRenderWindow >::New();
	renwin->AddRenderer(ren);

	vtkSmartPointer< vtkRenderWindowInteractor > iren = vtkSmartPointer< vtkRenderWindowInteractor >::New();
	iren->SetRenderWindow(renwin);
	
	renwin->Render();
	iren->Start();

	return 0;
}
