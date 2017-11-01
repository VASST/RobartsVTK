#include "vtkVolume.h"
#include "vtkRenderWindow.h"
#include "vtkRenderWindowInteractor.h"
#include "vtkInteractorStyleTrackballCamera.h"
#include "vtkRenderer.h"
#include "vtkMetaImageReader.h"

#include "vtkVolumeMapper.h"
#include "vtkCudaDRRImageVolumeMapper.h"

#include <vtkAutoInit.h> 
VTK_MODULE_INIT(vtkRenderingVolumeOpenGL2)
VTK_MODULE_INIT(vtkRenderingOpenGL2)

int main(int argc, char** argv) {

	//load image from command line argument
	if (argc != 2)
		return -1;
	char* filename = argv[1];
	vtkMetaImageReader* reader = vtkMetaImageReader::New();
	reader->SetFileName(filename);
	reader->Update();

	//create DRR mapper
	vtkCudaDRRImageVolumeMapper* mapper = vtkCudaDRRImageVolumeMapper::New();
	mapper->SetInputData(reader->GetOutput());
	unsigned char tint[4] = { 255,0,0,0 };
	//mapper->SetTint(tint);
	//mapper->SetImageFlipped(true);
	mapper->Update();

	//create VTK pipeline
	vtkVolume* volume = vtkVolume::New();
	volume->SetMapper(mapper);
	vtkRenderer* renderer = vtkRenderer::New();
	double bk[3] = { 1.0, 1.0, 1.0 };
	renderer->SetBackground(bk);
	vtkRenderWindow* window = vtkRenderWindow::New();
	window->AddRenderer(renderer);
	vtkRenderWindowInteractor* interactor = vtkRenderWindowInteractor::New();
	interactor->SetRenderWindow(window);
	vtkInteractorStyleTrackballCamera* style = vtkInteractorStyleTrackballCamera::New();
	interactor->SetInteractorStyle(style);
	renderer->AddVolume(volume);
	//run event loop
	interactor->Start();

	//cleanup
	style->Delete();
	interactor->Delete();
	window->Delete();
	renderer->Delete();
	volume->Delete();
	mapper->Delete();
	reader->Delete();
}