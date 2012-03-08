
#include <stdio.h>
#include <stdlib.h>
#include <direct.h>

#include "vtkFastRBFMesher.h"
#include "vtkPoints.h"
#include "vtkPolyData.h"
#include "vtkOBJReader.h"
#include "vtkObjectFactory.h"

vtkStandardNewMacro(vtkFastRBFMesher);

vtkFastRBFMesher::vtkFastRBFMesher() : vtkObject()
{
    mpReader = vtkOBJReader::New();
    mOBJFilename = NULL;

    int result;
    matlab = engOpenSingleUse(NULL, NULL, &result);
    if(!matlab)
    {
      printf("Could not open matlab\n"); fflush(stdout);
      return;
    }

    engSetVisible(matlab, 0);
}

vtkFastRBFMesher::~vtkFastRBFMesher()
{
    engClose(matlab);
}
    
vtkPolyData* vtkFastRBFMesher::GenerateMesh(vtkPoints *pts)
{
    char buf[1024];

    // If the points aren't visible, skip this
    if(!pts)
    {
        return NULL;
    }

    // Write the points out to a temporary file
    FILE *fp = fopen("TOBEDELETED.TXT", "w");
    int i, N = pts->GetNumberOfPoints();
    for(i = 0; i < N; i++)
    {
        double pt[3];
        pts->GetPoint(i, pt);
        fprintf(fp, "%f %f %f\n", pt[0], pt[1], pt[2]);
    }
    fclose(fp);

    // Get and set the working directory in matlab
    char dir[256];
    _getcwd( dir, 256 );
    sprintf(buf, "cd %s\\", dir);
    engEvalString(matlab, buf);

    engEvalString(matlab, "addpath(\'C:\\Program Files\\FarField Technology\\FastRBF v1.4\\\');");
    engEvalString(matlab, "addpath(\'C:\\Program Files\\FarField Technology\\FastRBF v1.4\\Matlab\\\');");
    engEvalString(matlab, "addpath(\'C:\\Program Files\\FarField Technology\\FastRBF v1.4\\Matlab\\Toolbox\\\');");
    engEvalString(matlab, "addpath(\'C:\\Program Files\\FarField Technology\\FastRBF v1.4\\Matlab\\Tutorial\\\');");
    engEvalString(matlab, "Mesh = fastrbf_import(\'TOBEDELETED.TXT\', \'format\', \'%x %y %z\')");
    engEvalString(matlab, "MeshWithNormals = fastrbf_normalsfrompoints(Mesh);");
    engEvalString(matlab, "Density = fastrbf_densityfromnormals(MeshWithNormals, 0.5, 5.0);");
    engEvalString(matlab, "Density = fastrbf_unique(Density);");
    engEvalString(matlab, "rbf = fastrbf_fit(Density, 0.5, \'reduce\');");
    engEvalString(matlab, "NewMesh = fastrbf_isosurf(rbf, 1);");
    engEvalString(matlab, "NewMeshWithNormals = fastrbf_normalsfrommesh(NewMesh);");
    if(!mOBJFilename)
    {
      engEvalString(matlab, "fastrbf_export(NewMeshWithNormals, \'TOBEDELETED.OBJ\', \'obj\');");
    }
    else
    {
      sprintf(buf, "fastrbf_export(NewMeshWithNormals, \'%s\', \'obj\');", mOBJFilename);
      engEvalString(matlab, buf);
    }

    // Open the OBJ file
    if(!mOBJFilename)
    {
      mpReader->SetFileName("TOBEDELETED.OBJ");
    }
    else
    {
      mpReader->SetFileName(mOBJFilename);
    }
    mpReader->Update();

    // Delete the files, but don't delete the OBJ file if it was specified by the user
    remove("TOBEDELETED.TXT");
    if(!mOBJFilename)
    {
      remove("TOBEDELETED.OBJ");
    }

    return mpReader->GetOutput();
}

