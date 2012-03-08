#ifndef _VTKDDC_
#define _VTKDDC_

#include <vector>
#include <string.h>
#include <engine.h>

#include "vtkObject.h"

class vtkPoints;
class vtkPolyData;
class vtkOBJReader;

class VTK_EXPORT vtkFastRBFMesher : public vtkObject
{
public:
    static vtkFastRBFMesher *New();
    vtkTypeMacro(vtkFastRBFMesher,vtkObject);

    vtkFastRBFMesher();
    virtual ~vtkFastRBFMesher();

    vtkPolyData* GenerateMesh(vtkPoints *pts);

    void SetOBJFilename(char *name)
    {
       mOBJFilename = new char[strlen(name)+1];
       strcpy(mOBJFilename, name);
       mOBJFilename[strlen(name)] = '\0';
    }

protected:
//BTX
    char *mOBJFilename;
    vtkOBJReader *mpReader;
    Engine *matlab;
//ETX
};

#endif

