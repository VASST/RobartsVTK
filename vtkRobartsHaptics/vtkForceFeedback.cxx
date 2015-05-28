#include <iostream>
#include <time.h>

#include "vtkForceFeedback.h"

 /*! \ingroup XarTraXRaster
   Will do a raster scan of the giving area inside -XLength/2 to XLength/2 and -YLength/2 to YLength/2.
   the smaller the increment the higher the resolution will be but the slower it will be.  The points will
   be dumbed 

  \param ndi          a pointer to a ndicapi structure
  \param OutFileName      file name of where the raster scan should be stored.
  \param XLength        Total number degrees at which it should scan in the X direction (center is always 0,0)
  \param YLength        Total number degrees at which it should scan in the Y direction (center is always 0,0)
  \param Increment        degrees in which it should increment at each step
  \param minRange        minimum value that should be stored.  anything smaller is assumed to be distortion and ignored.
  \param maxRange        maximum value that should be stored.  anything greater is assumed to be distortion and ignored.

  \return integer value representing error that occured or OK which finished
*/
vtkForceFeedback* vtkForceFeedback::New()
{
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkForceFeedback");
  if(ret)
    {
    return (vtkForceFeedback*)ret;
    }
  return new vtkForceFeedback;
}


void vtkForceFeedback::PrintSelf(ostream& os, vtkIndent indent)
{
    vtkObject::PrintSelf(os,indent);

}


vtkForceFeedback::vtkForceFeedback()
{
}

vtkForceFeedback::~vtkForceFeedback()
{
}

int vtkForceFeedback::GenerateForce(vtkMatrix4x4 * hapticPosition, double force[3]){
  return 0;
}
