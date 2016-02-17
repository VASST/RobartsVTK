#ifndef __VTKIMAGEFRANGIFILTER_H__
#define __VTKIMAGEFRANGIFILTER_H__

#include "vtkRobartsCommonModule.h"

#include "vtkSimpleImageToImageFilter.h"
#include "vtkImageData.h"
#include "vtkSetGet.h"

class VTKROBARTSCOMMON_EXPORT vtkImageFrangiFilter : public vtkSimpleImageToImageFilter
{
public:
  vtkTypeMacro( vtkImageFrangiFilter, vtkSimpleImageToImageFilter );
  static vtkImageFrangiFilter *New();

  virtual void SimpleExecute(vtkImageData* input, vtkImageData* output);

  vtkSetMacro(Sheet, double);
  vtkGetMacro(Sheet, double);
  vtkSetMacro(Line, double);
  vtkGetMacro(Line, double);
  vtkSetMacro(Blob, double);
  vtkGetMacro(Blob, double);
  vtkSetMacro(AssymmetrySensitivity, double);
  vtkGetMacro(AssymmetrySensitivity, double);
  vtkSetMacro(StructureSensitivity, double);
  vtkGetMacro(StructureSensitivity, double);
  vtkSetMacro(BlobnessSensitivity, double);
  vtkGetMacro(BlobnessSensitivity, double);
  vtkSetMacro(GradientSensitivity, double);
  vtkGetMacro(GradientSensitivity, double);

protected:
  vtkImageFrangiFilter();
  ~vtkImageFrangiFilter();

private:

  template<class T>
  void SimpleExecute(vtkImageData* input, vtkImageData* output);

  double Sheet;
  double Line;
  double Blob;

  double AssymmetrySensitivity;
  double StructureSensitivity;
  double BlobnessSensitivity;
  double GradientSensitivity;
};

#endif