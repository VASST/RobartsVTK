#ifndef _VTKDDC_
#define _VTKDDC_

#include <vector>

#include "vtkObject.h"

class vtkImageData;
class vtkCardinalSpline;
class vtkEnhancedCardinalSpline;

class VTK_EXPORT vtkDiscreteDynamicContour : public vtkObject
{
    float mSigma;
    vtkImageData *mpVelField;
    vtkEnhancedCardinalSpline *mpContourX;
    vtkEnhancedCardinalSpline *mpContourY;
//BTX
    int mSteps;
    std::vector<float> mVelocityX, mVelocityY;
    std::vector<float> mAccelX, mAccelY;
//ETX

    float mWint;
    float mWext;
    float mWdamp;
    float mTOL;

    int mMaxIter;

    void Update(float &maxaccel);

public:
    static vtkDiscreteDynamicContour *New();
    vtkTypeMacro(vtkDiscreteDynamicContour,vtkObject);

    vtkDiscreteDynamicContour();

    void SetSigma(float sigma);
    void SetInput(vtkImageData *input);
    
    void ResetContour(void);
    void AddPoint(float x, float y);
    void SetMaxIter(int maxIter);

    // Simulation functions
    int Step(void);

    vtkCardinalSpline *GetOutputX();
    vtkCardinalSpline *GetOutputY();
};

#endif

