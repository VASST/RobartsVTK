
#include "vtkDiscreteDynamicContour.h"
#include "vtkObjectFactory.h"
#include "vtkCardinalSpline.h"
#include "vtkPiecewiseFunction.h"
#include "vtkImageData.h"
#include "vtkImageGaussianSmooth.h"
#include "vtkImageLuminance.h"
#include "vtkImageGradientMagnitude.h"
#include "vtkImageGradient.h"
#include "vtkImageMagnitude.h"
#include "vtkImageCast.h"
#include "vtkImageFlip.h"
#include "vtkPNGWriter.h"

class vtkEnhancedCardinalSpline : public vtkCardinalSpline
{
public:  
    static vtkEnhancedCardinalSpline *New();
    vtkTypeRevisionMacro(vtkEnhancedCardinalSpline,vtkCardinalSpline);

    float EvaluateFirstDerivative(float t)
    {
      int i, index;
      int size = this->PiecewiseFunction->GetSize ();
      double *intervals;
      double *coefficients;

      // make sure we have at least 2 points
      if (size < 2)
        {
        vtkErrorMacro("Cannot evaluate a spline with less than 2 points. # of points is: " << size);
        return 0.0;
        }

      // check to see if we need to recompute the spline
      if (this->ComputeTime < this->GetMTime ())
        {
        this->Compute ();
        }   

      intervals = this->Intervals;
      coefficients = this->Coefficients;

      if ( this->Closed )
        {
        size = size + 1;
        }

      // clamp the function at both ends
      if (t < intervals[0])
        {
        t = intervals[0];
        }
      if (t > intervals[size - 1])
        {
        t = intervals[size - 1];
        }

      // find pointer to cubic spline coefficient
      index = 0;
      for (i = 1; i < size; i++)
        {
        index = i - 1;
        if (t < intervals[i])
          {
          break;
          }
        }

      // calculate offset within interval
      t = (t - intervals[index]);

      // evaluate y'
      return 3 * t * t * *(coefficients + index * 4 + 3)
               + 2 * t * *(coefficients + index * 4 + 2)
                       + *(coefficients + index * 4 + 1);
    }

    float EvaluateSecondDerivative(float t)
    {
      int i, index;
      int size = this->PiecewiseFunction->GetSize ();
      double *intervals;
      double *coefficients;

      // make sure we have at least 2 points
      if (size < 2)
        {
        vtkErrorMacro("Cannot evaluate a spline with less than 2 points. # of points is: " << size);
        return 0.0;
        }

      // check to see if we need to recompute the spline
      if (this->ComputeTime < this->GetMTime ())
        {
        this->Compute ();
        }   

      intervals = this->Intervals;
      coefficients = this->Coefficients;

      if ( this->Closed )
        {
        size = size + 1;
        }

      // clamp the function at both ends
      if (t < intervals[0])
        {
        t = intervals[0];
        }
      if (t > intervals[size - 1])
        {
        t = intervals[size - 1];
        }

      // find pointer to cubic spline coefficient
      index = 0;
      for (i = 1; i < size; i++)
        {
        index = i - 1;
        if (t < intervals[i])
          {
          break;
          }
        }

      // calculate offset within interval
      t = (t - intervals[index]);

      // evaluate y
      return 6 * t * *(coefficients + index * 4 + 3)
               + 2 * *(coefficients + index * 4 + 2);
    }
};

vtkCxxRevisionMacro(vtkEnhancedCardinalSpline, "$Revision: 1.1 $");
vtkStandardNewMacro(vtkEnhancedCardinalSpline);

vtkStandardNewMacro(vtkDiscreteDynamicContour);

vtkDiscreteDynamicContour::vtkDiscreteDynamicContour() : vtkObject()
{
    mWint = 0.2;
    mWext = 2.0f;
    mWdamp = -0.5f;
    mTOL = 0.1f;

    mSigma = 20.0f;

    mSteps = 0;
    mMaxIter = 64;

    mpContourX = vtkEnhancedCardinalSpline::New();
    mpContourX->ClosedOn();
    mpContourY = vtkEnhancedCardinalSpline::New();
    mpContourY->ClosedOn();
}

void vtkDiscreteDynamicContour::SetSigma(float sigma)
{
    mSigma = sigma;
}
    
void vtkDiscreteDynamicContour::SetInput(vtkImageData *input)
{
    vtkImageData *gradImg, *magImg;

    vtkImageCast *cast = vtkImageCast::New();
    cast->SetOutputScalarTypeToFloat();

    vtkImageLuminance *lum = NULL;
    if(input->GetNumberOfScalarComponents() > 1)
    {
      lum = vtkImageLuminance::New();
      lum->SetInput(input);
      cast->SetInput(lum->GetOutput());
    }
    else
    {
      cast->SetInput(input);
    }
    
    vtkImageGaussianSmooth *smooth = vtkImageGaussianSmooth::New();
    smooth->SetDimensionality(2);
    smooth->SetStandardDeviation(mSigma, mSigma);
    smooth->SetRadiusFactors(4*mSigma+1, 4*mSigma+1);
    smooth->SetInput(cast->GetOutput());

    vtkImageGradientMagnitude *gradmag = vtkImageGradientMagnitude::New();
    gradmag->SetDimensionality(2);
    gradmag->HandleBoundariesOn();
    gradmag->SetInput(smooth->GetOutput());

    vtkImageGradient *grad = vtkImageGradient::New();
    grad->SetDimensionality(2);
    grad->HandleBoundariesOn();
    grad->SetInput(gradmag->GetOutput());

    vtkImageMagnitude *mag = vtkImageMagnitude::New();
    mag->SetInput(grad->GetOutput());

    grad->Update();
    gradImg = grad->GetOutput();
    mag->Update();
    magImg = mag->GetOutput();

    // Find the maximum magnitude
    int i, j, extents[6], incs[3];
    float *imgdata, maxMag=-1;
    magImg->GetExtent(extents);
    imgdata = (float *)magImg->GetScalarPointerForExtent(extents);
    magImg->GetIncrements(incs);
    for(j = extents[2]; j <= extents[3]; j++)
    {
      for(i = extents[0]; i <= extents[1]; i++)
      {
        if(imgdata[i*incs[0]+j*incs[1]] > maxMag) maxMag = imgdata[i*incs[0]+j*incs[1]];
      }
    }

    // Normalize the gradient to [-2,2]
    gradImg->GetExtent(extents);
    imgdata = (float *)gradImg->GetScalarPointerForExtent(extents);
    gradImg->GetIncrements(incs);
    for(j = extents[2]; j <= extents[3]; j++)
    {
      for(i = extents[0]; i <= extents[1]; i++)
      {
        imgdata[i*incs[0]+j*incs[1]] = 2.0f * imgdata[i*incs[0]+j*incs[1]] / maxMag;
        imgdata[i*incs[0]+j*incs[1]+1] = 2.0f * imgdata[i*incs[0]+j*incs[1]+1] / maxMag;
      }
    }

    // Make a copy of the image
    mpVelField = vtkImageData::New();
    mpVelField->DeepCopy(gradImg);

    // Free the temporary images
    magImg->Delete();
    gradImg->Delete();
    grad->Delete();
    gradmag->Delete();
    smooth->Delete();
    if(lum) lum->Delete();
    cast->Delete();
}

void vtkDiscreteDynamicContour::ResetContour(void)
{
    mpContourX->RemoveAllPoints();
    mpContourY->RemoveAllPoints();

    mVelocityX.clear();
    mVelocityY.clear();

    mAccelX.clear();
    mAccelY.clear();

    mSteps = 0;
}

void vtkDiscreteDynamicContour::AddPoint(float x, float y)
{
    mpContourX->AddPoint(mVelocityX.size(), x);
    mpContourY->AddPoint(mVelocityX.size(), y);

    mVelocityX.push_back(0);
    mVelocityY.push_back(0);

    mAccelX.push_back(0);
    mAccelY.push_back(0);
}

void vtkDiscreteDynamicContour::SetMaxIter(int maxIter)
{
    mMaxIter = maxIter;
}

void vtkDiscreteDynamicContour::Update(float &maxaccel)
{
    int i;

    maxaccel = -1;

    // Compute the force and update the velocity at each point
    for(i = 0; i < mVelocityX.size(); i++)
    {
        float *extImg;

        // Apply the dampning
        mAccelX[i] = mWdamp * mVelocityX[i];
        mAccelY[i] = mWdamp * mVelocityY[i];

        // Apply the internal force
        float tangentLen = sqrt(pow(mpContourX->EvaluateFirstDerivative(i),2)+
                                pow(mpContourY->EvaluateFirstDerivative(i),2));
        mAccelX[i] += mWint * mpContourX->EvaluateSecondDerivative(i) / tangentLen;
        mAccelY[i] += mWint * mpContourY->EvaluateSecondDerivative(i) / tangentLen;

        // Sample and apply the external force
        int incs[3];
        mpVelField->GetIncrements(incs);
        extImg = (float *)mpVelField->GetScalarPointer();
        int u = int(mpContourX->Evaluate(i)+0.5);
        int v = int(mpContourY->Evaluate(i)+0.5);
        mAccelX[i] += mWext * extImg[u*incs[0]+v*incs[1]];
        mAccelY[i] += mWext * extImg[u*incs[0]+v*incs[1]+1];

        // Compute speed, accelaration magnitude, and their maximums
        float absAccel = float(sqrt(mAccelX[i]*mAccelX[i]+mAccelY[i]*mAccelY[i]));
        if(absAccel > maxaccel) maxaccel = absAccel;
    }
}

int vtkDiscreteDynamicContour::Step(void)
{
    int i;
    float deltaT = 1.0f;
    float maxaccel, maxvel=-1;

    // Compute the new forces
    Update(maxaccel);

    // Update the velocities
    for(i = 0; i < mVelocityX.size(); i++)
    {
        mVelocityX[i] += mAccelX[i] * deltaT;
        mVelocityY[i] += mAccelY[i] * deltaT;

        float absVel = float(sqrt(mVelocityX[i]*mVelocityX[i]+mVelocityY[i]*mVelocityY[i]));
        if(absVel > maxvel) maxvel = absVel;
    }

    // If the forces and velocities are small enough, or max iterations reached, stop
    mSteps++;
    if(((maxvel < mTOL) && (maxaccel < mTOL)) || (mSteps > mMaxIter)) return 2;

    // Find the ideal time-delta?
    if(maxvel > 1.0f) deltaT = 1.0f / maxvel;

    // Create new contours
    vtkEnhancedCardinalSpline *newxCont = vtkEnhancedCardinalSpline::New();
    newxCont->ClosedOn();
    vtkEnhancedCardinalSpline *newyCont = vtkEnhancedCardinalSpline::New();
    newyCont->ClosedOn();

    int extents[6];
    mpVelField->GetExtent(extents);
    for(i = 0; i < mVelocityX.size(); i++)
    {
        int u = int(mpContourX->Evaluate(i) + mVelocityX[i] * deltaT + 0.5);
        if(u < extents[0]) u = extents[0];
        if(u > extents[1]) u = extents[1];
        int v = int(mpContourY->Evaluate(i) + mVelocityY[i] * deltaT + 0.5);
        if(v < extents[2]) u = extents[2];
        if(v > extents[3]) u = extents[3];

        newxCont->AddPoint(i, u);
        newyCont->AddPoint(i, v);
    }

    // Delete the old contours, and replace them
    mpContourX->Delete(); mpContourY->Delete();
    mpContourX = newxCont; mpContourY = newyCont;

    return 1;
}

vtkCardinalSpline *vtkDiscreteDynamicContour::GetOutputX()
{
    return mpContourX;
}

vtkCardinalSpline *vtkDiscreteDynamicContour::GetOutputY()
{
    return mpContourY;
}
