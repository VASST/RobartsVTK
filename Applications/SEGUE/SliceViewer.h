#ifndef __SLICE_VIEWER_H__
#define __SLICE_VIEWER_H__


#include <QtGui>
#include "vtkImageData.h"
#include "QVTKWidget.h"
#include "LURange.h"
#include "vtkImageLuminance2.h"
#include "qspinbox.h"
#include "qpushbutton.h"
#include "qslider.h"
#include "qlayout.h"
#include "qlayoutitem.h"
//!
/*!
 * This
 *
 *\author Usaf E. Aladl 2006
 *
 *
 *\b Revisions:
 * - 2006/06 first implementation
 * - 
 */

class QVTKWidget;
class QHBoxLayout;
class vtkRenderer;
class vtkRenderWindow; 
class vtkRenderWindowInteractor;
class vtkImagePlaneWidget;
class vtkLookupTable;
class vtkTextActor;
class vtkImageMapToColors;
class vtkImageActor;
class vtkPoints;
class vtkCommand;
class vtkExtractVOI;
//////////////////////////////////////////////////////////////////////////
class SliceViewer : public QWidget
{
  Q_OBJECT

  public:
  SliceViewer(QWidget *parent=0,Qt::WindowFlags f = 0);
  virtual ~SliceViewer( void );

  vtkRenderWindowInteractor* GetInteractor(){return mView->GetInteractor();};
  vtkRenderWindow* GetWindow(){return mView->GetRenderWindow();};
  vtkRenderer* GetRenderer(){return mRenderer;};
  virtual void Render();
  void SetBackgroundColor(double r, double g, double b){mColor[0]=r; mColor[1]=g; mColor[2]=b;};
  void SetOrientation(int o){this->mOrientation = o;};
  int GetOrientation(){ return this->mOrientation;};
  void SetVolume( vtkImageData * v){this->mVolume=v;};
  void SetLookupTable(vtkLookupTable* ct){this->mColorTable=ct;};
  virtual void Create();
  virtual void SetSliceNumber(int n);
  int GetSliceNumber(){return this->mSliceNumber;};
  void SetContour(vtkPoints* p){this->mContour=p;};
  vtkPoints* GetContour(){return this->mContour;};
  vtkImageData* GetBinary(){return this->mBin;};
  void SetBin(vtkImageData* im){this->mBin=im;};
  void SetCut(vtkImageData* im){this->mCut=im;};
  void SetBinaryColorTable(vtkLookupTable* ct);
  void SetCutColorTable(vtkLookupTable* ct);
  vtkLookupTable* GetBinaryColorTable(){return this->mLabels;};
  void updateMouseMove();
  int GetState(){ return this->state; }
protected:
  void createPlaneWidget();
  void createPlaneWidget2();
  void createPlaneWidget3();
  void setupCamera();
  void addImageActor();
  void addImageActor2();
  void addText();
  void updateText();
  void drawRegion(double x,double y,double z);
  void eraseRegion(double x,double y,double z);
  void createSlicer();
  void addControls();
protected:
  QHBoxLayout*  mLayout;
  QVTKWidget*  mView;
  vtkRenderer      * mRenderer;
  vtkImageData* mVolume;
  vtkImageData* mBin;
  vtkImageData* mCut;
  vtkImagePlaneWidget* mPlaneWidget;
  vtkImagePlaneWidget* mPlaneWidget2;
  vtkImagePlaneWidget* mPlaneWidget3;
  double  mColor[3];
  vtkLookupTable*		mColorTable;
  vtkLookupTable*		mLabels;
  vtkLookupTable*		mLabelsCut;
  int mID;
  int	mSliceNumber;
  int     mOrientation; //Z=2, Y=1, X=0
  vtkTextActor* mTextActor;
  std::string mText;
  vtkPoints* mContour;
  double	mZoomFactor;
  int mRadius;
  double mAlpha;
  int mButton; // Left 0 middle 1 right 2 Unknown -1
  QSlider* mSlider;
  QSpinBox* mSpinBox;
  QPushButton* mTButton;
  vtkImageLuminance2* mLum;
  double mCoef[3];
  int state;
signals:
  void sliceChanged(int);
  void viewClicked(int);

  void crossSelected(int, int, int);

public slots:
  bool eventFilter( QObject *o, QEvent *e );
  void sliceSlot( int );
  void orientationSlot( int );
  void alphaSlot(int);
  void SelectPoint( );
  void ErasePoint( );
  void SelectCross( );
  void radiusSlot( int );
  void labelSlot(int);
  void buttonClicked();
  void cleanClicked();
};

#endif
