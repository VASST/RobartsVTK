#include "SliceViewer.h"
#include <iostream>
#include "QVTKWidget.h"
#include "vtkRenderWindow.h"
#include "vtkRenderWindowInteractor.h"
#include "vtkLookupTable.h"
#include "vtkInteractorStyleTrackballCamera.h"
#include "vtkInteractorStyleImage.h"
#include "vtkRenderer.h"
#include "vtkCamera.h"
#include "vtkProperty.h"
#include "vtkInteractorStyleImage.h"

#include "vtkImagePlaneWidget.h"
#include "vtkLookupTable.h"
#include "vtkImageActor.h"
#include "vtkTextActor.h"
#include "vtkTextProperty.h"
#include "vtkImageMapToColors.h"
#include "vtkImageActor.h"
#include "vtkPoints.h"
#include "vtkExtractVOI.h"
#include <sstream>

#include "qframe.h"
//////////////////////////////////////////////////////////////////////////

class MouseCommand : public vtkCommand {
public:
	vtkTypeMacro(MouseCommand,vtkCommand);
	static MouseCommand *New()
    {
      return new MouseCommand();
    }

	SliceViewer* self;

	void Execute(vtkObject *caller, unsigned long eventId, void *callData){
		int state = self->GetState();
		switch(state){
		case 1:
			self->SelectPoint();
			break;
		case 2:
			self->ErasePoint();
			break;
		case 3:
			self->SelectCross();
			break;
		default:
			break;
		}
	}
};

// from kbarcode
static const char* remove_xpm[]=
{
    "16 16 15 1",
    " 	c None",
    ".	c #B7B7B7",
    "+	c #FFFFFF",
    "@	c #6E6E6E",
    "#	c #E9E9E9",
    "$	c #E4E4E4",
    "%	c #000000",
    "&	c #DEDEDE",
    "*	c #D9D9D9",
    "=	c #D4D4D4",
    "-	c #CECECE",
    ";	c #C9C9C9",
    ">	c #C3C3C3",
    ",	c #BEBEBE",
    "'	c #B9B9B9",

    "...............+",
    ".@@@@@@@@@@@@@@+",
    ".@+++++++++++.@+",
    ".@+          .@+",
    ".@+  %    %  .@+",
    ".@+ %%%  %%% .@+",
    ".@+  %%%%%%  .@+",
    ".@+   %%%%   .@+",
    ".@+   %%%%   .@+",
    ".@+  %%%%%%  .@+",
    ".@+ %%%  %%% .@+",
    ".@+  %    %  .@+",
    ".@+           @+",
    ".@............@+",
    ".@@@@@@@@@@@@@@+",
    "++++++++++++++++"
};


//////////////////////////////////////////////////////////////////////////
#include "vtkImageClip.h"
#include "vtkImageCast.h"
vtkImageData* GetSlice(vtkImageData* vol, int n)
{
int ex[6];
vol->GetExtent(ex);
vtkImageClip* mClipSlice=vtkImageClip::New();

ex[4]=n;
ex[5]=n;

mClipSlice->SetInputDataObject(vol);
mClipSlice->SetOutputWholeExtent(ex);
mClipSlice->ClipDataOn();
mClipSlice->Update();


vtkImageCast* ic=vtkImageCast::New();
ic->SetOutputScalarTypeToFloat();
ic->SetInputDataObject(mClipSlice->GetOutput());
ic->Update();

vtkImageData* im=ic->GetOutput();

return (im);

}
//////////////////////////////////////////////////////////////////////////
template <typename T>
std::string toString(const T &thing) 
{
    std::ostringstream os;
    os << thing;
    return os.str();
}
//////////////////////////////////////////////////////////////////////////
class OnMouseMove : public vtkCommand
{
public:

SliceViewer *that;
OnMouseMove(SliceViewer *n){that=n;};
 virtual void Execute(vtkObject *caller, unsigned long eid, void*)
 {
	 that->updateMouseMove();
 }

};
//////////////////////////////////////////////////////////////////////////

#include "vtkInteractorStyleRubberBandZoom.h"
//////////////////////////////////////////////////////////////////////////
SliceViewer::SliceViewer(QWidget *parent,Qt::WindowFlags f )
:QWidget(parent,f)
{

setSizePolicy(QSizePolicy::Expanding,QSizePolicy::Expanding);	
mLayout = new QHBoxLayout( this); 
mRadius=2;
mButton=-1;
mPlaneWidget=NULL;
mPlaneWidget2=NULL;
mPlaneWidget3=NULL;
mZoomFactor = 1.0;
mColor[0]=0.0;
mColor[1]=0.0;
mColor[2]=0.0;
//mColor[0]=0.674509;
//mColor[1]=0.6588;
//mColor[2]=0.6;
mCoef[0]=1.0;//0.30;
mCoef[1]=0.0;//0.59;
mCoef[2]=0.0;//0.11;

mLum=NULL;
mVolume=NULL;
mColorTable=NULL;
mSliceNumber=-1;
mOrientation=2;
mID=0;
mContour=vtkPoints::New();
mView=new QVTKWidget(this);

vtkRenderWindow* win=vtkRenderWindow::New();

//win->StereoCapableWindowOff();
//
//win->SetStereoTypeToDresden();
//win->SetStereoTypeToRedBlue();
//win->StereoRenderOn(); 
//win->SetStereoTypeToCrystalEyes();
//win->SetStereoTypeToInterlaced(); 

 
mView->SetRenderWindow(win);
win->Delete();

mView->show();
mView->setSizePolicy(QSizePolicy::Expanding,QSizePolicy::Expanding);
mLayout->addWidget(mView);

mRenderer = vtkRenderer::New();
mRenderer->SetBackground(mColor[0],mColor[1],mColor[2]);
mRenderer->SetBackingStore(false);
mView->GetRenderWindow()->AddRenderer(mRenderer);
mRenderer->Delete();

mView->installEventFilter(this);

mView->GetInteractor()->SetInteractorStyle(vtkInteractorStyleImage::New());
//mView->GetInteractor()->SetInteractorStyle(vtkInteractorStyleTrackballCamera::New());
//mView->GetInteractor()->SetInteractorStyle(vtkInteractorStyleRubberBandZoom::New());

mTextActor=vtkTextActor::New();
resize( QSize(300, 300).expandedTo(minimumSizeHint()) );

}
//////////////////////////////////////////////////////////////////////////
SliceViewer::~SliceViewer()
{
	delete mView;
	mContour->Delete();
	mPlaneWidget->Delete();
	mPlaneWidget2->Delete();
	mPlaneWidget3->Delete();
	delete mLayout;
}
//////////////////////////////////////////////////////////////////////////
void SliceViewer::updateMouseMove()
{
std::cout<<"Mouse Move"<<std::endl;

int newPosition[3];
mView->GetInteractor()->GetEventPosition( newPosition );

std::cout << "newPosition " << newPosition[0] << " " << newPosition[1] << " " << newPosition[2] << std::endl;


}
//////////////////////////////////////////////////////////////////////////
bool SliceViewer::eventFilter( QObject *o, QEvent *e )
{

	if( e->type() == QEvent::MouseButtonPress ) 
		{   
			QMouseEvent *ee= static_cast<QMouseEvent *>(e);
			QPoint p=ee->pos();

			
			if( ee->buttons() & Qt::LeftButton)
				state = 1;
			else if( ee->buttons() & Qt::RightButton )
				state = 2;
			else if( ee->buttons() & Qt::MiddleButton )
				state = 3;
			else
				state = 0;

			this->repaint();

	  }
	else if(e->type() == QEvent::MouseButtonRelease)
		this->state = 0;

    return false;
}

//////////////////////////////////////////////////////////////////////////
void SliceViewer::SelectPoint( )
{
double  xyzv[4]; 
mPlaneWidget->GetCurrentCursorPosition( xyzv);
this->drawRegion(xyzv[0], xyzv[1], xyzv[2]);
Render();
}

void SliceViewer::ErasePoint( )
{
double  xyzv[4]; 
mPlaneWidget->GetCurrentCursorPosition( xyzv);
this->eraseRegion(xyzv[0], xyzv[1], xyzv[2]);
Render();
}

void SliceViewer::SelectCross( ){
double  xyzv[4]; 
mPlaneWidget->GetCurrentCursorPosition(xyzv);
this->crossSelected(xyzv[0],xyzv[1],xyzv[2]);
}
//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
void SliceViewer::Render()
{


if(mPlaneWidget!=NULL)
{
mPlaneWidget->SetSliceIndex(mSliceNumber);
mPlaneWidget2->SetSliceIndex(mSliceNumber);
mPlaneWidget3->SetSliceIndex(mSliceNumber);
}


//setupCamera();
mRenderer->ResetCameraClippingRange();
mView->GetRenderWindow()->Render();
this->mRenderer->Render();
}
//////////////////////////////////////////////////////////////////////////
void SliceViewer::createPlaneWidget()
{

if(mPlaneWidget!=NULL)
{
mPlaneWidget->Off();
mPlaneWidget->Delete();
}


mPlaneWidget= vtkImagePlaneWidget::New();
mPlaneWidget->SetInteractor( mView->GetInteractor());
mPlaneWidget->TextureInterpolateOff();
mPlaneWidget->SetResliceInterpolateToNearestNeighbour();
mPlaneWidget->SetInputData(mVolume);

if(mVolume->GetNumberOfScalarComponents()==1)
{
mPlaneWidget->UserControlledLookupTableOn(); 
mPlaneWidget->SetLookupTable( mColorTable);
}else{
mPlaneWidget->GetColorMap()->SetOutputFormatToRGB();
mPlaneWidget->GetColorMap()->SetLookupTable(NULL);

}
mPlaneWidget->DisplayTextOn();
//mPlaneWidget->SetLeftButtonAction(0);
//mPlaneWidget->SetLeftButtonAutoModifier(1);
mPlaneWidget->SetMiddleButtonAction(0);
mPlaneWidget->SetMiddleButtonAutoModifier(1);
mPlaneWidget->SetRightButtonAction(0);
mPlaneWidget->SetRightButtonAutoModifier(1);
int ext[6];

mVolume->GetExtent(ext);
switch( mOrientation )
{
  case 0:
      {
		mPlaneWidget->SetKeyPressActivationValue('x');
		mPlaneWidget->SetPlaneOrientation(0);
		if(mSliceNumber==-1)mSliceNumber=(ext[1]-ext[0])/2;
		
		break;
	  }
 case 1:
      {
		mPlaneWidget->SetKeyPressActivationValue('y');
		mPlaneWidget->SetPlaneOrientation(1);
		if(mSliceNumber==-1)mSliceNumber=(ext[3]-ext[2])/2;
		break;
	  }
 case 2:
      {
		mPlaneWidget->SetKeyPressActivationValue('z');
		mPlaneWidget->SetPlaneOrientation(2);
		if(mSliceNumber==-1)mSliceNumber=(ext[5]-ext[4])/2;
		break;
	  }
}

 
mPlaneWidget->SetSliceIndex(mSliceNumber);
mPlaneWidget->On();	
}

//////////////////////////////////////////////////////////////////////////
void SliceViewer::Create()
{
double range[2];
if(mColorTable==NULL)
{

mVolume->GetScalarRange(range);

mColorTable=vtkLookupTable::New();
mColorTable->SetAlpha(1.0);
mColorTable->SetNumberOfColors(256);
mColorTable->SetRange(range);
mColorTable->Build();


int i;
double c;
for (i=0; i<256; i++) 
{
c=(double)i/255.0;	
mColorTable->SetTableValue(i,c,c,c);
}


}

//mLabelImage=new LabelImage();
//mLabelImage->CreateImage(mVolume);
//mLabelImage->test();
//mLabelImage->CreateColorTable();
//mLabelImage->mImage->GetScalarRange(range);
//mLabelImage->mColorTable->SetRange(range);
//std::cout<<"min="<<range[0]<<" max="<<range[1]<<std::endl;


createPlaneWidget();
createPlaneWidget3();
createPlaneWidget2();

MouseCommand* mc = MouseCommand::New();
mc->self = this;

mPlaneWidget->AddObserver(vtkCommand::StartInteractionEvent,	mc);
mPlaneWidget->AddObserver(vtkCommand::EndInteractionEvent,		mc);
mPlaneWidget->AddObserver(vtkCommand::InteractionEvent,			mc);
//mPlaneWidget->GetInteractor()->AddObserver(vtkCommand::MouseMoveEvent,					mc);
//mPlaneWidget->GetInteractor()->AddObserver(vtkCommand::MiddleButtonPressEvent,			mc);
//mPlaneWidget->GetInteractor()->AddObserver(vtkCommand::MiddleButtonReleaseEvent,		mc);

mPlaneWidget2->AddObserver(vtkCommand::StartInteractionEvent,	mc);
mPlaneWidget2->AddObserver(vtkCommand::EndInteractionEvent,		mc);
mPlaneWidget2->AddObserver(vtkCommand::InteractionEvent,		mc);
//mPlaneWidget2->GetInteractor()->AddObserver(vtkCommand::MouseMoveEvent,					mc);
//mPlaneWidget2->GetInteractor()->AddObserver(vtkCommand::MiddleButtonPressEvent,			mc);
//mPlaneWidget2->GetInteractor()->AddObserver(vtkCommand::MiddleButtonReleaseEvent,		mc);

mPlaneWidget3->AddObserver(vtkCommand::StartInteractionEvent,	mc);
mPlaneWidget3->AddObserver(vtkCommand::EndInteractionEvent,		mc);
mPlaneWidget3->AddObserver(vtkCommand::InteractionEvent,		mc);
//mPlaneWidget3->GetInteractor()->AddObserver(vtkCommand::MouseMoveEvent,					mc);
//mPlaneWidget3->GetInteractor()->AddObserver(vtkCommand::MiddleButtonPressEvent,			mc);
//mPlaneWidget3->GetInteractor()->AddObserver(vtkCommand::MiddleButtonReleaseEvent,		mc);

mc->Delete();

if(mVolume->GetNumberOfScalarComponents()==3)
{
mLum=vtkImageLuminance2::New();
mLum->SetInputDataObject(mVolume);
mLum->SetCoef(mCoef);
mLum->Update();

}

//OnMouseMove *mouseMove =new OnMouseMove(this);
//mView->GetInteractor()->AddObserver(vtkCommand::MouseMoveEvent,mouseMove);
//mView->GetInteractor()->AddObserver(vtkCommand::LeftButtonPressEvent,mouseMove);
//mView->GetInteractor()->AddObserver(vtkCommand::RightButtonPressEvent,mouseMove);



setupCamera();

updateText();
addText();
mRenderer->ResetCamera();
mView->GetRenderWindow()->Render();
addControls();
}

//////////////////////////////////////////////////////////////////////////
void SliceViewer::addText()
{

mTextActor->SetInput(mText.c_str());
mTextActor->SetTextScaleModeToNone();

  vtkTextProperty* textprop = mTextActor->GetTextProperty();
  textprop->SetColor(1,1,0);
  textprop->SetFontFamilyToArial();
  textprop->SetFontSize(16);
  textprop->BoldOn();
  textprop->ItalicOff();
  textprop->ShadowOff();
  textprop->SetJustificationToLeft();
  textprop->SetVerticalJustificationToBottom();

  vtkCoordinate* coord = mTextActor->GetPositionCoordinate();
  coord->SetCoordinateSystemToNormalizedViewport();
  coord->SetValue(.01, .88);
mRenderer->AddActor2D(mTextActor);
mTextActor->Delete();
mTextActor->VisibilityOn();
}
//////////////////////////////////////////////////////////////////////////
void SliceViewer::updateText()
{
mText=toString(mSliceNumber);
mTextActor->SetInput(mText.c_str());
}
//////////////////////////////////////////////////////////////////////////
void  SliceViewer::SetSliceNumber( int slice )
{
  int ext[6];
  mVolume->GetExtent( ext );

  switch( mOrientation )
    {
  case 0:
      if ((slice>=ext[0]) && (slice<=ext[1]))
      {
		  mSlider->setValue(mSliceNumber);
		  mSliceNumber = slice;
      }
      break;
  case 1:
      if ((slice>=ext[2]) && (slice<=ext[3]))
      {
		  mSlider->setValue(mSliceNumber);
		  mSliceNumber = slice;
      }
      break;
  case 2:
      if ((slice>=ext[4]) && (slice<=ext[5]))
      {
		  mSlider->setValue(mSliceNumber);
		  mSliceNumber = slice;
      }
      break;
    }
 updateText();
}

//////////////////////////////////////////////////////////////////////////
void SliceViewer::setupCamera()
{
if( mVolume==NULL ) return;
  

double spacing[3];
double origin[3];
double focalPoint[3];
double position[3];
int   dimensions[3];

mVolume->GetSpacing(spacing);
mVolume->GetOrigin(origin);
mVolume->GetDimensions(dimensions);
vtkCamera* cam=mRenderer->GetActiveCamera();

  for ( unsigned int cc = 0; cc < 3; cc++)
    {
    focalPoint[cc] = origin[cc] + ( spacing[cc] * dimensions[cc] ) / 2.0;
    position[cc]   = focalPoint[cc];
    }

  int idx = 0;
  switch( mOrientation )
    {
  case 0:
      {
      idx = 0;
      cam->SetViewUp (     0,  0,  1 );
      break;
      }
  case 1:
      {
      idx = 1;
      cam->SetViewUp (     0,  0,  1 );
      break;
      }
  case 2:
      {
      idx = 2;
      cam->SetViewUp (     0,  -1,  0 );
      break;
      }
    }

  const double distanceToFocalPoint = 600;
  position[idx] += distanceToFocalPoint;

  cam->SetPosition (   position );
  cam->SetFocalPoint ( focalPoint );

#define myMAX(x,y) (((x)>(y))?(x):(y))  

int d1=(idx + 1) % 3;
int d2=(idx + 2) % 3;
 
double max = myMAX( spacing[d1] * dimensions[d1],    spacing[d2] * dimensions[d2]);

cam->SetParallelScale( max/2*mZoomFactor );

}
//////////////////////////////////////////////////////////////////////////
void SliceViewer::createPlaneWidget2()
{

if(mPlaneWidget2!=NULL)
{
mPlaneWidget2->Off();
mPlaneWidget2->Delete();
}


mPlaneWidget2= vtkImagePlaneWidget::New();
mPlaneWidget2->SetInteractor( mView->GetInteractor());
mPlaneWidget2->TextureInterpolateOff();
mPlaneWidget2->SetResliceInterpolateToNearestNeighbour();
mPlaneWidget2->SetInputData(mBin);
mPlaneWidget2->UserControlledLookupTableOn(); 
mPlaneWidget2->SetLookupTable( mLabels);
mPlaneWidget2->DisplayTextOff();
mPlaneWidget2->SetMiddleButtonAction(0);
mPlaneWidget2->SetMiddleButtonAutoModifier(1);
mPlaneWidget2->SetRightButtonAction(0);
mPlaneWidget2->SetRightButtonAutoModifier(1);



int ext[6];


mVolume->GetExtent(ext);
switch( mOrientation )
{
  case 0:
      {
		mPlaneWidget2->SetKeyPressActivationValue('x');
		mPlaneWidget2->SetPlaneOrientation(0);
		if(mSliceNumber==-1)mSliceNumber=(ext[1]-ext[0])/2;
		
		break;
	  }
 case 1:
      {
		mPlaneWidget2->SetKeyPressActivationValue('y');
		mPlaneWidget2->SetPlaneOrientation(1);
		if(mSliceNumber==-1)mSliceNumber=(ext[3]-ext[2])/2;
		break;
	  }
 case 2:
      {
		mPlaneWidget2->SetKeyPressActivationValue('z');
		mPlaneWidget2->SetPlaneOrientation(2);
		if(mSliceNumber==-1)mSliceNumber=(ext[5]-ext[4])/2;
		break;
	  }
}

 
mPlaneWidget2->SetSliceIndex(mSliceNumber);
mPlaneWidget2->On();	
}

//////////////////////////////////////////////////////////////////////////
void SliceViewer::createPlaneWidget3()
{

if(mPlaneWidget3!=NULL)
{
mPlaneWidget3->Off();
mPlaneWidget3->Delete();
}

mPlaneWidget3= vtkImagePlaneWidget::New();
mPlaneWidget3->SetInteractor( mView->GetInteractor());
mPlaneWidget3->TextureInterpolateOff();
mPlaneWidget3->SetResliceInterpolateToNearestNeighbour();
mPlaneWidget3->SetInputData(mCut);
mPlaneWidget3->UserControlledLookupTableOn(); 
mPlaneWidget3->SetLookupTable( this->mLabelsCut );
mPlaneWidget3->GetPlaneProperty()->SetOpacity(1.0);
mPlaneWidget3->DisplayTextOff();
mPlaneWidget3->SetMiddleButtonAction(0);
mPlaneWidget3->SetMiddleButtonAutoModifier(1);
mPlaneWidget3->SetRightButtonAction(0);
mPlaneWidget3->SetRightButtonAutoModifier(1);


int ext[6];


mVolume->GetExtent(ext);
switch( mOrientation )
{
  case 0:
      {
		mPlaneWidget3->SetKeyPressActivationValue('x');
		mPlaneWidget3->SetPlaneOrientation(0);
		if(mSliceNumber==-1)mSliceNumber=(ext[1]-ext[0])/2;
		
		break;
	  }
 case 1:
      {
		mPlaneWidget3->SetKeyPressActivationValue('y');
		mPlaneWidget3->SetPlaneOrientation(1);
		if(mSliceNumber==-1)mSliceNumber=(ext[3]-ext[2])/2;
		break;
	  }
 case 2:
      {
		mPlaneWidget3->SetKeyPressActivationValue('z');
		mPlaneWidget3->SetPlaneOrientation(2);
		if(mSliceNumber==-1)mSliceNumber=(ext[5]-ext[4])/2;
		break;
	  }
}

 
mPlaneWidget3->SetSliceIndex(mSliceNumber);
mPlaneWidget3->On();	
}
//////////////////////////////////////////////////////////////////////////
void SliceViewer::sliceSlot( int n)
{
SetSliceNumber(n);
emit sliceChanged(mSliceNumber);
Render();
}

//////////////////////////////////////////////////////////////////////////
void SliceViewer::orientationSlot( int n)
{
int ex[6];
mVolume->GetExtent(ex);
mOrientation=n;
mSliceNumber=-1;

setupCamera();
Render();
}
//////////////////////////////////////////////////////////////////////////
void SliceViewer::radiusSlot( int n)
{
mRadius=n;
}
//////////////////////////////////////////////////////////////////////////
void SliceViewer::alphaSlot( int n)
{
mAlpha=(double)n/100.0;


double rgb[3];

int s=mLabels->GetNumberOfColors();
for(int i=0; i<s; i++)
{
mLabels->GetColor(i,rgb);
mLabels->SetTableValue(i,rgb[0],rgb[1],rgb[2],mAlpha);
}


mLabels->GetColor(0,rgb);
mLabels->SetTableValue(0,rgb[0],rgb[1],rgb[2],0.0);


Render();

}
//////////////////////////////////////////////////////////////////////////
void SliceViewer::drawRegion(double x,double y,double z)
{	


int xx,yy,zz;
xx=x;
yy=y;
zz=z;
int i,j;
double r=mRadius;

unsigned char* voxel;
int ex[6];
mBin->GetExtent(ex);
if(x<ex[0] || x>ex[1] || y<ex[2] || y>ex[3] || z<ex[4] || z>ex[5])return;

if(mRadius==0.0)
{
voxel=(unsigned char*)mBin->GetScalarPointer(xx, yy,zz);	
*voxel=(unsigned char)mID;	
mBin->Modified();
return;
}


int r2=2*r;

x=x-r;
y=y-r;
z=z-r;



if(mOrientation==2)
{

	for(i=0; i<r2; i++)
	{
	xx=x+i;
	for(j=0; j<r2; j++)
	{
	yy=y+j;


		if(xx<ex[0] || xx>ex[1] || yy<ex[2] || yy>ex[3] || zz<ex[4] || zz>ex[5])
		{
		}else{
		voxel=(unsigned char*)mBin->GetScalarPointer(xx, yy,zz);	
	*voxel=(unsigned char)mID;	
		}

	}
	}
mBin->Modified();
return;
}



if(mOrientation==1)
{

	for(i=0; i<r2; i++)
	{
	xx=x+i;
	for(j=0; j<r2; j++)
	{
	zz=z+j;


		if(xx<ex[0] || xx>ex[1] || yy<ex[2] || yy>ex[3] || zz<ex[4] || zz>ex[5])
		{
		}else{
		voxel=(unsigned char*)mBin->GetScalarPointer(xx, yy,zz);	
	*voxel=(unsigned char)mID;	
		}

	}
	}
mBin->Modified();
return;
}



if(mOrientation==0)
{

	for(i=0; i<r2; i++)
	{
	yy=y+i;
	for(j=0; j<r2; j++)
	{
	zz=z+j;


		if(xx<ex[0] || xx>ex[1] || yy<ex[2] || yy>ex[3] || zz<ex[4] || zz>ex[5])
		{
		}else{
		voxel=(unsigned char*)mBin->GetScalarPointer(xx, yy,zz);	
	*voxel=(unsigned char)mID;	
		}

	}
	}
return;
mBin->Modified();
}


}

void SliceViewer::eraseRegion(double x,double y,double z)
{	


int xx,yy,zz;
xx=x;
yy=y;
zz=z;
int i,j;
double r=mRadius;

unsigned char* voxel;
int ex[6];
mBin->GetExtent(ex);
if(x<ex[0] || x>ex[1] || y<ex[2] || y>ex[3] || z<ex[4] || z>ex[5])return;

if(mRadius==0.0)
{
voxel=(unsigned char*)mBin->GetScalarPointer(xx, yy,zz);	
*voxel=(unsigned char)mID;	
mBin->Modified();
return;
}


int r2=2*r;

x=x-r;
y=y-r;
z=z-r;



if(mOrientation==2)
{

	for(i=0; i<r2; i++)
	{
	xx=x+i;
	for(j=0; j<r2; j++)
	{
	yy=y+j;


		if(xx<ex[0] || xx>ex[1] || yy<ex[2] || yy>ex[3] || zz<ex[4] || zz>ex[5])
		{
		}else{
		voxel=(unsigned char*)mBin->GetScalarPointer(xx, yy,zz);	
	*voxel= (*voxel==(unsigned char)mID) ? 0 : *voxel;	
		}

	}
	}
mBin->Modified();
return;
}



if(mOrientation==1)
{

	for(i=0; i<r2; i++)
	{
	xx=x+i;
	for(j=0; j<r2; j++)
	{
	zz=z+j;


		if(xx<ex[0] || xx>ex[1] || yy<ex[2] || yy>ex[3] || zz<ex[4] || zz>ex[5])
		{
		}else{
		voxel=(unsigned char*)mBin->GetScalarPointer(xx, yy,zz);	
	*voxel= (*voxel==(unsigned char)mID) ? 0 : *voxel;	
		}

	}
	}
mBin->Modified();
return;
}



if(mOrientation==0)
{

	for(i=0; i<r2; i++)
	{
	yy=y+i;
	for(j=0; j<r2; j++)
	{
	zz=z+j;


		if(xx<ex[0] || xx>ex[1] || yy<ex[2] || yy>ex[3] || zz<ex[4] || zz>ex[5])
		{
		}else{
		voxel=(unsigned char*)mBin->GetScalarPointer(xx, yy,zz);	
	*voxel= (*voxel==(unsigned char)mID) ? 0 : *voxel;	
		}

	}
	}
return;
mBin->Modified();
}


}

//////////////////////////////////////////////////////////////////////////
void SliceViewer::labelSlot(int n)
{
this->mID=n;
}


//////////////////////////////////////////////////////////////////////////
void SliceViewer::addControls()
{


int ext[6];
mVolume->GetExtent(ext);
int range[2];
switch( mOrientation )
{
  case 0:
      {
		range[0]=ext[0];
		range[1]=ext[1];
		break;
	  }
 case 1:
      {
		range[0]=ext[2];
		range[1]=ext[3];
		break;
	  }
 case 2:
      {
		range[0]=ext[4];
		range[1]=ext[5];
		break;
	  }
}



QFrame* w0=new QFrame;
w0->setFrameStyle(QFrame::WinPanel | QFrame::Sunken);
//w0->setFrameStyle(QFrame::NoFrame);
//w0->setLineWidth(2);
QVBoxLayout* l= new QVBoxLayout;
w0->setLayout(l);

QSizePolicy sizePolicy( QSizePolicy::Fixed,QSizePolicy::Preferred);
    sizePolicy.setHorizontalStretch(0);
    sizePolicy.setVerticalStretch(0);
 QPixmap* pixx;
pixx= new QPixmap(remove_xpm);
mTButton=new QPushButton(w0);
mTButton->setIcon(QIcon(*pixx));
mTButton->setIconSize(QSize(22,22));
mTButton->resize(22, 22);
mTButton->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
connect( mTButton, SIGNAL( clicked() ), this, SLOT( buttonClicked() ) );


l->addWidget(mTButton);


mSlider=new QSlider(Qt::Vertical,w0);
mSlider->setMinimum(range[0]);
mSlider->setMaximum(range[1]);
mSlider->setValue(mSliceNumber);
mSlider->setTickPosition(QSlider::TicksRight);
l->addWidget(mSlider,0,Qt::AlignHCenter);



mSpinBox=new QSpinBox(w0);
mSpinBox->setMinimum(range[0]);
mSpinBox->setMaximum(range[1]);
mSpinBox->setValue(mSliceNumber);
l->addWidget(mSpinBox,0,Qt::AlignHCenter);


QObject::connect(mSlider, SIGNAL(valueChanged(int)), mSpinBox, SLOT(setValue(int)));
QObject::connect(mSpinBox, SIGNAL(valueChanged(int)), mSlider, SLOT(setValue(int)));

QObject::connect(mSlider, SIGNAL(valueChanged(int)), this, SLOT(sliceSlot(int)));



QPushButton* b=new QPushButton(w0);
b->setText( tr( "c" ) );
b->setFixedSize(22, 22);
b->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
connect( b, SIGNAL( clicked() ), this, SLOT( cleanClicked() ) );
l->addWidget(b);



mLayout->addWidget(w0);
}

//////////////////////////////////////////////////////////////////////////
void SliceViewer::buttonClicked()
{
emit viewClicked(mOrientation);
}
//////////////////////////////////////////////////////////////////////////
void SliceViewer::cleanClicked()
{
int i,j;
int ext[6];
unsigned char* val;
mBin->GetExtent(ext);

switch( mOrientation )
{
  case 0:
      {
		  for(i=ext[4]; i<=ext[5]; i++)
		  for(j=ext[2]; j<=ext[3]; j++)
		  {
			val=(unsigned char*)mBin->GetScalarPointer(mSliceNumber,j,i);
			*val=0;
		  }
		break;
	  }
 case 1:
      {
	  for(i=ext[4]; i<=ext[5]; i++)
		  for(j=ext[0]; j<=ext[1]; j++)
		  {
			val=(unsigned char*)mBin->GetScalarPointer(j,mSliceNumber,i);
			*val=0;	
		  }
		break;
	  }
 case 2:
      {
  for(i=ext[2]; i<=ext[3]; i++)
		  for(j=ext[0]; j<=ext[1]; j++)
		  {
			val=(unsigned char*)mBin->GetScalarPointer(j,i,mSliceNumber);
			*val=0;		
		  }
		break;
	  }
}

 
Render();
}


void SliceViewer::SetBinaryColorTable(vtkLookupTable* ct){
	this->mLabels=ct;
}

void SliceViewer::SetCutColorTable(vtkLookupTable* ct){
	this->mLabelsCut=ct;
}
