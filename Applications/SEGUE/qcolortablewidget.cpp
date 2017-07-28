#include "qcolortablewidget.h"
#include <QtGui>

#include <QString>
#include <QPainter>
#include <QPixmap>

#include <math.h>
#include <algorithm>


#include "vtkLookupTable.h"
//////////////////////////////////////////////////////////////////////////
template <typename T>
T mapRange(const T& val,
              const T& srcMin, const T& srcMax,
              const T& dstMin, const T& dstMax)
{
 return ((val-srcMin)*(dstMax-dstMin)/(srcMax-srcMin)+dstMin);
}
//////////////////////////////////////////////////////////////////////////
// This is a ripoff from project      : (X)MedCon by Erik Nolf 
/* cti source */
struct {int n,r,g,b,dr,dg,db; }
  bitty[] = {  {32,0,0,0,2,0,4},          /* violet to indigo */
               {32,64,0,128,-2,0,4},      /* indigo to blue */
               {32,0,0,255,0,8,-8},       /* blue to green */
               {64,0,255,0,4,0,0},        /* green to yellow */
               {32,255,255,0,0,-2,0},     /* yellow to orange */
               {64,255,192,0,0,-3,0} };   /* orange to red */


typedef unsigned char cbyte;
CRgb cpat[]={
{0,0,0},
{255*1.0, 255*0.36, 255*0.36},
{255*1.0, 255*0.8431, 255*0.0},
{255*0.751640, 255*0.606480, 255*0.226480},
{255*0.507540, 255*0.507540, 255*0.507540},
{255*0.075680, 255*0.614240, 255*0.075680},
{255*0.540000, 255*0.890000, 255*0.630000},
{255*1.000000, 255*0.829000, 255*0.829000},
{255*0.614240, 255*0.041360, 255*0.041360},
{255*0.396000, 255*0.741510, 255*0.691020},
{255*0.780392, 255*0.568627, 255*0.113725},
{255*0.714000, 255*0.428400, 255*0.181440},
{255*0.400000, 255*0.400000, 255*0.400000},
{255*0.703800, 255*0.270480, 255*0.082800},
{0,0,255},
{255,255,255},
{255,255,0}
};


//////////////////////////////////////////////////////////////////////////
QColorTableWidget::QColorTableWidget( int ctn,QWidget *parent)
:QComboBox(parent)
{
mOpacity=1.0;
mColorTable.resize(256);
mLookupTable=vtkLookupTable::New();
mLookupTable->SetNumberOfColors(256);
mLookupTable->SetRange(0.0, 255.0);
mLookupTable->Build();

QString nc[12];
nc[0]="Gray";
nc[1]="Rainbow";
nc[2]="Combined";
nc[3]="Hotmetal";
nc[4]="Log";
nc[5]="Mixed";
nc[6]="Red";
nc[7]="Green";
nc[8]="Blue";
//nc[9]="File";
nc[9]="Center Gray";
nc[10]="Color Steps";
nc[11]="Gray Steps";

int i;
setIconSize(QSize(100,15));
setFrame(true);
for(i=0; i<12; i++)
{
create(i);

QPixmap colorbar(85, 15);
colorbar.fill(Qt::white);
fillPixmap(&colorbar);
QIcon ic(colorbar);
insertItem(i,ic, nc[i] );

}
create(ctn);
updateLookupTable();
setCurrentIndex(ctn);
	   connect( this, SIGNAL( activated(int) ), this, SLOT( colorTableChanged(int) ) );
}
//////////////////////////////////////////////////////////////////////////
QColorTableWidget::~QColorTableWidget()
{
mLookupTable->Delete();
}
//////////////////////////////////////////////////////////////////////////
void QColorTableWidget::fillPixmap(QPixmap* pixmap)
{
QPainter brush(pixmap);

CRgb cc0;
CRgb cc1;
CRgb cc2;
int r,g,b;

for (int i=0; i<85; i++) 
{
cc0=mColorTable[i*3];
cc1=mColorTable[i*3+1];
cc2=mColorTable[i*3+2];

r= (cc0.r+cc1.r+cc2.r) / 3;
g= (cc0.g+cc1.g+cc2.g) / 3;
b= (cc0.b+cc1.b+cc2.b) / 3;

brush.setPen(QColor(r,g,b));
brush.drawLine(i, 1, i, 13);
}

}


//////////////////////////////////////////////////////////////////////////
void QColorTableWidget::create(int n)
{
//std::cout<<"ct="<<mType<<std::endl;
switch (n) {
  
    case 0    : gray(); 
                      break;
    case 1 : rainbow();
                      break;
    case 2 : combined();
                      break;
    case 3: hotmetal();
                      break;
    case 4: log();
                      break;
    case 5: mixed(); 
                       break;
   case 6: red(); 
                       break;
   case 7: green(); 
                       break;
   case 8: blue(); 
                       break;
	case 9: centerGray(); 
                       break;
	case 10: colorSteps(); 
                       break;
	case 11: graySteps(); 
                       break;

    default: gray(); 
  }

}
//////////////////////////////////////////////////////////////////////////
void QColorTableWidget::gray()
{
int i;
cbyte c;
   
for (i=0; i<256; i++) 
{
c=(cbyte)i;
mColorTable[i].r=mColorTable[i].g=mColorTable[i].b=c;
}

}
//////////////////////////////////////////////////////////////////////////
void QColorTableWidget::rainbow()
{
 int p=0,i,j,r,g,b;
        
 for(j=0;j<6;j++) 
 {
 
mColorTable[p].r=r=bitty[j].r;
mColorTable[p].g=g=bitty[j].g;
mColorTable[p].b=b=bitty[j].b;

p++;

for (i=1;i<bitty[j].n;i++) 
{
r+=bitty[j].dr; mColorTable[p].r=r;
g+=bitty[j].dg; mColorTable[p].g=g;
b+=bitty[j].db; mColorTable[p].b=b; 
p++;
} 
  
}

}
//////////////////////////////////////////////////////////////////////////
void QColorTableWidget::combined()
{
 int t=0,p=0,i,j,r,g,b;

  /* lower 128 = gray    levels */
  for (i=0; i<256; i+=2) {
     mColorTable[t].r=mColorTable[t].g=mColorTable[t].b=(cbyte)i; t+=1;
  }

  /* upper 128 = rainbow levels */
  for (j=0;j<6;j++) {
     r=bitty[j].r;
     g=bitty[j].g;
     b=bitty[j].b;
     if (p++ % 2 && p <= 256) {
       mColorTable[t].r =(cbyte)r;
      mColorTable[t].g=(cbyte)g;
       mColorTable[t].b=(cbyte)b;
       t+=1;
     }
     for (i=1;i<bitty[j].n;i++) {
        r+=bitty[j].dr;
        g+=bitty[j].dg;
        b+=bitty[j].db;
        if (p++ % 2 && p <= 256) {
          mColorTable[t].r=(cbyte)r;
          mColorTable[t].g=(cbyte)g;
         mColorTable[t].b=(cbyte)b;
          t+=1;
        }
     }
  }
	
}

//////////////////////////////////////////////////////////////////////////
void QColorTableWidget::hotmetal()
{
 int i, p=0;
  float intensity, delta_intensity;
 
  intensity = 0.0;
  delta_intensity = 1.0/182;

  for (i=0;i<182;i++) {               /* red */
     mColorTable[i].r=255*intensity;
     intensity+=delta_intensity;
  }
  
  for (i=182;i<256;i++) mColorTable[i].r=255;
  
  
  for (i=0; i<128;i++) mColorTable[i].g=0;   /* green */
 
  intensity = 0.0;
  delta_intensity = 1.0/91;

  for (i=128;i<219; i++) {
     mColorTable[i].g=255*intensity;
     intensity+=delta_intensity; 
  }


  for (i=219;i<256;i++)mColorTable[i].g=255;   

  for (i=0,p=2;i<192;i++) mColorTable[i].b=0;   /* blue */


  
  intensity=0.0;
  delta_intensity = 1.0/64;
  for (i=192;i<256;i++) {
     mColorTable[i].b=255*intensity;
     intensity += delta_intensity; 
  }
	
}
//////////////////////////////////////////////////////////////////////////
void QColorTableWidget::log()
{

for(int  i=0; i<256; i++)
{ 
mColorTable[i].r=(cbyte)255*log10(i+1.0)/log10(256.0);
mColorTable[i].g=(cbyte)255*exp(i*0.01)/exp(255.0*0.01);
mColorTable[i].b=(cbyte)0;

}

}
//////////////////////////////////////////////////////////////////////////
void QColorTableWidget::mixed()
{

int ii;
int i;
for( i=0; i<127; i++)
{
ii=mapRange(i,0,127,0,255);
mColorTable[i].r=mColorTable[i].g=mColorTable[i].b=(cbyte)ii;
}


for( i=128; i<256; i++)
{
ii=mapRange(i,128,255,0,255);
mColorTable[i].r=(cbyte)255*log10(ii+1.0)/log10(256.0);
mColorTable[i].g=(cbyte)255*exp(ii*0.01)/exp(255.0*0.01);
mColorTable[i].b=(cbyte)0;
}

}
//////////////////////////////////////////////////////////////////////////
void QColorTableWidget::red()
{
int i;
cbyte c;
   
for (i=0; i<256; i++) 
{
c=(cbyte)i;
mColorTable[i].r=c;
mColorTable[i].g=0;
mColorTable[i].b=0;
}

}

//////////////////////////////////////////////////////////////////////////
void QColorTableWidget::green()
{
int i;
cbyte c;
   
for (i=0; i<256; i++) 
{
c=(cbyte)i;
mColorTable[i].r=0;
mColorTable[i].g=c;
mColorTable[i].b=0;
}

}

//////////////////////////////////////////////////////////////////////////

void QColorTableWidget::blue()
{
int i;
cbyte c;
 

//spectra();
//return;



for (i=0; i<256; i++) 
{
c=(cbyte)i;
mColorTable[i].r=0;
mColorTable[i].g=0;
mColorTable[i].b=c;
}



}
//////////////////////////////////////////////////////////////////////////
void QColorTableWidget::centerGray()
{

int ii;
int i;

for( i=0; i<128; i++)
{
ii=mapRange(i,0,127,0,255);
mColorTable[i].r=mColorTable[i].g=mColorTable[i].b=(cbyte)ii;
mColorTable[255-i].r=mColorTable[255-i].g=mColorTable[255-i].b=(cbyte)ii;
}


}
//////////////////////////////////////////////////////////////////////////
void QColorTableWidget::colorSteps()
{

int ii;
int i;

CRgb c;

int count=0;
for( i=0; i<16; i++)
{
	c=cpat[i];
	for( ii=0; ii<16; ii++)
	{
	mColorTable[count]=c;
	++count;
	}
}



}
//////////////////////////////////////////////////////////////////////////
void QColorTableWidget::graySteps()
{

int ii;
int i;



int c;

int count=0;
for( i=0; i<16; i++)
{
	c=16*i;
	for( ii=0; ii<16; ii++)
	{
	mColorTable[count].r=mColorTable[count].g=mColorTable[count].b=(cbyte)c;
	++count;
	}
}

}
//////////////////////////////////////////////////////////////////////////
void QColorTableWidget::updateLookupTable()
{

CRgb cc;
for(int  i=0; i<256; i++)
{
cc=mColorTable[i];	
mLookupTable->SetTableValue(i,(double)cc.r/255.0, (double)cc.g/255.0,(double)cc.b/255.0,mOpacity);
}
	
}
//////////////////////////////////////////////////////////////////////////
void QColorTableWidget::colorTableChanged(int n)
{
create(n);
updateLookupTable();
this->LookupTableChange(this->GetLookupTable());
}