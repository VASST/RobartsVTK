#ifndef __QLABELLINGWIDGET_H__
#define __QLABELLINGWIDGET_H__

#include "qwidget.h"
#include "qobject.h"

#include <set>

#include "QTreeView.h"
#include "QStandardItemModel.h"
#include "QBoxLayout.h"

#include "vtkImageReader2.h"

#include "SliceViewer.h"
#include "QVTKWidget.h"

#include "vtkRenderer.h"
#include "vtkRenderWindow.h"
#include "vtkImageData.h"

class QLabellingWidget : public QWidget {
	Q_OBJECT

public:
	QLabellingWidget(vtkImageReader2* file, vtkLookupTable* CT, QWidget* parent = 0);
	~QLabellingWidget();

	vtkImageData* GetSeeding();
	void SetCut(vtkImageData* newCut );
	vtkImageData* GetCut();

public slots:
	void ChangeToLabel(int);
	void ClearSamplePoints(int);
	void SwapLabels(int Node1, int Node2);
	void UpdateColourTable(vtkLookupTable*);
	void SetColour(int Label, int r, int g, int b);
	void Update();
	void SetLabelOpacity(int percentage);
	void SetCutOpacity(int percentage);
	void MoveToPlanes(int x, int y, int z);
	void SetBrushSize(int pixelRadius);

	void Update3DRendering();
	void BrushSize(int);

signals:

private:
	int CurrentLabel;

	double LabelOpacity;
	double CutOpacity;

	vtkImageReader2* Reader;
	vtkImageData* mInput;
	vtkImageData* mBin;
	vtkImageData* mCut;
	
	vtkLookupTable* BinaryLookupTable;
	vtkLookupTable* CutLookupTable;

	SliceViewer*		ImagePlane[3];
	QVTKWidget*			MeshedView;
	vtkRenderer*		MeshedRenderer;

private slots:

};

#endif //__QLABELLINGWIDGET_H__