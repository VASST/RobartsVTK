#ifndef __SEGUEMAINWINDOW_H__
#define __SEGUEMAINWINDOW_H__

#include "qmainwindow.h"
#include "qobject.h"
#include "qboxlayout.h"

#include "QHierarchyWidget.h"
#include "QLabellingWidget.h"
#include "QSmoothnessScalarWidget.h"

#include "qcolortablewidget.h"

#include "vtkTree.h"
#include "vtkImageReader2.h"
#include <map>

class QShortcut;

class SEGUEMainWindow : public QMainWindow {
	Q_OBJECT

public:
	SEGUEMainWindow();
	~SEGUEMainWindow();

private slots:
	
	void SuggestPlanes();

	void ToggleGPU(int device);

	void UpdateSegmentation();

	void OpenImage();
	void SaveSegmentation();
	void SaveSeeding();
	void LoadSeeding();
	void SaveTree();
	void LoadTree();

	void BrushTypeChanged(QString& NewBrushType);
	void LabelRemoved(int RemovedLabel);
	void LabelAdded(int AddedLabel);
	void LabelSelected(int SelectedLabel);

	void SelectLabelFromKey();

	void ClosingLabellingWindow();

	void BrushSize(int);

private:
	
	int NumGPUs;
	bool* GPUUsed;

	std::map<QShortcut*,int> ShortcutToIdentifier;
	std::map<int,QShortcut*> IdentifierToShortcut;

	QHierarchyWidget* hWidget;
	QLabellingWidget* lWidget;
	QSmoothnessScalarWidget* smWidget;

	QColorTableWidget* ctWidget;
	QSlider* cutAlphaSlider;
	QSlider* labAlphaSlider;

	vtkImageReader2* Reader;

	vtkImageData* LeafSegmentation(int leaf, vtkImageData* cut);
	vtkImageData* BranchSegmentation(int branch, vtkImageData* cut);

	int SuggestedPlanes[3];

};

#endif //__SEGUEMAINWINDOW_H__
