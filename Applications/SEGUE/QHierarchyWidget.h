#ifndef __QHIERARCHYWIDGET_H__
#define __QHIERARCHYWIDGET_H__

#include "qwidget.h"
#include "qobject.h"

#include <map>
#include <string>

#include "QTreeView.h"
#include "QStandardItemModel.h"
#include "QBoxLayout.h"
#include "qtcolortriangle.h"

#include "vtkTree.h"
#include "vtkMutableDirectedGraph.h"
#include "vtkStringArray.h"
#include "vtkDoubleArray.h"

class QHierarchyWidget : public QWidget {
	Q_OBJECT

public:
	QHierarchyWidget(QWidget* parent = 0);
	void Initialize();
	~QHierarchyWidget();

	void SetHierarchy(vtkTree* hierarchy, std::map<vtkIdType,int>* map1, std::map<int,vtkIdType>* map2);
	void GetHierarchy(vtkTree* retVal, std::map<vtkIdType,int>* map1, std::map<int,vtkIdType>* map2);
	
	int GetCurrentItem();
	int GetParentItem(int node);

	QColor GetColour(int node);
	std::string GetName(int node);
	bool IsLeaf(int node);
	bool IsBranch(int node);

	std::vector<int>* GetChildren(int node);

public slots:
	void ForceSelectLabel(int Node);
	void ForceColourReiterate();

signals:
	void AddLabel(int Node);
	void RemoveLabel(int Node);
	void RecolourLabel(int Node, int r, int g, int b);
	void ClearLabel(int Node);
	void SelectLabel(int Node);

	void SwapLabels(int Node1, int Node2);

private:
	bool Initialized;

    QTreeView* HierarchyView;
	QStandardItemModel* ItemModel;
	QtColorTriangle* ColourTriangle;

	QVBoxLayout* Layout;
	
	void RemoveNodeRecursive(QStandardItem* Node);
	void ClearNodeRecursive(QStandardItem* Node);
	void EmitMapSignal(QStandardItem* OldNode, QStandardItem* NewNode);
	void EmitColourSignal(QStandardItem* Node);
	void AddToHierarchy(vtkIdType Node, QStandardItem* Item, vtkMutableDirectedGraph* Builder,
		std::map<vtkIdType,int>* map1, std::map<int,vtkIdType>* map2, vtkStringArray* Names,
		vtkDoubleArray* colour[3] );


	bool IsBranch(QStandardItem* Node);

	std::map<QStandardItem*,int> ItemToIdentifier;
	std::map<int,QStandardItem*> IdentifierToItem;
	std::vector<int> UnusedIdentifiers;
	
	QStandardItem* GetCurrentNode();

	bool InsertingItem;
	bool RemovingItem;
	QModelIndex NewParentDrop;
	int			NewStartDrop;

private slots:

	void PropogateSelection();
	void AddLeafToHierarchy();
	void RemoveNodeFromHierarchy();
	void StartColourDialog();
	void ClearSamplePoints();
	
	void ColourChange(const QColor& colour);

	void MapLabel(QStandardItem* OldItem, QStandardItem* NewItem);

	void ReorderSelectionRemove( const QModelIndex & parent, int start, int end );
	void ReorderSelectionInsert( const QModelIndex & parent, int start, int end );

	void AppendToHierarchy(vtkTree* hierarchy, vtkIdType Node, QStandardItem* Item,
							std::map<vtkIdType,int>* map1, std::map<int,vtkIdType>* map2);
	void AppendToHierarchy(vtkTree* hierarchy, vtkIdType Node, QStandardItemModel* Item,
							std::map<vtkIdType,int>* map1, std::map<int,vtkIdType>* map2);

};

#endif //__QHIERARCHYWIDGET_H__