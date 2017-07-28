#include "QHierarchyWidget.h"

#include "qheaderview.h"
#include "qabstractitemmodel.h"
#include "qpushbutton.h"
#include "qaction.h"
#include "qcolordialog.h"
#include "qcolor.h"
#include "qevent.h"
#include "qmimedata.h"

#include "vtkMutableDirectedGraph.h"
#include "vtkType.h"

#include <iostream>
#include <algorithm>

#include "vtkStringArray.h"
#include "vtkDataSetAttributes.h"

QHierarchyWidget::QHierarchyWidget(QWidget* parent)
	: QWidget(parent), InsertingItem(false), RemovingItem(false), Initialized(false)
{
	
	this->Layout = new QVBoxLayout(this);
	this->setLayout(this->Layout);

	//create the item model in the hierarchy
	ItemModel = new QStandardItemModel(1, 1);
    
	//Create the hierarchical view
	this->HierarchyView = new QTreeView(0);
	this->HierarchyView->setModel(ItemModel);
	this->HierarchyView->header()->close();
	this->Layout->addWidget(this->HierarchyView);
	this->HierarchyView->setDragEnabled(true);
	this->HierarchyView->setAcceptDrops(true);
	this->HierarchyView->setDragDropMode(QAbstractItemView::InternalMove);
	this->HierarchyView->setDropIndicatorShown(true);
	this->HierarchyView->expandAll();
	QObject::connect(this->HierarchyView->selectionModel(), SIGNAL(selectionChanged(const QItemSelection &, const QItemSelection &)), this, SLOT(PropogateSelection()));
	QObject::connect(this->ItemModel, SIGNAL(rowsAboutToBeRemoved ( const QModelIndex &, int, int )),
		this, SLOT(ReorderSelectionRemove( const QModelIndex &, int, int )));
	QObject::connect(this->ItemModel, SIGNAL(rowsInserted ( const QModelIndex &,  int, int )),
		this, SLOT(ReorderSelectionInsert( const QModelIndex &, int, int )));

	
	//set up colour triangle subwidget
	ColourTriangle = new QtColorTriangle();
	this->ColourTriangle->setColor(Qt::blue);
	QObject::connect(ColourTriangle,SIGNAL(colorChanged(const QColor &)),this,SLOT(ColourChange(const QColor &)));
	this->Layout->addWidget(this->ColourTriangle);
	
	//create right click menu
	this->HierarchyView->setContextMenuPolicy(Qt::ActionsContextMenu);
	QAction* NewItemAction = new QAction(tr("Add Label"), this->HierarchyView);
	this->HierarchyView->addAction(NewItemAction);
	QObject::connect(NewItemAction,SIGNAL(triggered()),this,SLOT(AddLeafToHierarchy()));
	QAction* RemoveItemAction = new QAction(tr("Remove Label"), this->HierarchyView);
	this->HierarchyView->addAction(RemoveItemAction);
	QObject::connect(RemoveItemAction,SIGNAL(triggered()),this,SLOT(RemoveNodeFromHierarchy()));
	QAction* ColourItemAction = new QAction(tr("Colour"), this->HierarchyView);
	this->HierarchyView->addAction(ColourItemAction);
	QObject::connect(ColourItemAction,SIGNAL(triggered()),this,SLOT(StartColourDialog()));
	QAction* ClearItemAction = new QAction(tr("Clear Points"), this->HierarchyView);
	this->HierarchyView->addAction(ClearItemAction);
	QObject::connect(ClearItemAction,SIGNAL(triggered()),this,SLOT(ClearSamplePoints()));

	//add redundant buttons for the menu actions buttons
	QPushButton* NewItemButton = new QPushButton("New Item",this);
	this->Layout->addWidget(NewItemButton);
	QObject::connect(NewItemButton,SIGNAL(pressed()),this,SLOT(AddLeafToHierarchy()));
	QPushButton* RemoveItemButton = new QPushButton("Remove Item",this);
	this->Layout->addWidget(RemoveItemButton);
	QObject::connect(RemoveItemButton,SIGNAL(pressed()),this,SLOT(RemoveNodeFromHierarchy()));

}

void QHierarchyWidget::Initialize(){
	
	//don't allow reinitialization
	if( Initialized ) return;
	Initialized = true;

	//create background object
	QStandardItem* backItem = new QStandardItem();
	QPixmap pix(16, 16);
    pix.fill(Qt::blue);
    backItem->setIcon(pix);
    backItem->setText("Background");
    ItemModel->setItem(0, 0, backItem);
	int backIdentifier = 1;
	this->ItemToIdentifier[backItem] = backIdentifier;
	this->IdentifierToItem[backIdentifier] = backItem;
	AddLabel(backIdentifier);
	this->HierarchyView->setCurrentIndex(backItem->index());

	//create foreground object
	QStandardItem* foreItem = new QStandardItem();
    pix.fill(Qt::red);
    foreItem->setIcon(pix);
    foreItem->setText("Foreground");
    ItemModel->setItem(1, 0, foreItem);
	int foreIdentifier = 2;
	this->ItemToIdentifier[foreItem] = foreIdentifier;
	this->IdentifierToItem[foreIdentifier] = foreItem;
	AddLabel(foreIdentifier);
}

QHierarchyWidget::~QHierarchyWidget(){
	delete this->ItemModel;
	delete this->HierarchyView;
	delete this->Layout;
}

void QHierarchyWidget::ForceSelectLabel(int Node){
	if( this->IdentifierToItem.find(Node) == this->IdentifierToItem.end() ) return;
	this->HierarchyView->setCurrentIndex(this->IdentifierToItem[Node]->index());
}

void QHierarchyWidget::AddLeafToHierarchy(){

	//find currently selected index
	QStandardItem* selectedItem = this->GetCurrentNode();
	if( selectedItem == 0 ) return;

	//find if parent was a leaf node
	bool moveLabels = !this->IsBranch(selectedItem);

	//find appropriate colour (parent's)
	QColor oldColour = QColor::fromRgb( selectedItem->icon().pixmap(16,16).toImage().bits()[2],
										selectedItem->icon().pixmap(16,16).toImage().bits()[1],
										selectedItem->icon().pixmap(16,16).toImage().bits()[0] );
	
	//create new child item
	QStandardItem* newItem = new QStandardItem();
    QPixmap pix(16, 16);
    pix.fill(oldColour);
    newItem->setIcon(pix);
    newItem->setText("New Item");
	this->InsertingItem = true;
	selectedItem->appendRow(newItem);

	//put in mapping
	int NewIdentifier = 0;
	if( this->UnusedIdentifiers.size() == 0 ){
		NewIdentifier = (int) this->ItemToIdentifier.size() + 1;
	}else{
		NewIdentifier = this->UnusedIdentifiers.back();
		this->UnusedIdentifiers.pop_back();
	}
	this->ItemToIdentifier[newItem] = NewIdentifier;
	this->IdentifierToItem[NewIdentifier] = newItem;

	//inform interface
	this->AddLabel(NewIdentifier);
	if( moveLabels ) this->SwapLabels(NewIdentifier,this->ItemToIdentifier[selectedItem]);

	//expand parent
	this->HierarchyView->expand(selectedItem->index());

}

void QHierarchyWidget::RemoveNodeRecursive(QStandardItem* Node){

	//call recursively on all children
	int NumChildren = Node->rowCount();
	for(int i = 0; i < NumChildren; i++)
		this->RemoveNodeRecursive(Node->child(i));

	//release remove signal
	int Identifier = this->ItemToIdentifier[Node];
	this->RemoveLabel(Identifier);
	this->ItemToIdentifier.erase(this->ItemToIdentifier.find(Node));
	this->IdentifierToItem.erase(this->IdentifierToItem.find(Identifier));
	this->UnusedIdentifiers.push_back(Identifier);

}

void QHierarchyWidget::RemoveNodeFromHierarchy(){

	//find currently selected index (do nothing if this is empty or the source node)
	QStandardItem* selectedItem = this->GetCurrentNode();
	if( selectedItem == 0 ) return;
	QStandardItem* parent = selectedItem->parent();
	QStandardItem* nextInLine = parent;
	if(parent && selectedItem->row() != 0){
		QStandardItem* biggerSibling = parent->child(selectedItem->row()-1);
		nextInLine = biggerSibling;
	}else if(parent && selectedItem->row() == 0){
		nextInLine = parent;
	}else if( selectedItem->row() != 0 ){
		QStandardItem* biggerSibling = this->ItemModel->item(selectedItem->row()-1,0);
		nextInLine = biggerSibling;
	}else if(this->ItemModel->rowCount() > 1){
		QStandardItem* biggerSibling = this->ItemModel->item(1,0);
		nextInLine = biggerSibling;
	}else{
		return; //cannot remove entire hierarchy
	}
	this->RemovingItem = true;

	//remove nodes recursively
	this->RemoveNodeRecursive(selectedItem);
	if( parent )
		parent->removeRow(selectedItem->row());
	else
		this->ItemModel->removeRow(selectedItem->row());

	//update model
	this->HierarchyView->setModel(this->ItemModel);
	this->HierarchyView->setCurrentIndex(nextInLine->index());

	//resort list of unused identifiers
	std::sort( this->UnusedIdentifiers.begin(), this->UnusedIdentifiers.end() );
	std::reverse( this->UnusedIdentifiers.begin(), this->UnusedIdentifiers.end() );

}

void QHierarchyWidget::AddToHierarchy(vtkIdType Node, QStandardItem* Item, vtkMutableDirectedGraph* Builder,
		std::map<vtkIdType,int>* map1, std::map<int,vtkIdType>* map2, vtkStringArray* Names,
		vtkDoubleArray* Colours[3] ){
	
	//add portion to map
	(*map1)[Node] = this->ItemToIdentifier[Item];
	(*map2)[this->ItemToIdentifier[Item]] = Node;

	//call recursively on all children
	int NumChildren = Item->rowCount();
	for(int i = 0; i < NumChildren; i++){
		
		QStandardItem* childItem = Item->child(i);
		QColor oldColour = QColor::fromRgb( childItem->icon().pixmap(16,16).toImage().bits()[2],
											childItem->icon().pixmap(16,16).toImage().bits()[1],
											childItem->icon().pixmap(16,16).toImage().bits()[0] );

		vtkIdType childNode = Builder->AddVertex();
		Builder->AddEdge(Node,childNode);
		Names->InsertValue(childNode,childItem->text().toStdString());
		Colours[0]->InsertValue( childNode, (double) oldColour.red() / 255.0 );
		Colours[1]->InsertValue( childNode, (double) oldColour.green() / 255.0 );
		Colours[2]->InsertValue( childNode, (double) oldColour.blue() / 255.0 );
		AddToHierarchy(childNode, childItem, Builder, map1, map2, Names, Colours);
	}
}

void QHierarchyWidget::GetHierarchy(vtkTree* retVal, std::map<vtkIdType,int>* map1, std::map<int,vtkIdType>* map2){
	//create a tree builder
	vtkMutableDirectedGraph* Builder = vtkMutableDirectedGraph::New();
	vtkStringArray* Names = vtkStringArray::New();
	Names->SetName("Names");
	vtkDoubleArray* Colours[3] = {vtkDoubleArray::New(), vtkDoubleArray::New(), vtkDoubleArray::New()};
	Colours[0]->SetName("Colour-Red");
	Colours[1]->SetName("Colour-Green");
	Colours[2]->SetName("Colour-Blue");

	//recursively build the tree
	vtkIdType Source = Builder->AddVertex();
	Names->InsertValue(Source,"Source - Root");
	Colours[0]->InsertValue(Source,0.0);
	Colours[1]->InsertValue(Source,0.0);
	Colours[2]->InsertValue(Source,0.0);
	(*map1)[Source] = 0;
	(*map2)[0] = Source;
	for(int i = 0; i < this->ItemModel->rowCount(); i++ ){
		
		QStandardItem* childItem = this->ItemModel->item(i,0);
		QColor oldColour = QColor::fromRgb( childItem->icon().pixmap(16,16).toImage().bits()[2],
											childItem->icon().pixmap(16,16).toImage().bits()[1],
											childItem->icon().pixmap(16,16).toImage().bits()[0] );

		vtkIdType childNode = Builder->AddVertex();
		Builder->AddEdge(Source,childNode);
		Names->InsertValue(childNode,this->ItemModel->item(i,0)->text().toStdString());
		Colours[0]->InsertValue( childNode, (double) oldColour.red() / 255.0 );
		Colours[1]->InsertValue( childNode, (double) oldColour.green() / 255.0 );
		Colours[2]->InsertValue( childNode, (double) oldColour.blue() / 255.0 );
		AddToHierarchy(childNode, childItem, Builder, map1, map2, Names, Colours);
	}

	//copy it into the correct form and return
	Builder->GetVertexData()->AddArray(Names);
	Builder->GetVertexData()->AddArray(Colours[0]);
	Builder->GetVertexData()->AddArray(Colours[1]);
	Builder->GetVertexData()->AddArray(Colours[2]);
	retVal->CheckedDeepCopy(Builder);
	Builder->Delete();
	Names->Delete();
	Colours[0]->Delete();
	Colours[1]->Delete();
	Colours[2]->Delete();
}

int QHierarchyWidget::GetCurrentItem(){
	QModelIndex selectedIndex = this->HierarchyView->selectionModel()->currentIndex();
	QStandardItem* selectedItem = this->ItemModel->itemFromIndex(selectedIndex);
	if( this->ItemToIdentifier.find(selectedItem) == this->ItemToIdentifier.end() ) return 0;
	return this->ItemToIdentifier[selectedItem];
}

QStandardItem* QHierarchyWidget::GetCurrentNode(){
	QModelIndex selectedIndex = this->HierarchyView->selectionModel()->currentIndex();
	QStandardItem* selectedItem = this->ItemModel->itemFromIndex(selectedIndex);
	return selectedItem;

}

void QHierarchyWidget::StartColourDialog(){
	QStandardItem* selectedItem = this->GetCurrentNode();
	if( selectedItem == 0 ) return;

	//get a colour, returning if not valid (ie: cancelled dialog)
	QColor oldColour = QColor::fromRgb( selectedItem->icon().pixmap(16,16).toImage().bits()[2],
										selectedItem->icon().pixmap(16,16).toImage().bits()[1],
										selectedItem->icon().pixmap(16,16).toImage().bits()[0] );
	QColor colour = QColorDialog::getColor(oldColour, this, "Text Color",  QColorDialog::DontUseNativeDialog);
	if( !colour.isValid() ) return;

	//apply colour to location in hierarchy
    QPixmap pix(16, 16);
    pix.fill(colour);
    selectedItem->setIcon(pix);

	//update colour triangle (will send signal to rest of interface)
	this->ColourTriangle->setColor(colour);
	//RecolourLabel(this->GetCurrentItem(),colour.red(),colour.green(), colour.blue());
}

void QHierarchyWidget::ClearNodeRecursive(QStandardItem* Node){
	
	//call recursively on all children
	int NumChildren = Node->rowCount();
	for(int i = 0; i < NumChildren; i++)
		this->ClearNodeRecursive(Node->child(i));

	//release remove signal
	this->ClearLabel(this->ItemToIdentifier[Node]);

}

void QHierarchyWidget::ClearSamplePoints(){
	QStandardItem* selectedItem = this->GetCurrentNode();
	if( selectedItem == 0 ) return;

	//send off the clear signals
	ClearNodeRecursive(selectedItem);
}


void QHierarchyWidget::PropogateSelection(){
	int selectedItem = this->GetCurrentItem();
	if( selectedItem == 0 ) return;
	this->ColourTriangle->setColor(this->GetColour(selectedItem));
	this->SelectLabel(selectedItem);
}

void QHierarchyWidget::EmitMapSignal(QStandardItem* OldNode, QStandardItem* NewNode){
	//call recursively on all children
	int NumChildren = OldNode->rowCount();
	for(int i = 0; i < NumChildren; i++)
		this->EmitMapSignal(OldNode->child(i),NewNode->child(i));
	MapLabel(OldNode, NewNode);
}

void QHierarchyWidget::ReorderSelectionRemove( const QModelIndex & parent, int start, int end ){
	if( !this->RemovingItem ){
		
		QStandardItem* NewLocation = this->ItemModel->itemFromIndex(this->NewParentDrop.child(this->NewStartDrop,0));
		if( NewLocation == 0 ) NewLocation = this->ItemModel->item(this->NewStartDrop,0);
		
		QStandardItem* OldLocation = this->ItemModel->itemFromIndex(parent.child(start,0));
		if( OldLocation == 0 ) OldLocation = this->ItemModel->item(start,0);

		this->EmitMapSignal(OldLocation,NewLocation);
	}
	this->RemovingItem = false;
}

void QHierarchyWidget::ReorderSelectionInsert( const QModelIndex & parent, int start, int end ){
	if( !InsertingItem ){
		this->NewParentDrop = parent;
		this->NewStartDrop = start;
	}
	this->InsertingItem = false;
}

QColor QHierarchyWidget::GetColour(int node){
	if( this->IdentifierToItem.find(node) == this->IdentifierToItem.end() ) return QColor(0,0,0,0);
	QStandardItem* item = this->IdentifierToItem[node];
	return QColor::fromRgb( item->icon().pixmap(16,16).toImage().bits()[2],
							item->icon().pixmap(16,16).toImage().bits()[1],
							item->icon().pixmap(16,16).toImage().bits()[0] );
}

std::string QHierarchyWidget::GetName(int node){
	if( this->IdentifierToItem.find(node) == this->IdentifierToItem.end() ) return "";
	QStandardItem* item = this->IdentifierToItem[node];
	return item->text().toStdString();
}

void QHierarchyWidget::EmitColourSignal(QStandardItem* Node){
	//call recursively on all children
	int NumChildren = Node->rowCount();
	for(int i = 0; i < NumChildren; i++)
		this->EmitColourSignal(Node->child(i));
	QColor colour = QColor::fromRgb( Node->icon().pixmap(16,16).toImage().bits()[2],
									 Node->icon().pixmap(16,16).toImage().bits()[1],
									 Node->icon().pixmap(16,16).toImage().bits()[0] );
	RecolourLabel(this->ItemToIdentifier[Node],colour.red(),colour.green(),colour.blue());
}

void QHierarchyWidget::ForceColourReiterate(){
	for(int i = 0; i < this->ItemModel->rowCount(); i++)
		EmitColourSignal( this->ItemModel->item(i) );
}

void QHierarchyWidget::MapLabel(QStandardItem* OldLabel, QStandardItem* NewLabel){
	int Identifier = this->ItemToIdentifier[OldLabel];
	this->ItemToIdentifier.erase(this->ItemToIdentifier.find(OldLabel));
	this->ItemToIdentifier[NewLabel] = Identifier;
	this->IdentifierToItem[Identifier] = NewLabel;

	std::cout << "Relabelling:\t" << Identifier << "\t" << OldLabel << " : " << OldLabel->text().toStdString() << std::endl;
	std::cout << "         to:\t" << Identifier << "\t" << NewLabel << " : " << NewLabel->text().toStdString() << std::endl;
}

void QHierarchyWidget::ColourChange(const QColor& colour){
	QStandardItem* selectedItem = this->GetCurrentNode();
	if( selectedItem == 0 ) return;

	QPixmap pix(16, 16);
    pix.fill(colour);
    selectedItem->setIcon(pix);
	
	RecolourLabel(this->ItemToIdentifier[selectedItem],colour.red(),colour.green(),colour.blue());
}

bool QHierarchyWidget::IsLeaf(int node){
	if( this->IdentifierToItem.find(node) == this->IdentifierToItem.end() ) return false;
	return this->IdentifierToItem[node]->rowCount() == 0;
}

bool QHierarchyWidget::IsBranch(int node){
	if( this->IdentifierToItem.find(node) == this->IdentifierToItem.end() ) return false;
	return IsBranch(this->IdentifierToItem[node]);
}

bool QHierarchyWidget::IsBranch(QStandardItem* Node){
	return Node->rowCount() != 0;
}

int QHierarchyWidget::GetParentItem(int node){
	if( this->IdentifierToItem.find(node) == this->IdentifierToItem.end() ) return -1;
	if( this->IdentifierToItem[node]->parent() )
		return this->ItemToIdentifier[ this->IdentifierToItem[node]->parent() ];
	else
		return 0;
}

std::vector<int>* QHierarchyWidget::GetChildren(int node){
	std::vector<int>* retVal = new std::vector<int>();
	
	if( this->IsLeaf(node) ){
		retVal->push_back(node);
	}else{
		std::vector<QStandardItem*> branches;
		branches.push_back(this->IdentifierToItem[node]);
		while( branches.size() > 0 ){
			QStandardItem* currBranch = branches.back();
			branches.pop_back();
			
			int NumChildren = currBranch->rowCount();
			for(int i = 0; i < NumChildren; i++){
				QStandardItem* child = currBranch->child(i);
				if( child->rowCount() )
					branches.push_back(child);
				else
					retVal->push_back(this->ItemToIdentifier[child]);
			}
		}
	}

	return retVal;
}


void QHierarchyWidget::AppendToHierarchy(vtkTree* hierarchy, vtkIdType Node, QStandardItemModel* Item, std::map<vtkIdType,int>* map1, std::map<int,vtkIdType>* map2){

	for(int i = 0; i < hierarchy->GetNumberOfChildren(Node); i++){
		
		vtkIdType childNode = hierarchy->GetChild(Node,i);
		vtkStdString name = hierarchy->GetVertexData()->GetAbstractArray("Names")->GetVariantValue(childNode).ToString();
		double colour[3] = { hierarchy->GetVertexData()->GetAbstractArray("Colour-Red")->GetVariantValue(childNode).ToDouble(),
							 hierarchy->GetVertexData()->GetAbstractArray("Colour-Green")->GetVariantValue(childNode).ToDouble(),
							 hierarchy->GetVertexData()->GetAbstractArray("Colour-Blue")->GetVariantValue(childNode).ToDouble() };

		QStandardItem* backItem = new QStandardItem();
		QPixmap pix(16, 16);
		pix.fill( QColor( (int)(colour[0]*255.0), (int)(colour[1]*255.0), (int)(colour[2]*255.0) ) ); 
		backItem->setIcon(pix);
		backItem->setText(name.c_str());
		ItemModel->setItem(i, 0, backItem);
		int backIdentifier = IdentifierToItem.size() + 1;
		this->ItemToIdentifier[backItem] = backIdentifier;
		this->IdentifierToItem[backIdentifier] = backItem;
		AddLabel(backIdentifier);
		
		(*map1)[childNode] = backIdentifier;
		(*map2)[backIdentifier] = childNode;

		AppendToHierarchy(hierarchy,childNode,backItem,map1,map2);
	}

}

void QHierarchyWidget::AppendToHierarchy(vtkTree* hierarchy, vtkIdType Node, QStandardItem* Item,
										std::map<vtkIdType,int>* map1, std::map<int,vtkIdType>* map2){

	for(int i = 0; i < hierarchy->GetNumberOfChildren(Node); i++){
		
		vtkIdType childNode = hierarchy->GetChild(Node,i);
		vtkStdString name = hierarchy->GetVertexData()->GetAbstractArray("Names")->GetVariantValue(childNode).ToString();
		double colour[3] = { hierarchy->GetVertexData()->GetAbstractArray("Colour-Red")->GetVariantValue(childNode).ToDouble(),
							 hierarchy->GetVertexData()->GetAbstractArray("Colour-Green")->GetVariantValue(childNode).ToDouble(),
							 hierarchy->GetVertexData()->GetAbstractArray("Colour-Blue")->GetVariantValue(childNode).ToDouble() };

		QStandardItem* newItem = new QStandardItem();
		QPixmap pix(16, 16);
		pix.fill( QColor( (int)(colour[0]*255.0), (int)(colour[1]*255.0), (int)(colour[2]*255.0) ) ); 
		newItem->setIcon(pix);
		newItem->setText(name.c_str());
		this->InsertingItem = true;
		Item->appendRow(newItem);
		
		//put in mapping
		int NewIdentifier = 0;
		if( this->UnusedIdentifiers.size() == 0 ){
			NewIdentifier = (int) this->ItemToIdentifier.size() + 1;
		}else{
			NewIdentifier = this->UnusedIdentifiers.back();
			this->UnusedIdentifiers.pop_back();
		}
		this->ItemToIdentifier[newItem] = NewIdentifier;
		this->IdentifierToItem[NewIdentifier] = newItem;
		AddLabel(NewIdentifier);
		
		(*map1)[childNode] = NewIdentifier;
		(*map2)[NewIdentifier] = childNode;

		AppendToHierarchy(hierarchy,childNode,newItem,map1,map2);
	}

}

void QHierarchyWidget::SetHierarchy(vtkTree* hierarchy, std::map<vtkIdType,int>* map1, std::map<int,vtkIdType>* map2){
	if( !hierarchy || hierarchy->GetNumberOfVertices() < 2 ) return;

	//remove all items from the hierarchy
	int top = this->ItemModel->rowCount();
	for(int x = top-1; x >= 0; x--)
		RemoveNodeRecursive(this->ItemModel->item(x));
	UnusedIdentifiers.clear();
	
	(*map1)[hierarchy->GetRoot()] = 0;
	(*map2)[0] = hierarchy->GetRoot();

	AppendToHierarchy(hierarchy,hierarchy->GetRoot(),this->ItemModel,map1,map2);

	this->HierarchyView->setCurrentIndex(this->ItemModel->item(0)->index());

}