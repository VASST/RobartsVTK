#include "QSmoothnessScalarWidget.h"

#include <sstream>

const double QSmoothnessScalarWidget::MinValue = 0.0;
const double QSmoothnessScalarWidget::MaxValue = 1.0;
const double QSmoothnessScalarWidget::Increment = 0.001;

QSmoothnessScalarWidget::QSmoothnessScalarWidget(QWidget* parent)
	: QWidget(parent)
{

	this->Layout = new QVBoxLayout();
	this->setLayout(this->Layout);

	this->Slider = new QSlider(Qt::Orientation::Horizontal);
	this->Slider->setMinimum(0);
	this->Slider->setMaximum( (int) (( this->MaxValue - this->MinValue) / this->Increment) );
	this->Slider->setValue(0);
	QObject::connect(this->Slider,SIGNAL(valueChanged(int)),this,SLOT(SliderChanged(int)));

	this->Label = new QLabel("Smoothness: 0");
	this->Label->setAlignment(Qt::AlignLeft);
	
	this->Layout->addWidget(this->Slider);
	this->Layout->addWidget(this->Label);

	this->CurrentLabel = 0;
}

QSmoothnessScalarWidget::~QSmoothnessScalarWidget(){
	delete this->Label;
	delete this->Slider;
}

void QSmoothnessScalarWidget::SelectLabel(int Node){
	if( this->Smoothness.find(Node) == this->Smoothness.end() )
		this->Smoothness[Node] = 0.1;
	this->CurrentLabel = Node;
	this->Slider->setValue( (int) ((this->Smoothness[Node] - this->MinValue) / this->Increment) );
}

void QSmoothnessScalarWidget::RemoveLabel(int Node){
	if( this->Smoothness.find(Node) == this->Smoothness.end() ) return;
	this->Smoothness.erase( this->Smoothness.find(Node) );
}

void QSmoothnessScalarWidget::AddLabel(int Node){
	if( this->Smoothness.find(Node) != this->Smoothness.end() ) return;
	this->Smoothness[Node] = 0.1;
}

void QSmoothnessScalarWidget::SetSmoothness(int Node, double value){
	if( this->Smoothness.find(Node) == this->Smoothness.end() ) return;
	this->Smoothness[Node] = value;
}

void QSmoothnessScalarWidget::SliderChanged(int value){
	double SmoothnessValue = (double) value * this->Increment + this->MinValue;

	std::ostringstream NewText;
	NewText << "Smoothness: " << SmoothnessValue;

	this->SetSmoothness(this->CurrentLabel, SmoothnessValue);

	this->Label->setText(QString::fromStdString(NewText.str()));
	this->SmoothnessChange(this->CurrentLabel,SmoothnessValue);
}

double QSmoothnessScalarWidget::GetCurrentSmoothness(){
	return this->GetSmoothness( this->CurrentLabel );
}

double QSmoothnessScalarWidget::GetSmoothness(int label){
	if( this->Smoothness.find(label) ==  this->Smoothness.end() ) return -1.0;
	return this->Smoothness[label];
}