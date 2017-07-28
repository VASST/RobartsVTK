#ifndef __QSmoothnessScalarWidget_H__
#define __QSmoothnessScalarWidget_H__

#include "qwidget.h"
#include "qobject.h"
#include "qboxlayout.h"

#include "qslider.h"
#include "qlabel.h"

#include <map>

class QSmoothnessScalarWidget : public QWidget {
	Q_OBJECT

public:
	QSmoothnessScalarWidget(QWidget* parent = 0);
	~QSmoothnessScalarWidget();

	double GetCurrentSmoothness();
	double GetSmoothness(int label);

public slots:
	void SelectLabel(int Node);
	void RemoveLabel(int Node);
	void AddLabel(int Node);
	void SetSmoothness(int Node, double value);

signals:
	void SmoothnessChange(int label, double smoothness);

private:
	
	static const double MinValue;
	static const double MaxValue;
	static const double Increment;

	QVBoxLayout* Layout;
	QSlider* Slider;
	QLabel* Label;
	int CurrentLabel;

	std::map<int,double> Smoothness;

private slots:
	void SliderChanged(int value);

};

#endif //__QSmoothnessScalarWidget_H__