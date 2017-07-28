#ifndef _Q_COLOR_TABLE_WIDGET_H_
#define _Q_COLOR_TABLE_WIDGET_H_



#include <QComboBox>
#include <vector>

class QPixmap;
class vtkLookupTable;
//////////////////////////////////////////////////////////////////////////
typedef struct _CRGB{
 unsigned char r, g, b;
} CRgb;



//////////////////////////////////////////////////////////////////////////
class QColorTableWidget : public QComboBox
{
	Q_OBJECT
public:
	QColorTableWidget(int ctn=0,QWidget *parent=0);
	~QColorTableWidget();
	vtkLookupTable* GetLookupTable(){return this->mLookupTable;};
	void SetOpacity(double a){this->mOpacity=a;};
	void updateLookupTable();
protected:
	void fillPixmap(QPixmap* pixmap);
	void create(int);
	void gray();
	void rainbow();
	void combined();
	void hotmetal();
	void log();
	void mixed();
	void red();
	void green();
	void blue();
	void centerGray();
	void colorSteps();
	void graySteps();

signals:
	void LookupTableChange(vtkLookupTable*);

protected:
	std::vector<CRgb> mColorTable;
	vtkLookupTable* mLookupTable;
	double mOpacity;
protected slots:
  	void colorTableChanged(int);
}; 
















#endif
