#ifndef RESIZABLEQVTKWIDGET_H
#define RESIZABLEQVTKWIDGET_H

#include <qwidget.h>
#include "QVTKWidget.h"

class ResizableQVTKWidget : public QVTKWidget {
public:
  ResizableQVTKWidget(QWidget* p = 0);
  bool ready;
private:
  void resizeEvent(QResizeEvent * event );
  void changeEvent ( QEvent * event );
  void paintEvent ( QPaintEvent * event );
};





#endif