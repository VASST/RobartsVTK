#include <QApplication>
#include "DUALTF_transferFunctionWindowWidget.h"

//open the user interface
int main( int argc, char** argv ){
  QApplication a(argc, argv);
  DUALTF_transferFunctionWindowWidget widget(0);
  widget.show();
  return a.exec();
}