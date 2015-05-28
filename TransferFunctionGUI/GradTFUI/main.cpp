#include <QApplication>
#include "transferFunctionWindowWidget.h"

//open the user interface
int main( int argc, char** argv ){
  QApplication a(argc, argv);
  transferFunctionWindowWidget widget(0);
  widget.show();
  return a.exec();
}