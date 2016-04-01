#include <QtGui>
#include <QApplication>
#include "qTransferFunctionWindowWidget.h"

int main(int argc, char** argv)
{
  QApplication a(argc, argv);
  qTransferFunctionWindowWidget* w = new qTransferFunctionWindowWidget();
  w->show();
  return a.exec();
}