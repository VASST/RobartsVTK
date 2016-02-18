#include <QtGui>
#include <QApplication>
#include "FluoroPredViz.h"

int main(int argc, char** argv)
{
  QApplication a(argc, argv);
  FluoroPredViz* w = new FluoroPredViz();
  if( w->GetSuccessInit() )
  {
    w->show();
  }
  else
  {
    return 0;
  }
  return a.exec();
}