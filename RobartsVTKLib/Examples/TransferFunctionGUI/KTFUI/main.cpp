#include <QApplication>
#include "qTransferFunctionWindowWidget.h"

#include "vtkMetaImageWriter.h"
#include "vtkMetaImageReader.h"
#include "vtkCudaKohonenApplication.h"
#include "vtkCudaKSOMProbability.h"

//open the user interface
int main( int argc, char** argv ){

  //start application
  QApplication a(argc, argv);
  qTransferFunctionWindowWidget widget(0);
  widget.show();
  return a.exec();


}