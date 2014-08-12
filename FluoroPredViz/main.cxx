#include <QtGui>
#include "FluoroPredViz.h"

int main(int argc, char** argv){

	QApplication a(argc, argv);
	FluoroPredViz* w = new FluoroPredViz();
	if(w->GetSuccessInit() == 0) w->show();
	else return 0;
	return a.exec();

}