#include <QtGui>
#include "qapplication.h"
#include "SEGUEMainWindow.h"
 
int main(int argc, char **argv)
{
    QApplication a(argc, argv);

	SEGUEMainWindow* main = new SEGUEMainWindow();
	//QWidget* main = new QLabellingWidget();
	main->show();
    return a.exec();
}