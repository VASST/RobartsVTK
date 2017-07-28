#ifndef __QNUMBERACTION_H__
#define __QNUMBERACTION_H__

#include "qobject.h"
#include "qaction.h"

class QNumberAction : public QAction {
Q_OBJECT
public:
	int value;
	QNumberAction(const QString& string,QWidget* parent) : QAction(string,parent) {
		QObject::connect(this,SIGNAL(triggered()),this,SLOT(map()));
	};
signals:
	void triggered(int value);
public slots:
	void map(){ triggered(value); }
};

#endif //__QNUMBERACTION_H__