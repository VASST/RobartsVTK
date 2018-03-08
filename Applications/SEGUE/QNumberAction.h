#ifndef __QNUMBERACTION_H__
#define __QNUMBERACTION_H__

#include <QObject>
#include <QAction>

class QNumberAction : public QAction {
Q_OBJECT
public:
	int value;
	QNumberAction(const QString& string,QWidget* parent) : QAction(string,parent) {
		QObject::connect(this,SIGNAL(triggered()),this,SLOT(map()));
	};
        virtual ~QNumberAction(){};
signals:
	void triggered(int value);
public slots:
	void map(){ triggered(value); }
};

#endif //__QNUMBERACTION_H__
