/*
  Copyright(c) 2010 Johannes Jordan <johannes.jordan@cs.fau.de>.

  This file may be licensed under the terms of of the GNU General Public
  License, version 3, as published by the Free Software Foundation. You can
  find it here: http://www.gnu.org/licenses/gpl.html
*/

#ifndef SCALEDVIEW_H
#define SCALEDVIEW_H

#include <QOpenGLWidget>

class QScaledView : public QOpenGLWidget
{
  Q_OBJECT
public:
  QScaledView(QWidget *parent = 0);
  void resizeEvent(QResizeEvent * ev);
  virtual void paintEvent(QPaintEvent *ev);
  void mouseMoveEvent(QMouseEvent *ev);
  void mousePressEvent(QMouseEvent *ev);
  virtual void setPixmap(const QPixmap &pixmap);
  virtual void clearPixmap();

protected:
  virtual void cursorAction(QMouseEvent *ev, bool click = false);

  bool isDataSet;
  qreal scale;
  QTransform scaler, scalerI;
  QPixmap pixmap;
};

#endif // SCALEDVIEW_H
