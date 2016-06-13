/*
  Copyright(c) 2010 Johannes Jordan <johannes.jordan@cs.fau.de>.

  This file may be licensed under the terms of of the GNU General Public
  License, version 3, as published by the Free Software Foundation. You can
  find it here: http://www.gnu.org/licenses/gpl.html
*/

#include "QScaledView.h"

#include <QPainter>
#include <QPaintEvent>
#include <iostream>

//----------------------------------------------------------------------------
QScaledView::QScaledView(QWidget *parent)
  : QOpenGLWidget(parent)
{
}

//----------------------------------------------------------------------------
void QScaledView::setPixmap(const QPixmap &p)
{
  isDataSet = true;
  QSize oldSize = pixmap.size();
  pixmap = p;
  if( oldSize != pixmap.size() )
  {
    resizeEvent(0);
  }
  update();
}


//----------------------------------------------------------------------------
void QScaledView::clearPixmap()
{
  pixmap = QPixmap();
  isDataSet = false;
}

//----------------------------------------------------------------------------
void QScaledView::resizeEvent(QResizeEvent *ev)
{
  if (!isDataSet)
  {
    return;
  }

  // determine scale of correct aspect-ratio
  float src_aspect = pixmap.width()/(float)pixmap.height();
  float dest_aspect = width()/(float)height();
  float w;  // new width
  if (src_aspect > dest_aspect)
  {
    w = width() - 1;
  }
  else
  {
    w = height()*src_aspect - 1;
  }

  scale = w/pixmap.width();
  scaler = QTransform().scale(scale, scale);
  scalerI = scaler.inverted();
}

//----------------------------------------------------------------------------
void QScaledView::paintEvent(QPaintEvent *ev)
{
  QPainter painter(this);

  painter.setRenderHints(QPainter::SmoothPixmapTransform);

  painter.setWorldTransform(scaler);
  QRect damaged = scalerI.mapRect(ev->rect());
  painter.drawPixmap(damaged, pixmap, damaged);
}


//----------------------------------------------------------------------------
void QScaledView::mouseMoveEvent(QMouseEvent *ev)
{
  cursorAction(ev);
}

//----------------------------------------------------------------------------
void QScaledView::mousePressEvent(QMouseEvent *ev)
{
  cursorAction(ev, true);
}

//----------------------------------------------------------------------------
void QScaledView::cursorAction(QMouseEvent *ev, bool click)
{
}

