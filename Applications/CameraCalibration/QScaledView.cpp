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
  , pixmap(NULL)
{
}

//----------------------------------------------------------------------------
void QScaledView::setPixmap(const QPixmap &p)
{
  pixmap = &p;
  resizeEvent(0);
  update();
}

//----------------------------------------------------------------------------
void QScaledView::resizeEvent(QResizeEvent *ev)
{
  if (!pixmap)
  {
    return;
  }

  // determine scale of correct aspect-ratio
  float src_aspect = pixmap->width()/(float)pixmap->height();
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

  scale = w/pixmap->width();
  scaler = QTransform().scale(scale, scale);
  scalerI = scaler.inverted();
}

//----------------------------------------------------------------------------
void QScaledView::paintEvent(QPaintEvent *ev)
{
  QPainter painter(this);
  if (!pixmap)
  {
    painter.fillRect(this->rect(), QBrush(Qt::gray, Qt::BDiagPattern));
    return;
  }

  painter.setRenderHint(QPainter::SmoothPixmapTransform);

  painter.setWorldTransform(scaler);
  QRect damaged = scalerI.mapRect(ev->rect());
  painter.drawPixmap(damaged, *pixmap, damaged);
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

