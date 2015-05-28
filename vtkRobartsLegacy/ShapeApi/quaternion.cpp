#include "stdafx.h"
#include "quaternion.h"
#include <cmath>

quaternion::quaternion(void) // Constructor.
{
    m_w = 1.0;       // Set a default quaternion that is equivalent to
    m_x = 0.0;       // the identity rotation matrix.
  m_y = 0.0;
    m_z = 0.0;
}

quaternion::quaternion(double w, double x, double y, double z)  // Constructor.
{
  m_w = w;    // Set the quaternion object to the passed values.
  m_x = x;
  m_y = y;
  m_z = z;
}

quaternion::~quaternion(void) 
{}

// Set / get functions.
void quaternion::setQuaternion(double w, double x, double y, double z)   
{
  m_w = w;
  m_x = x;
  m_y = y;
  m_z = z;
}

void quaternion::setQuaternion(double vals[4]) 
{
  m_w = vals[0];
  m_x = vals[1];
  m_y = vals[2];
  m_z = vals[3];  
}

void quaternion::getQuaternion(double vals[4])  
{
  vals[0] = m_w;
  vals[1] = m_x;
  vals[2] = m_y;
  vals[3] = m_z;
}

// Operators.
quaternion quaternion::operator*(const quaternion &q)    // Multiplication
{
  quaternion result;
    
    result.m_w = (m_w * q.m_w) - (m_x * q.m_x) - (m_y * q.m_y) - (m_z * q.m_z);
    result.m_x = (m_x * q.m_w) + (m_w * q.m_x) - (m_z * q.m_y) + (m_y * q.m_z);
    result.m_y = (m_y * q.m_w) + (m_z * q.m_x) + (m_w * q.m_y) - (m_x * q.m_z);
    result.m_z = (m_z * q.m_w) - (m_y * q.m_x) + (m_x * q.m_y) + (m_w * q.m_z);  
    
    return result;
}

quaternion quaternion::operator=(const quaternion &quat)    // Assignment
{
    m_w = quat.m_w;
    m_x = quat.m_x;
    m_y = quat.m_y;
    m_z = quat.m_z;
    return *this;
}

// Member functions.
void quaternion::normalize(void) 
{
  double factor;
    
    factor = sqrt( m_w * m_w + m_x * m_x + m_y * m_y + m_z * m_z );
    if( factor != 0.0 )
    {
    m_w = m_w / factor;
    m_x = m_x / factor;
    m_y = m_y / factor;
    m_z = m_z / factor;
    }
  else 
  {
    m_w = 0.0;
    m_x = 0.0;
    m_y = 0.0;
    m_z = 0.0;
  }
}

// Invert the current quaternion.
void quaternion::invert(void)
{
    m_x = -m_x;       // _w is unchanged
    m_y = -m_y;
    m_z = -m_z;
} 

// Convert a U,B vector set into a quaternion.
// Where U is along the tape and B is perpendicular to that towards
// the edge of the tape.  Therefore the data set is ratated by 
// bend around B and by twist around U.
quaternion::vector2Quaternion(double U[3], double B[3])
{
  double Uo[3] = {1,0,0}; // Base U vector.
  double Bo[3]; // Base B vector.
  double newB[4]; // Interim B vector to find the base B from.
  double U_axis[3], B_axis[3];
  double U_angle, B_angle;
  quaternion U_quat, B_quat, newB_q, inverse_U ,result;

  // first compare U to the U at the base {1,0,0}
  // find the axis to rotate from base to current U
  cross(Uo,U,U_axis);
  normalize(U_axis);
  // next, find the angle between them.
  U_angle = angleBetween(U,Uo);
  // create a quaternion from this axis and angle.
  U_quat = rotation2Quaternion(U_axis,U_angle);

  // repeat for B w/ a base at B' where B' = {0,0,1}
  // rotated by U_angle around U_axis...
  // first rotate B , using quaternions...
  newB_q.setQuaternion(0,0,0,1); // equivilant to the point {0,0,1}
  inverse_U = U_quat; // Set inverse_U to the values of U_quat
  inverse_U.invert(); // Now inverse_U = the inverse of U_quat.
  newB_q = U_quat * newB_q; // Want to find the new B vector after 
  newB_q = newB_q  * inverse_U; // the U transformation is applied.
  newB_q.getQuaternion(newB); // Used the formula P'=qPq' to find the 
  // new B.
  
  for (int i = 1; i < 4;i++)
  {
    Bo[i-1] = newB[i]; // Take the vector component out of the quaternion. 
  }

  cross(Bo,B,B_axis);
  normalize(B_axis);
  B_angle = angleBetween(B,Bo);
  B_quat = rotation2Quaternion(B_axis,B_angle);

  // normalize both quaternions...
  U_quat.normalize();
  B_quat.normalize();

  // multiply them together to get a single quaternion to 
  // represent a bend/twist pair.
  result = U_quat * B_quat;

  m_w = result.m_w;
  m_x = result.m_x;
  m_y = result.m_y;
  m_z = result.m_z;
}

quaternion::bt2Quaternion(double bend, double twist)
{
  quaternion bend_quat;
  double bend_axis[3] = {0,0,1};
  quaternion twist_quat;
  double twist_axis[3] = {1,0,0};
  quaternion result;

  bend_quat = rotation2Quaternion(bend_axis,bend);
  twist_quat = rotation2Quaternion(twist_axis,twist);

  result = bend_quat * twist_quat;
  m_w = result.m_w;
  m_x = result.m_x;
  m_y = result.m_y;
  m_z = result.m_z;
}

quaternion quaternion::rotation2Quaternion(double axis[3], double angle)
{
  quaternion result;

  double half_angle = angle /2;
  result.m_w = cos(half_angle);
  result.m_x = sin(half_angle) * axis[0];
  result.m_y = sin(half_angle) * axis[1];
  result.m_z = sin(half_angle) * axis[2];

  return result;
}

quaternion::cross(double A[3], double B[3], double C[3])
{
  C[0] = A[1]*B[2]-A[2]*B[1];
  C[1] = A[2]*B[0]-A[0]*B[2];
  C[2] = A[0]*B[1]-A[1]*B[0];
}

double quaternion::dot(double A[3], double B[3])
{
  double result;

  result = A[0]*B[0] + A[1]*B[1] + A[2]*B[2];
  return result;
}

double quaternion::magnitude(double A[3])
{
  double result;

  result = sqrt(A[0]*A[0] + A[1]*A[1] + A[2]*A[2]);
  return result;
}

double quaternion::angleBetween(double A[3], double B[3])
{
  double result;

  result = acos(dot(A,B)/magnitude(A));
  return result;
}

quaternion::normalize(double vector[3])
{
  double length;
  length = magnitude(vector);

  if (length != 0.0)
  {
    vector[0] = vector[0] / length;
    vector[1] = vector[1] / length;
    vector[2] = vector[2] / length;
  }
}

quaternion quaternion::slerp(quaternion from, quaternion to, float t)
{
  double   to1[4];
  double  omega, cosom, sinom, scale0, scale1;
  quaternion res;
  
  // calc cosine
  cosom = from.m_x * to.m_x + from.m_y * to.m_y + from.m_z * to.m_z
    + from.m_w * to.m_w;
  
  // adjust signs (if necessary)
  if ( cosom <0.0 )
  { 
    cosom = -cosom; 
    to1[0] = - to.m_x;
    to1[1] = - to.m_y;
    to1[2] = - to.m_z;
    to1[3] = - to.m_w;
  } else  {
    to1[0] = to.m_x;
    to1[1] = to.m_y;
    to1[2] = to.m_z;
    to1[3] = to.m_w;
  }
  
  // calculate coefficients
  
  if ( (1.0 - cosom) > DELTA ) {
    // standard case (slerp)
    omega = acos(cosom);
    sinom = sin(omega);
    scale0 = sin((1.0 - t) * omega) / sinom;
    scale1 = sin(t * omega) / sinom;
    
  } else {        
    // "from" and "to" quaternions are very close 
    //  ... so we can do a linear interpolation
    scale0 = 1.0 - t;
    scale1 = t;
  }
  // calculate final values
  res.m_x = scale0 * from.m_x + scale1 * to1[0];
  res.m_y = scale0 * from.m_y + scale1 * to1[1];
  res.m_z = scale0 * from.m_z + scale1 * to1[2];
  res.m_w = scale0 * from.m_w + scale1 * to1[3];

  return res;
}
