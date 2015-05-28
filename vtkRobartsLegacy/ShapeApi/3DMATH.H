// this file uses JavaDoc style comment blocks for automatic extraction of source code documentation.

/*!\file
 \brief Contains some useful math helpers for common 3D graphics entities.

 In this file are implemented some mathematical entities often used in 3D graphics (vecors, points, etc.) plus some useful constants, macros and definitions
 \version 0.1
 \date 12/06/99
 \author Alessandro Falappa
*/

#ifndef _3D_MATH_H
#define _3D_MATH_H

#include <iostream>
#include <assert.h>
#include <math.h>
#ifdef _AFXDLL // see if we are using MFC...
#include <afxwin.h>
#endif

#ifdef HIGH_PRECISION

//! The base type for all the math helpers
typedef double real;
//! the treshold for comparisons with zero, mainly used to avoid division by zero errors
const real epsilon=1e-12;
//! defined when high precision is requested
#define REAL_IS_DOUBLE

#else

// WARNING: these pragmas below could be MSVC compiler specific
#pragma warning( push )// memorize the warning status
#pragma warning( disable : 4305 )// disable "initializing : truncation from 'const double' to 'float'" warning
#pragma warning( disable : 4244 )// disable "double to float conversion possible loss of data" warning
#pragma warning( disable : 4136 )// disable "conversion between different floating-point types" warning
#pragma warning( disable : 4309 )// disable " 'conversion' : truncation of constant value" warning
#pragma warning( disable : 4051 )// disable " 'type conversion' ; possible loss of data" warning
//! The base type for all the math helpers
typedef float real;
//! the treshold for comparisons with zero, mainly used to avoid division by zero errors
const real epsilon=1e-7;
//! defined when high precision is not requested
#define REAL_IS_FLOAT

#endif

//=============================================================================
//=============================================================================

//!A vector class.
/*!
The vector class incapsulates a classic C++ vector of three real values and treats them as a 3 dimensional mathematical vectors.

The most common operations between mathematical vectors (and some which involves scalars too) are defined.
*/
class vector
{
private:
  static int counter;//!< counts how many vector objects are present
  real vec[3];//!< the actual vector

public:
  vector cross(vector &other);
  real dot(vector &other);
// operators
  vector();//!< default constructor
  vector(const real& x,const real& y,const real& z);//!< constructs a vector from three values
  vector(vector& from, vector& to);//!< constructs a vector from two other vectors
  vector(const vector& other);//!< the copy constructor
  ~vector();//!< the distructor
  real& x();//!< accessor for the x component (can be used as l-value too)
  real& y();//!< accessor for the y component (can be used as l-value too)
  real& z();//!< accessor for the z component (can be used as l-value too)
  real x() const;//!< returns the x component (r-value only)
  real y() const;//!< returns the y component (r-value only)
  real z() const;//!< returns the z component (r-value only)
  vector& operator=(const vector& other);//!< the assignment
  vector& operator+=(const vector& other);//!< the sum & assign
  vector& operator-=(const vector& other);//!< the subtract & assign
  vector& operator*=(const real& fact);//!< the short multiply by a scalar factor & assign
  vector& operator/=(const real& fact);//!< the short divide by a scalar factor & assign
  real& operator[](const int& index);//!< an "access like a C++ vector"
#ifdef HIGH_PRECISION
  operator double*();//!< the conversion to a 3 double elements C++ vector
#else
  operator float*();//!< the conversion to a 3 float elements C++ vector
#endif
//  operator char*();//!< the conversion of the vector into a textual form (null terminated string)
  real normalize();//!< normalize the vector
  vector normalized() const;//!< normalized copy of the vector
  real length() const;//!< get the length of the vector
  real length2() const;//!< get the squared length of the vector
  void EpsilonCorrect(const vector& v);//!< if the vector is almost equal to the origin substitute it with v
// ststic functions
  static int howMany();//!< returns how many vector objects exists

// friend functions  
  friend int operator==(const vector& v1,const vector& v2);//!< the equality operator
  friend int operator!=(const vector& v1,const vector& v2);//!< the inequality operator
  friend vector operator+(const vector& v1,const vector& v2);//!< the sum
  friend vector operator-(const vector& v1,const vector& v2);//!< the difference
  friend vector operator-(const vector& v1);//!< the negation
  friend real operator*(const vector& v1,const vector& v2);//!< the vector dot product
  friend vector operator^(const vector& v1,const vector& v2);//!< the vector cross product
  friend vector operator*(const vector& v,const real& fact);//!< the multiply a vector by a scalar factor
  friend vector operator*(const real& fact,const vector& v);//!< the multiply a scalar factor by a vector
  friend vector operator/(const vector& v,const real& fact);//!< the divide a vector by a scalar factor
  friend vector Bisect(vector v0,vector v1);//!< returns the unit vector which halves the arc between v0 and v1
  friend void clamp(vector& vec,const real& low,const real& high);//!< clamps all the vector components between the given tresholds
  friend std::ostream& operator<<(std::ostream& os,const vector& vect);//!< the print vector to C++ stream
#ifdef _AFXDLL // see if we are using MFC...
#ifdef _DEBUG    //.. and if we are in a debug build
  friend CDumpContext& operator<<(CDumpContext& cd,const vector& vect);//!< the print vector to MSVC++ debug console
#endif
#endif
};
//-----------------------------------------------------------------------------
// inlines

inline vector::vector()
{
  vec[0]=vec[1]=vec[2]=0.0;
  counter++;
}

inline vector::vector(const real& x,const real& y,const real& z)
{
  vec[0]=x;
  vec[1]=y;
  vec[2]=z;
  counter++;
}

/*!
This function constructs the vector which goes from the vector \e from to the vector \e to.
It is useful when a vector has to be constructed from the difference of two other vectors.
\param from the first vector
\param to the second vector
*/
inline vector::vector(vector& from, vector& to)
{
  vec[0]=to.vec[0]-from.vec[0];
  vec[1]=to.vec[1]-from.vec[1];
  vec[2]=to.vec[2]-from.vec[2];
  counter++;
}

inline vector::vector(const vector& other)
{
  *this=other;
  counter++;
}

inline vector::~vector()
{
  counter--;
}

inline real& vector::x()
{
  return vec[0];
}

inline real& vector::y()
{
  return vec[1];
}

inline real& vector::z()
{
  return vec[2];
}

inline real vector::x() const
{
  return vec[0];
}

inline real vector::y() const
{
  return vec[1];
}

inline real vector::z() const
{
  return vec[2];
}

inline vector& vector::operator=(const vector& other)
{
  //check for 'a=a' case
  if (this==&other) return *this;
  vec[0]=other.vec[0];
  vec[1]=other.vec[1];
  vec[2]=other.vec[2];
  return *this;
}

inline vector& vector::operator+=(const vector& other)
{
  vec[0]+=other.vec[0];
  vec[1]+=other.vec[1];
  vec[2]+=other.vec[2];
  return *this;
}

inline vector& vector::operator-=(const vector& other)
{
  vec[0]-=other.vec[0];
  vec[1]-=other.vec[1];
  vec[2]-=other.vec[2];
  return *this;
}

inline vector& vector::operator*=(const real& fact)
{
  vec[0]*=fact;
  vec[1]*=fact;
  vec[2]*=fact;
  return *this;
}

inline vector& vector::operator/=(const real& fact)
{
  assert(fabs(fact) >= epsilon);
  vec[0]/=fact;
  vec[1]/=fact;
  vec[2]/=fact;
  return *this;
}

/*!
This operator redirects the access to the internal vector. It does make a range check on the index in debug builds trough the ANSI assert function.
It can be used on both sides of an assignment.
\return a reference to the requested element
\param index the index which should be one of 0,1,2
*/
inline real& vector::operator[](const int& index)
{
  assert(index>=0 && index<=2);
  return vec[index];
}

inline real vector::length2() const
{
  return (*this)*(*this);
}

inline real vector::length() const
{
  return sqrt(this->length2());
}

/*!
The counting of the vector objects is realized trough a static counter variable.
\return the number of vector objects in memory
*/
inline int vector::howMany()
{
  return counter;
}

/*!
this conversion operator allows to use a vector in places of a classic real[3]
*/
/*
inline vector::operator real*()
{
  return (real*)vec;
}
*/
#ifdef HIGH_PRECISION

/*!
this conversion operator allows to use a vector in places of a classic double[3]
*/
inline vector::operator double*()
{
  return (double*)vec;
}

#else

/*!
this conversion operator allows to use a vector in place of a classic float[3]
*/
inline vector::operator float*()
{
  return (float*)vec;
}

#endif

//-----------------------------------------------------------------------------
// useful constants (declaration)

extern const vector ORIGIN;
extern const vector X_AXIS;
extern const vector Y_AXIS;
extern const vector Z_AXIS;

//=============================================================================
//=============================================================================

/*!
A transformation matrix class.

The matrix class groups sixteen real values an treats them as a 4x4 matrix. Standard
C++ matrices (2 dimensional vectors) are stored by row, since for graphical applications
a column major order is preferable the access indices are internally swapped.

The most common operators between matrices (and some which involves scalars and vectors too) are defined.
*/
class tmatrix
{
private:
  double covalue(int a1, int a2, int a3, int b1, int b2, int b3);
  static int counter;//!< counts how many matrices objects are present
  real mat[4][4];//!< the matrix data
public:
  double  * getMat();
  void getRPY(double &roll, double &pitch, double &yaw);
  void setAngles(float roll, float pitch, float yaw);
  void getAngles(float &roll, float &pitch, float &yaw);
  
  double Determinate();
  void inverse();
  //! Espresses how to store a tmatrix in a single 16 elements vector, by column or by row
  enum ordermode
  {
    COLUMN,//!< column major order
    ROW//!< row major order
  };
  tmatrix();//!< default constructor
  tmatrix(const double vector[16]);
  tmatrix(const real& val);//!< constructs a tmatrix and fills it with a value
  tmatrix(const real vector[16],ordermode ord=COLUMN);//!< constructs a tmatrix from a vector which contains a 4x4 matrix in row major or column major order
  tmatrix(const tmatrix& other);//!< the copy constructor
  ~tmatrix();//!< the distructor
  tmatrix& operator-();//!< negation
  tmatrix& operator=(const tmatrix& other);//!< assignment
  tmatrix& operator+=(const tmatrix& other);//!< sum & assign
  tmatrix& operator-=(const tmatrix& other);//!< subtract & assign
  tmatrix& operator*=(const tmatrix& other);//!< multiply by a tmatrix & assign
  double* times(const double vector[4]);
  tmatrix& operator*=(const real& fact);//!< multiply by a scalar factor & assign
  tmatrix& operator/=(const real& fact);//!< divide by a scalar factor & assign
  real& operator()(const int& row,const int& col);//!< an "access like a C++ tmatrix"
#ifdef HIGH_PRECISION
  operator double*();//!< conversion to a 16 doubles C++ vector (column major order)
#else
  operator float*();//!< conversion to a 16 floats C++ vector (column major order)
#endif
  void loadIdentity();//!< fills the matrix with the identity matrix
//  operator char*();//!< conversion of the tmatrix into a textual form (null terminated string)
//static functions
  static int howMany();//!< returns how many tmatrix objects exists
// friend functions  
  friend int operator==(const tmatrix& t1,const tmatrix& t2);//!< the equality operator
  friend int operator!=(const tmatrix& t1,const tmatrix& t2);//!< the inequality operator
  friend tmatrix operator+(const tmatrix& t1,const tmatrix& t2);//!< the sum
  friend tmatrix operator-(const tmatrix& t1,const tmatrix& t2);//!< the difference
  friend tmatrix operator*(const tmatrix& t1,const tmatrix& t2);//!< tmatrix product
  friend tmatrix operator*(const tmatrix& tmat,const real& fact);//!< multiply a tmatrix by a scalar factor
  friend tmatrix operator*(const real& fact,const tmatrix& tmat);//!< multiply a scalar factor by a tmatrix
  friend tmatrix operator/(const tmatrix& tmat,const real& fact);//!< divide a tmatrix by a scalar factor
  friend std::ostream& operator<<(std::ostream& os,const tmatrix& m);//!< print tmatrix to C++ stream
#ifdef _AFXDLL // see if we are using MFC
#ifdef _DEBUG
  friend CDumpContext& operator<<(CDumpContext& cd,const tmatrix& m);//!< print tmatrix to MSVC++ debug console
#endif
#endif
};

//-----------------------------------------------------------------------------
// inlines

inline tmatrix::tmatrix(const tmatrix& other)
{
  *this=other;
  counter++;
}

inline tmatrix::~tmatrix()
{
  counter--;
}

/*!
The counting of the tmatrix objects is realized trough a static counter variable.
\return the number of tmatrix objects in memory
*/
inline int tmatrix::howMany()
{
  return counter;
}

/*!
This operator redirects the access to the internal matrix. It does make a range
check on the index in debug builds trough the ANSI assert function.
It can be used on both sides of an assignment.
\return a reference to the requested element
\param row the row index which should be one of 0,1,2,3
\param col the column index which should be one of 0,1,2,3
\date 18/06/99
*/
inline real& tmatrix::operator()(const int& row,const int& col)
{
  assert(row>=0 && row<=3);
  assert(col>=0 && col<=3);
  return mat[col][row];// swap indices to store by column
}

#ifdef HIGH_PRECISION

/*!
this conversion operator allows to use a tmatrix in places where a column major
order vector of 16 double elements is requested (e.g. the OpenGL functions for
retrieving/setting the modelview or projection matrix).
*/
inline tmatrix::operator double*()
{
  return (double*)mat;
}

#else

/*!
this conversion operator allows to use a tmatrix in places where a column major
order vector of 16 float elements is requested (e.g. the OpenGL functions for
retrieving/setting the modelview or projection matrix).
*/
inline tmatrix::operator float*()
{
  return (float*)mat;
}

#endif

//-----------------------------------------------------------------------------
// useful constants


//=============================================================================
//=============================================================================

/*!
A generic quaternion2 class.
The quaternion2 uses internally the (s,<B>v</B>) representation, where s is a scalar and <B>v</B> a vector.
The most common operations from quaternion2 calculus are implemented.

Some internet references are:
<UL>
<LI>http://www.cs.berkley.edu/~laura/cs184/quat/quaternion2.html
<LI>http://www.astro.virginia.edu/~eww6n/math/quaternion2.html
<LI>http://forum.swarthmore.edu/dr.math/problems/prasad2.17.96.html
<LI>http://graphics.cs.ucdavis.edu/GraphicsNotes/quaternion2s/quaternion2s.html
</UL>

A paper by Ken Shoemake (the quaternion2s "inventor") is available at<BR>
<UL>
<LI>ftp://ftp.cis.upenn.edu/pub/graphics/shoemake/
</UL>
*/
class quaternion2
{
private:
  static int counter;//!< counts how many quaternion2 objects are present
protected:
  real s;//!< the scalar part of a quaternion2
  vector v;//!< the vector part of a quaternion2
public:
  quaternion2();//!< default constructor
  quaternion2(const real& scal, const vector& vec);//!< constructs a quaternion2 from the scalar and vector components
  quaternion2(const vector& axis, const real& angle);
  quaternion2(const real& s1,const real& s2,const real& s3,const real& s4);//!< constructs a quaternion2 from four real values
  quaternion2(const real& roll, const real& pitch, const real& yaw);
  quaternion2(const double& roll, const double& pitch, const double& yaw, BOOL bTyphoon=FALSE);
  quaternion2(const quaternion2& other);//!< the copy constructor
  virtual ~quaternion2();//!< the distructor
  real& x();//!< accessor for the x component of the vector part (can be used as l-value too)
  real& y();//!< accessor for the y component of the vector part (can be used as l-value too)
  real& z();//!< accessor for the z component of the vector part (can be used as l-value too)
  real& w();//!< accessor for the scalar part (can be used as l-value too)
  real& scalarPart();//!< other accessor for the scalar component (can be used as l-value too)
  vector& vectorPart();//!< accessor for the vector part (can be used as l-value too)
  quaternion2& operator=(const quaternion2& other);//!< assignment
  quaternion2& operator+=(const quaternion2& other);//!< sum & assign
  quaternion2& operator-=(const quaternion2& other);//!< subtract & assign
  quaternion2& operator*=(const quaternion2& other);//!< multiply by a quaternion2 & assign
  quaternion2& operator/=(const quaternion2& other);//!< divide by a quaternion2 & assign
  quaternion2& operator*=(const real& fact);//!< multiply by a scalar factor & assign
  quaternion2& operator/=(const real& fact);//!< divide by a scalar factor & assign
//  operator float*();//!< conversion to a 4 elements C++ vector
//  operator char*();//!< conversion of the quaternion2 into a textual form (null terminated string)
  real normalize();//!< normalize the quaternion2
  quaternion2 normalized() const;//!< normalized copy of the quaternion2
  void conjugate();//!< conjugate of the quaternion2
  quaternion2 conjugated() const;//!< conjugated copy of the quaternion2
  real inverse();//!< inverse of the quaternion2
  quaternion2 inversed() const;//!< inversed copy of the quaternion2
  real length() const;//!< get the length of the quaternion2
  real norm() const;//!< get the norm (similar to the squared length) of the quaternion2
  virtual tmatrix getRotMatrix();//<! constructs a rotation matrix from the quaternion2
// static functions
  static int howMany();//!< returns how many quaternion2 objects exists
// friend functions  
  friend int operator==(const quaternion2& q1,const quaternion2& q2);//!< the equality operator
  friend int operator!=(const quaternion2& q1,const quaternion2& q2);//!< the inequality operator
  friend quaternion2 operator+(const quaternion2& q1,const quaternion2& q2);//!< the sum
  friend quaternion2 operator-(const quaternion2& q1,const quaternion2& q2);//!< the difference
  friend quaternion2 operator-(const quaternion2& q1);//!< negation
  friend quaternion2 operator*(const quaternion2& q1,const quaternion2& q2);//!< quaternion2 product
  friend quaternion2 operator*(const quaternion2& q,const real& s);//!< multiply a quaternion2 by a scalar
  friend quaternion2 operator*(const real& s,const quaternion2& q);//!< multiply a scalar by a quaternion2
  friend quaternion2 operator/(const quaternion2& q,const real& s);//!< divide a quaternion2 by a scalar factor
  friend quaternion2 operator/(const quaternion2& q1,const quaternion2& q2);//!< divide a quaternion2 by a quaternion2
  friend std::ostream& operator<<(std::ostream& os,const quaternion2& q);//!< print quaternion2 to C++ stream
#ifdef _AFXDLL // see if we are using MFC
#ifdef _DEBUG
  friend CDumpContext& operator<<(CDumpContext& ad,const quaternion2& q);//!< print quaternion2 to MSVC++ debug console
#endif
#endif
};

//-----------------------------------------------------------------------------
// inlines

inline quaternion2::quaternion2()
{
  s=0.0;
  counter++;
};

/*!
This constructor assumes an (s,\b v) form so interprets the real calues as w,x,y,z respectively.
Another common representation for quaternion2s, the "homogeneous" one, is x,y,x,w ordered.
\param s1 the scalar part
\param s2 the x component of the vector part
\param s3 the y component of the vector part
\param s4 the z component of the vector part
\date 15/06/99
*/
inline quaternion2::quaternion2 (const real& s1,const real& s2,const real& s3,const real& s4)
:s(s1),v(s2,s3,s4)
{
  counter++;
}

inline quaternion2::quaternion2 (const real& roll, const real& pitch, const real& yaw)
{
  vector vx( 1, 0, 0 );
  vector vy( 0, 1, 0 );
  vector vz( 0, 0, 1 );
  
  quaternion2 qx(vx,-pitch*0.01745329251994); // convert angles from deg to rad
  quaternion2 qy(vy,yaw*0.01745329251994);
  quaternion2 qz(vz,-roll*0.01745329251994);
  quaternion2 qt1(1,0,0,0);
  quaternion2 qt2(1,0,0,0);

  qx.normalize();
  qy.normalize();
  qz.normalize();

  //qt1 = qy * qx;
  //qt2 = qt1 * qz;
  qt1 = qz * qy;
  qt2 = qt1 * qx;

  qt2.normalize();

  this->s = qt2.s;
  this->v = qt2.v;

  float angle = acos(qt2.w())*2;
  double x = qt2.x()/sin(angle/2);
  double y = qt2.y()/sin(angle/2);
  double z = qt2.z()/sin(angle/2);


 /* quaternion2_from_euler( quaternion2 *q, VFLOAT ax, VFLOAT ay, VFLOAT az )
    {
    VECTOR3 vx = { 1, 0, 0 }, vy = { 0, 1, 0 }, vz = { 0, 0, 1 };
    quaternion2 qx, qy, qz, qt;

    quaternion2_from_axisangle( qx, &vx, rx );
    quaternion2_from_axisangle( qy, &vy, ry );
    quaternion2_from_axisangle( qz, &vz, rz );

    quaternion2_multiply( &qt, &qx, &qy );
    quaternion2_multiply( &q,  &qt, &qz );
    }*/
}

inline quaternion2::quaternion2 (const double& roll, const double& pitch, const double& yaw, BOOL bTyphoon/*=FALSE*/)
{
  //constructor used for Measurand's definition of roll, pitch, and yaw
  //assume all angles are already in radians, and that the rotation order is yaw, pitch, roll
  vector vx(1,0,0);
  vector vy(0,1,0);
  vector vz(0,0,1);
  if (!bTyphoon)
  {
    quaternion2 qx(vx,-roll*0.01745329251994);
    quaternion2 qy(vy,-yaw*0.01745329251994);
    quaternion2 qz(vz,pitch*0.01745329251994);
    quaternion2 qt1((real)1,(real)0,(real)0,(real)0);
    quaternion2 qt2((real)1,(real)0,(real)0,(real)0);
    qx.normalize();
    qy.normalize();
    qz.normalize();
    qt1 = qy*qz;
    qt2 = qt1*qx;
    
    qt2.normalize();

    this->s = qt2.s;
    this->v = qt2.v;
  }
  else
  {
    quaternion2 qx(vx,pitch*0.01745329251994); // convert angles from deg to rad
    quaternion2 qy(vy,roll*0.01745329251994);
    quaternion2 qz(vz,yaw*0.01745329251994);
      
    quaternion2 qt1((real)1,(real)0,(real)0,(real)0);
    quaternion2 qt2((real)1,(real)0,(real)0,(real)0);

    qt1 = qy * qz;
    qt2 = qx * qt1;

    qt2.normalize();

    this->s = qt2.s;
    this->v = qt2.v;
  }
}

inline quaternion2::quaternion2 (const real& scal, const vector& vec)
:s(scal),v(vec)
{
  s=scal;
  counter++;
}

inline quaternion2::quaternion2 (const vector& axis, const real& angle)
:s(cos(angle/2)),v(sin(angle/2)*axis.normalized())
{ 
}

inline quaternion2::~quaternion2()
{
  counter--;
}

inline quaternion2::quaternion2(const quaternion2& other)
{
  *this=other;
  counter++;
}

/*!
The counting of the quaternion2 objects is realized trough a static counter variable.
\return the number of quaternion2 objects in memory
*/
inline int quaternion2::howMany()
{
  return counter;
}

inline real& quaternion2::x()
{
  return v.x();
}

inline real& quaternion2::y()
{
  return v.y();
}

inline real& quaternion2::z()
{
  return v.z();
}

inline real& quaternion2::w()
{
  return s;
}

inline real& quaternion2::scalarPart()
{
  return s;
}

inline vector& quaternion2::vectorPart()
{
  return v;
}

inline quaternion2& quaternion2::operator=(const quaternion2& other)
{
  //check for 'a=a' case
  if (this==&other) return *this;
  s=other.s;
  v=other.v;
  return *this;
}

inline quaternion2& quaternion2::operator+=(const quaternion2& other)
{
  s+=other.s;
  v+=other.v;
  return *this;
}

inline quaternion2& quaternion2::operator-=(const quaternion2& other)
{
  s-=other.s;
  v-=other.v;
  return *this;
}

inline quaternion2& quaternion2::operator*=(const real& fact)
{
  s*=fact;
  v*=fact;
  return *this;
}

inline quaternion2& quaternion2::operator/=(const real& fact)
{
  assert(fabs(fact) >= epsilon);
  s/=fact;
  v/=fact;
  return *this;
}

inline real quaternion2::length() const
{
  return sqrt(norm());
}

inline void quaternion2::conjugate()
{
  v=-v;
}

inline quaternion2 quaternion2::conjugated() const
{
  return quaternion2(s,-v);
}

/*!
This function check if all elements of \e v1 are equal to the corresponding elements of \e v2.
*/
inline int operator==(const vector& v1,const vector& v2)
{
  if(v1.vec[0]==v2.vec[0] && v1.vec[1]==v2.vec[1] && v1.vec[2]==v2.vec[2]) return 1;
  else return 0;
}

/*!
This function check if some of the elements of \e v1 differs from the corresponding elements of \e v2.
*/
inline int operator!=(const vector& v1,const vector& v2)
{
  if(v1.vec[0]==v2.vec[0] && v1.vec[1]==v2.vec[1] && v1.vec[2]==v2.vec[2]) return 0;
  else return 1;
}


//=============================================================================
//=============================================================================

/*!
A unit lenght quaternion2 class.

The unitquaternion class manages quaternion2s of unit length, such quaternion2s
can be used to represent arbitrary rotations. To mantain the unit lenght property
along quaternion2 calculus addition and subtraction as well as multiplication and
division by scalars are not allowed (they're made private or they're overloaded
by "trap" functions).

Actually only a few member functions of the base class are overloaded since
the quaternion2 calculus apply with no modification to unit quaternion2s.
*/
class unitquaternion : public quaternion2
{
private:
  unitquaternion& operator+=(const unitquaternion& other);//!< sum & assign is not allowed
  unitquaternion& operator-=(const unitquaternion& other);//!< subtract & assign is not allowed
  unitquaternion& operator*=(const real& fact);//!< multiply by a scalar factor & assign is not allowed
  unitquaternion& operator/=(const real& fact);//!< divide by a scalar factor & assign is not allowed
  unitquaternion(const vector& v, const real& s);//!< constructs a unitquaternion assigning directly the scalar and vector parts
public:
  real x() const;//!< accessor for the x component of the vector part
  real y() const;//!< accessor for the y component of the vector part
  real z() const;//!< accessor for the z component of the vector part
  real w() const;//!< accessor for the scalar part
  real scalarPart() const;//!< other accessor for the scalar component
  vector vectorPart() const;//!< accessor for the vector part
  unitquaternion();//!< default constructor
  unitquaternion(const real& angle, const vector& axis);//!< constructs a unitquaternion representing a rotation of angle radiants about axis 
  unitquaternion(const real& roll, const real& pitch, const real& yaw);// constructs a quaternion2 from roll, pitch and yaw values.
  unitquaternion(const quaternion2& q);//!< constructs a unitquaternion from a generic one (conversion by normalizing)
  ~unitquaternion();//!< the distructor
  unitquaternion inversed() const;//!< inversed copy of the unitquaternion
  void inverse();//!< inverse of the unitquaternion
  tmatrix getRotMatrix();//<! constructs a rotation matrix from the quaternion2
  void getVectorsOnSphere(vector& vfrom,vector& vto);//<! converts a unitquaternion to two vectors on a unit sphere (the extremes of a rotation)
  unitquaternion& operator*=(const unitquaternion& other);//!< multiply by another unitquaternion & assign is not allowed
// friend functions  
  friend unitquaternion operator+(const unitquaternion& q1,const unitquaternion& q2);//!< the sum is not allowed
  friend unitquaternion operator-(const unitquaternion& q1,const unitquaternion& q2);//!< the difference is not allowed
  friend unitquaternion operator*(const unitquaternion& q,const real& s);//!< multiply a unitquaternion by a scalar is not allowed
  friend unitquaternion operator*(const real& s,const unitquaternion& q);//!< multiply a scalar by a unitquaternion is not allowed
  friend unitquaternion operator/(const unitquaternion& q,const real& s);//!< divide a unitquaternion by a scalar factor is not allowed
};

//-----------------------------------------------------------------------------
// inlines

inline unitquaternion::unitquaternion()
:quaternion2(1,0,0,0)
{
};


inline unitquaternion::unitquaternion (const quaternion2& q)
:quaternion2(q)
{
  normalize();
}


/*!
This constructor has a different meaning from the same in the base class. Here
the vector means an axis of rotation while the real means the angle to rotate
about the axis.

\b NOTE: the angle and axis vector are not directly assigned to the real part
and the vector part, respectively, of the quaternion2. The unit quaternion2 (<I>s</I>,<B>v</B>)
represents a rotation of \b angle radians about the axis \b a if:

\e s = cos(\e angle / 2)<BR>
\b v = \b a * sin( \e angle / 2)
\param angle the rotation angle
\param axis the axis of rotation
\date 18/06/99
*/
inline unitquaternion::unitquaternion (const real& angle, const vector& axis)
:quaternion2( cos(angle/2), sin(angle/2)*axis.normalized() )
{
}

inline unitquaternion::unitquaternion (const real &roll, const real &pitch, const real &yaw)
:quaternion2( roll, pitch, yaw )
{
}

/*!
This constructor has been introduced exclusively to make the inversed function
more efficient by avoiding too many unitquaternion copies and conversions to
quaternion2s.
*/
inline unitquaternion::unitquaternion (const vector& v, const real& s)
:quaternion2( s, v )
{
}

inline unitquaternion::~unitquaternion()
{
}

/*!
For unitquaternions the inverse equals the conjugate (which is simpler to calculate)
This function doesn't modifies the quaternion2 upon which has been called, it returns a new quaternion2 instead.
\return the normalized copy of the quaternion2
\date 18/06/99
*/
inline unitquaternion unitquaternion::inversed() const
{
  return unitquaternion(-v,s);
}

/*!
For unitquaternions the inverse equals the conjugate (which is simpler
to calculate). The function doesn't call the base conjugate function to
avoid an expensive quaternion2 to unitquaternion conversion.
This function modifies the quaternion2 upon which has been called.
*/
inline void unitquaternion::inverse()
{
  v=-v;
}

/*!
This function overrides the same in base class to prevent the use as an l-value
(that is to modify the x component of the vector part).
*/
inline real unitquaternion::x() const
{
  return ((vector&)v).x();// explicit cast to use the non const vector x() function
}

/*!
This function overrides the same in base class to prevent the use as an l-value
(that is to modify the y component of the vector part).
*/
inline real unitquaternion::y() const
{
  return ((vector&)v).y();// explicit cast to use the non const vector y() function
}

/*!
This function overrides the same in base class to prevent the use as an l-value
(that is to modify the z component of the vector part).
*/
inline real unitquaternion::z() const
{
  return ((vector&)v).z();// explicit cast to use the non const vector z() function
}

/*!
This function overrides the same in base class to prevent the use as an l-value
(that is to modify the scalar part).
*/
inline real unitquaternion::w() const
{
  return s;
}

/*!
This function overrides the same in base class to prevent the use as an l-value
(that is to modify the scalar part).
*/
inline real unitquaternion::scalarPart() const
{
  return s;
}

/*!
This function overrides the same in base class to prevent the use as an l-value
(that is to modify the vector part).
*/
inline vector unitquaternion::vectorPart() const
{
  return v;
}


//-----------------------------------------------------------------------------
// useful constants

//=============================================================================
//=============================================================================

/* inserire qui alcune macro o funzioni globali come ad esempio:
  - clamping
  - conversione angoli gradi<->radianti
*/

/*!
limits a value in a range, modifying it.
\param val the value to clamp
\param low the lower treshold
\param high the higher treshold
*/
inline void clamp(real& val,const real& low,const real& high)
{
  if(val<low) val=low;
  if(val>high) val=high;
}

/*!
limits a value in a range, returning the clamped value.
\return the clamped value
\param val the value to clamp
\param low the lower treshold
\param high the higher treshold
*/
inline real clamped(const real& val,const real& low,const real& high)
{
  if(val<low) return low;
  else if(val>high) return high;
    else return val;
}


/*!
limits the vector components in a range, modifying it.
\param theVec the vector to clamp
\param low the lower treshold
\param high the higher treshold
*/
inline void clamp(vector& theVec,const real& low,const real& high)
{
  clamp(theVec.vec[0],low,high);
  clamp(theVec.vec[1],low,high);
  clamp(theVec.vec[2],low,high);
}

/*!
\short returns \e angle expressed in degrees.
\return the angle expresses in radians
\param angle the angle value
\author Alessandro Falappa
*/
inline real RadToDeg(const real& angle)
{
  return angle*57.29577951308;
}

/*!
\short returns \e angle expressed in radians.
\return the angle expresses in degrees
\param angle the angle value
*/
inline real DegToRad(const real& angle)
{
  return angle*0.01745329251994;
}

/*!
\short converts radiants to degrees.
This function modify its argument.
\param angle the angle to be converted
*/
inline void ConvertToDeg(real& angle)
{
  angle*=57.29577951308;
}

/*!
\short converts degrees to radiants.
This function modify its argument.
\param angle the angle to be converted
*/
inline void ConvertToRad(real& angle)
{
  angle*=0.01745329251994;
}

/*!
\short absolute value function which executes a simple test
This function executes a simple test on \e val negativity returning the
opposite if true. Such a test can be faster than the call to the \e fabs
library routine
\return the absolute value of \e val
*/
inline real simpleabs(const real &val)
{
  return val>0?val:-val;
}
//! the greek pi constant
extern const real G_PI;

//! greek pi / 2
extern const real G_HALF_PI;

//!2 * greek pi
extern const real G_DOUBLE_PI;

#ifdef _AFXDLL
#undef assert
//!maps the assert function to the MFC ASSERT macro
#define assert(exp) ASSERT(exp)
#endif

#ifndef HIGH_PRECISION
// this below could be MSVC compiler specific
#pragma warning( pop )// reset the warning status
#endif

#endif // _3D_MATH_H