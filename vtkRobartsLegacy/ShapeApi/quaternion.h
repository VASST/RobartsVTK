#ifndef QUATERNION_H
#define QUATERNION_H

#define DELTA 0.01

class quaternion
{
public:
	quaternion(void); // Constructor.
	quaternion(double w, double x, double y, double z); // Constructor.
    ~quaternion(void);
	// Set / get functions.
	void setQuaternion(double w, double x, double y, double z);  
	void setQuaternion(double vals[4]);
	void getQuaternion(double vals[4]); 
	// Operators.
	quaternion operator*(const quaternion &);   // Multiplication
	quaternion operator=(const quaternion &);   // Assignment
	// Member functions.
	void normalize(void);
	void invert(void);
	vector2Quaternion(double U[3], double B[3]);
	bt2Quaternion(double bend, double twist);
	quaternion rotation2Quaternion(double axis[3], double angle);
	quaternion slerp(quaternion from, quaternion to, float t);

private:
	// Vector math - maybe create another class for this later.
	cross(double A[3], double B[3], double C[3]); // A cross B = C
	double dot(double A[3], double B[3]);
	double magnitude(double A[3]);
	double angleBetween(double A[3], double B[3]);
	normalize(double vector[3]);

public:
	double m_w; // Twist about axis (real part)
	double m_x; // Axis (imaginary part of quaternion)
	double m_y;
	double m_z;
};
#endif // quaternion.h