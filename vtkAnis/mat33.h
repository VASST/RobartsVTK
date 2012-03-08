
#ifndef __mat33_h
#define __mat33_h

#include <vtkMatrix4x4.h>
#include <math.h>

struct mat33;
mat33 log(mat33& in);
mat33 exp(mat33& in);
mat33 pow(mat33& in, const double exponent);

struct mat33
{
    double m[3][3];

    mat33()
    {
        Identity();
    }

    mat33(const vtkMatrix4x4 *mat)
    {
        FromMatrix4x4(mat);
    }

    void FromMatrix4x4(const vtkMatrix4x4 *mat)
    {
        int i, j;

        for(i = 0; i < 3; i++)
            for(j = 0; j < 3; j++)
              m[i][j] = mat->GetElement(i, j);
    }

    void ToMatrix4x4(vtkMatrix4x4 *mat)
    {
        int i, j;

        for(i = 0; i < 3; i++)
            for(j = 0; j < 3; j++)
              mat->SetElement(i, j, m[i][j]);
    }

    void Zero(void)
    {
        int i, j;

        for(i = 0; i < 3; i++)
            for(j = 0; j < 3; j++)
                m[i][j] = 0.0;
    }

    void Identity(void)
    {
        int i, j;

        for(i = 0; i < 3; i++)
            for(j = 0; j < 3; j++)
                if(i == j) m[i][j] = 1.0;
                else m[i][j] = 0.0;
    }

    mat33 Transpose(void)
    {
        int i, j;
        mat33 result;

        for(i = 0; i < 3; i++)
            for(j = 0; j < 3; j++)
                result.m[i][j] = m[j][i];

        return result;
    }

    mat33 operator+(const mat33& rhs)
    {
        int i, j;
        mat33 result;

        for(i = 0; i < 3; i++)
            for(j = 0; j < 3; j++)
                result.m[i][j] = m[i][j] + rhs.m[i][j];

        return result;
    }

    mat33 operator-(const mat33& rhs)
    {
        int i, j;
        mat33 result;

        for(i = 0; i < 3; i++)
            for(j = 0; j < 3; j++)
                result.m[i][j] = m[i][j] - rhs.m[i][j];

        return result;
    }

    mat33 operator*(const mat33& rhs)
    {
        int i, j, k;
        mat33 result;

        for(i = 0; i < 3; i++)
            for(j = 0; j < 3; j++)
            {
                result.m[i][j] = 0.0;

                for(k = 0; k < 3; k++)
                    result.m[i][j] += m[i][k] * rhs.m[k][j];
            }

        return result;
    }

    double Trace(void)
    {
        int i;
        double result = 0.0;

        for(i = 0; i < 3; i++)
            result += m[i][i] * m[i][i];

        return result;
    }

    double Norm(void)
    {
        return sqrt((Transpose() * (*this)).Trace());
    }

    void Average(int matNo, mat33& mat)
    {
        (*this) = mat * pow(mat.Transpose() * (*this), double(matNo) / double(matNo+1));
    }
};
    
mat33 operator*(const double& sc, const mat33& in)
{
    mat33 result;

    int i, j;

    for(i = 0; i < 3; i++)
        for(j = 0; j < 3; j++)
            result.m[i][j] = sc * result.m[i][j];

    return result;
}

mat33 exp(mat33& in)
{
    double a;
    mat33 result;

    // Initialize to the result
    result.Identity();

    // Compute the value of a
    a = sqrt(0.5 * in.Norm());
    if(fabs(a) < 1e-6)
    {
        return result;
    }

    // Compute the result
    result = result + (sin(a) / a) * in + ((1.0 - cos(a)) / (a * a)) * in * in;

    return result;
}

mat33 log(mat33& in)
{
    double theta;
    mat33 result;

    // Compute theta
    theta = acos(0.5 * (in.Trace() - 1.0));

    // Compute result
    if(fabs(theta) < 1e-6)
    {
        result.Zero();
    }
    else
    {
        result = (theta / (2.0 * sin(theta))) * (in - in.Transpose());
    }

    return result;
}

mat33 pow(mat33& in, const double exponent)
{
    return exp( exponent * log(in) );
}

#endif