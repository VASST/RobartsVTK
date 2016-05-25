#include "vtkRetinex.h"
#include "vtkObjectFactory.h"
#include "vtkMath.h"
#include "fftw3.h"

//----------------------------------------------------------------------------

vtkStandardNewMacro(vtkRetinex);

//----------------------------------------------------------------------------
vtkRetinex::vtkRetinex(const std::vector< double > &n_scales)
{
  SetScaleValues(n_scales);
}

//----------------------------------------------------------------------------
vtkRetinex::~vtkRetinex()
{

}

//----------------------------------------------------------------------------
void vtkRetinex::PrintSelf(ostream& os, vtkIndent indent)
{

}

//----------------------------------------------------------------------------
const std::vector< double >& vtkRetinex::GetScaleValues() const
{
  return ScaleValues;
}

//----------------------------------------------------------------------------
void vtkRetinex::SetScaleValues(const std::vector< double >& val)
{
  ScaleValues = val;
}

//----------------------------------------------------------------------------
double * vtkRetinex::GaussianConvolution(double *img_data, double *out_data, size_t w, size_t h, double scale)
{
  double *tmp_out;
  tmp_out = (double*) fftw_malloc(sizeof(double) * ( w * h ) );

  /// the fast Fourier transform
  fftw_plan pt;
  pt = fftw_plan_r2r_2d( (int)h, (int)w, img_data, tmp_out, FFTW_REDFT10, FFTW_REDFT10,FFTW_ESTIMATE);
  fftw_execute( pt );
  fftw_destroy_plan( pt );

  /// some parameters for the Gaussian convolution
  double sigma= scale * scale  / 2.;
  double w_norm = vtkMath::Pi() / ( double )w;
  double h_norm = vtkMath::Pi() / ( double )h;
  w_norm *= w_norm;
  h_norm *= h_norm;
  int img_size = (int)w * (int)h;
  int img_quartet = 4 * img_size;

  int tmp;
  for( int j = 0; j < (int)h; j++ )
  {
    tmp = j*(int)w;
    for( int i = 0; i < (int)w; i++ )
    {
      tmp_out[ i + tmp ]*=exp( ( double )( -sigma ) * ( w_norm*i*i + h_norm*j*j ) );
    }
  }

  /// the Inverse fast Fourier transform
  pt = fftw_plan_r2r_2d( (int)h, (int)w, tmp_out, out_data, FFTW_REDFT01, FFTW_REDFT01, FFTW_ESTIMATE );
  fftw_execute( pt );

  for( int k = 0; k < img_size; k++ )
  {
    out_data[ k ] /= ( double )img_quartet;
  }

  fftw_destroy_plan( pt );
  fftw_free( tmp_out );

  return out_data;
}

//----------------------------------------------------------------------------
double* vtkRetinex::MultiscaleRetinex(double *img_data, double *out_data, int w, int h, double omega)
{
  double *out_convolution;
  int img_size = w*h;
  out_convolution = (double*) malloc( img_size*sizeof(double) );
  for( int j = 0; j < img_size; j++ )
  {
    out_data[ j ]=0.0;
  }

  /// retinex output
  for( int k = 0; k < GetScaleValues().size( ); k++ )
  {
    GaussianConvolution( img_data, out_convolution, w, h, GetScaleValues()[k] );
    for( int i = 0; i < img_size; i++ )
    {
      out_data[i] += omega * ( log( img_data[i] ) - log(out_convolution[i]) );
    }
  }

  free( out_convolution );

  return out_data;
}

//----------------------------------------------------------------------------
double* vtkRetinex::ColorRestore(double *img_data, double *out_data, int w, int h, double *gray)
{
  double G;
  int img_size = w*h;
  for( int i = 0; i < img_size; i++ )
  {
    G = log( 3 * gray[i] );
    out_data[i] *= ( log( 125. * img_data[i] ) - G );
  }

  return out_data;
}

//----------------------------------------------------------------------------
double* vtkRetinex::HistogramEqualizer(double *img_data, double *out_data, int w, int h, float p_left, float p_right)
{
  double *sort_data;
  int img_size = w*h;
  sort_data= (double*) malloc(img_size*sizeof(double));

  std::memcpy( sort_data, img_data, img_size*sizeof(double) );
  std::qsort( sort_data, img_size, sizeof sort_data[0], &CompareValues);

  int per_left  = (int)( p_left * img_size / 100 );
  double _min_  = sort_data[per_left];
  int per_right = (int)(p_right * img_size / 100 );
  double _max_  = sort_data[ img_size - per_right - 1 ];

  double se = 255.0 / ( _max_ - _min_ );
  if( _max_  <= _min_ )
  {
    for( int i = 0; i < img_size; i++ )
    {
      out_data[ i ] = _max_;
    }
  }
  else
  {
    for( int k = 0; k < img_size; k++ )
    {
      if( img_data[k] < _min_ )
      {
        out_data[k] = 0.0;
      }
      else if( img_data[k] > _max_ )
      {
        out_data[k]=255.0;
      }
      else
      {
        out_data[k] = se * ( img_data[k] - _min_ );
      }
    }
  }

  return out_data;
}