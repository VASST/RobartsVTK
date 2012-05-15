#ifndef ULTERIUS_DEF_H
#define ULTERIUS_DEF_H

////////////////////////////////////////////////////////////////////////////////
/// Callback for when a non-blocking function has completed
////////////////////////////////////////////////////////////////////////////////
typedef bool (*ULTERIUS_CALLBACK)(void * data, int type, int sz, bool cine, int frmnum);
typedef bool (*ULTERIUS_PARAM_CALLBACK)(void * paramID, int ptX, int ptY);

////////////////////////////////////////////////////////////////////////////////
/// Ulterius data types.
////////////////////////////////////////////////////////////////////////////////
enum uData
{
    udtScreen = 0x00000001,
    udtBPre = 0x00000002,
    udtBPost = 0x00000004,
    udtBPost32 = 0x00000008,
    udtRF = 0x00000010,
    udtMPre = 0x00000020,
    udtMPost = 0x00000040,
    udtPWRF = 0x00000080,
    udtPWSpectrum = 0x00000100,
    udtColorRF = 0x00000200,
    udtColorCombined = 0x00000400,
    udtColorVelocityVariance = 0x00000800,
    udtElastoCombined = 0x00002000,
    udtElastoOverlay = 0x00004000,
    udtElastoPre = 0x00008000,
    udtECG = 0x00010000,
    udtPNG = 0x10000000
};

////////////////////////////////////////////////////////////////////////////////
/// Ulterius region of interest descriptor.
////////////////////////////////////////////////////////////////////////////////
class uROI
{
public:
    /// roi - upper left (x)
    int ulx;
    /// roi - upper left (y)
    int uly;
    /// roi - upper right (x)
    int urx;
    /// roi - upper right (y)
    int ury;
    /// roi - bottom right (x)
    int brx;
    /// roi - bottom right (y)
    int bry;
    /// roi - bottom left (x)
    int blx;
    /// roi - bottom left (y)
    int bly;
};

////////////////////////////////////////////////////////////////////////////////
/// Ulterius data descriptor.
////////////////////////////////////////////////////////////////////////////////
class uDataDesc
{
public:
    /// data type
    uData type;
    /// data width
    int w;
    /// data height
    int h;
    /// data sample size in bits
    int ss;
    /// roi of data
    uROI roi;
};

////////////////////////////////////////////////////////////////////////////////
/// Ulterius parameter definition.
////////////////////////////////////////////////////////////////////////////////
class uParam
{
public:
    /// Unique identifier
    char id[80];
    /// Parameter name
    char name[80];
    /// Parameter type
    int type;
    /// Units of parameter value
    int unit;
    /// Optimimzation group of parameter
    int optgroup;
};

////////////////////////////////////////////////////////////////////////////////
/// Ulterius rectangle definition.
////////////////////////////////////////////////////////////////////////////////
class uRect
{
public:
    /// Left coordinate
    int left;
    /// Top coordinate
    int top;
    /// Right coordinate
    int right;
    /// Bottom coordinate
    int bottom;
};

////////////////////////////////////////////////////////////////////////////////
/// Ulterius curve definition.
////////////////////////////////////////////////////////////////////////////////
class uCurve
{
public:
    /// horizontal position of top point
    int t;
    /// horizontal position of middle point
    int m;
    /// horizontal position of bottom point
    int b;
    /// vertical position of middle point
    int vm;
};

////////////////////////////////////////////////////////////////////////////////
/// Ulterius TGC definition.
////////////////////////////////////////////////////////////////////////////////
class uTGC
{
public:
    /// First TGC slider value
    int v1;
    /// Second TGC slider value
    int v2;
    /// Third TGC slider value
    int v3;
    /// Fourth TGC slider value
    int v4;
    /// Fifth TGC slider value
    int v5;
    /// Sixth TGC slider value
    int v6;
    /// Seventh TGC slider value
    int v7;
    /// Eighth TGC slider value
    int v8;
};

////////////////////////////////////////////////////////////////////////////////
/// Ulterius data file header.
////////////////////////////////////////////////////////////////////////////////
class uFileHeader
{
public:
    /// data type - data types can also be determined by file extensions
    int type;
    /// number of frames in file
    int frames;
    /// width - number of vectors for raw data, image width for processed data
    int w;
    /// height - number of samples for raw data, image height for processed data
    int h;
    /// data sample size in bits
    int ss;
    /// roi - upper left (x)
    int ulx;
    /// roi - upper left (y)
    int uly;
    /// roi - upper right (x)
    int urx;
    /// roi - upper right (y)
    int ury;
    /// roi - bottom right (x)
    int brx;
    /// roi - bottom right (y)
    int bry;
    /// roi - bottom left (x)
    int blx;
    /// roi - bottom left (y)
    int bly;
    /// probe identifier - additional probe information can be found using this id
    int probe;
    /// transmit frequency
    int txf;
    /// sampling frequency
    int sf;
    /// data rate - frame rate or pulse repetition period in Doppler modes
    int dr;
    /// line density - can be used to calculate element spacing if pitch and native # elements is known
    int ld;
    /// extra information - ensemble for color RF
    int extra;
};

enum uVariableType
{
    uTypeInteger = 0,
    uTypeFloat = 1,
    uTypeString = 2,
    uTypeGainCurve = 3,
    uTypeRectangle = 4,
    uTypeCurve = 5,
    uTypeColor = 6,
    uTypeBoolean = 7
};

#endif
