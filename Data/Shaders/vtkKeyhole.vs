// The following line handles system declarations such as
// default precisions, or defining precisions to null
//VTK::System::Dec

attribute vec4 vertexMC;
attribute vec2 tcoordMC;

varying vec2 tcoordVC;

void main()
{
  tcoordVC = tcoordMC;
  gl_Position = vertexMC;
}