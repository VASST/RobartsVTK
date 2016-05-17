// Fragment shader used by vtkKeyholePass render pass.
#version 440 core

in vec2 tcoordVC;
layout(location = 0) out vec4 color;

uniform sampler2D _volume, _mask, _foreground, _foreground_grad;
uniform float th;

// Parameters for keyhole
uniform float x0,y0; // Center of the keyhole. Eventually this will be in 3D. This is normalized. 
uniform float radius; // Radius of the keyhole. This is normalized. 
uniform float gamma; // gamma variable to control distance fall off
uniform int use_mask_texture, use_hard_edges; 
uniform float aspect_ratio;

void main(void)
{
  vec4 volume = texture2D(_volume, tcoordVC);
  vec4 foreground = texture2D(_foreground, tcoordVC);
  vec4 foreground_grad = texture2D(_foreground_grad, tcoordVC);
  vec3 inside, outside, surface;

  // Convert the edge map to gray-scale.
  float gray = 0.299*foreground_grad.r + 0.58*foreground_grad.g + 0.114*foreground_grad.b;
  float n;
  // Threshold the gray image
  if( gray > th )
        n = 1.0;
  else
     n = 0.0;

  //--------------------------------------------------------------------------------------------
  // Compute the keyhole mask based on x0,y0, radius and gamma
  float x    = tcoordVC.x-x0;
  float y    = tcoordVC.y-y0;
  float mask  = 1;

  if( pow(sqrt(aspect_ratio*x*x + y*y/aspect_ratio)/radius, gamma) < 1.0)
  {
    if( use_hard_edges == 1 )
        mask = 0;
    else
        mask = pow(sqrt(aspect_ratio*x*x + y*y/aspect_ratio)/radius, gamma);
  }  
  //--------------------------------------------------------------------------------------------

  if( use_mask_texture == 0)
  {
      // keyhole effect
      inside = (1-mask)*volume.rgb;
      outside = mask*foreground.rgb;
      surface = (1-mask)*foreground_grad.rgb;
  }
  else
  {
      float m = texture2D(_mask, tcoordVC).r;
      // keyhole effect
      inside = (1-m)*volume.rgb;
      outside = m*foreground.rgb;
      surface = (1-m)*foreground_grad.rgb;
  }

  // Blend textures to get the final effect. 
  color.rgb = inside + surface + outside;
  color.a = 1.0;
}