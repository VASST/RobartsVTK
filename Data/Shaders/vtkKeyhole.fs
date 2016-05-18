// Fragment shader used by vtkKeyholePass render pass.
// (c) Uditha Jayarathne <ujayarat@robarts.ca>
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

float calc_mask(vec2, float, float, float, float, float, int);

void main(void)
{
  vec4 volume = texture2D(_volume, tcoordVC);
  vec4 foreground = texture2D(_foreground, tcoordVC);
  vec4 foreground_grad = texture2D(_foreground_grad, tcoordVC);
  vec3 inside, outside, surface;

  // Convert the edge map to gray-scale.
  float gray = 0.299*foreground_grad.r + 0.58*foreground_grad.g + 0.114*foreground_grad.b;
 
  if( use_mask_texture == 0)
  {
      // Compute the keyhole mask based on x0,y0, radius and gamma
      float mask = calc_mask( tcoordVC, x0, y0, radius, aspect_ratio, gamma, use_hard_edges);
      
      // keyhole effect
      inside  = (1-mask)*volume.rgb;
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

// Calculates the mask
float calc_mask(vec2 tcoord, float x0, float y0, float r, float ratio, float gamma, int t)
{
    float x = tcoord.x - x0;
    float y = tcoord.y - y0;

    if( pow(sqrt(ratio*x*x + y*y/ratio)/r, gamma) <= 1.0)
    {
        if( t == 1 )
            return 0;
        else
        {  
            float base_opacity = 0.08; // base opacity value
            return (base_opacity + pow(sqrt(ratio*x*x + y*y/ratio)/r, gamma));
        }
    }
    else 
        return 1;
}