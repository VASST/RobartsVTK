// Fragment shader used by vtkKeyholePass render pass.
// (c) Uditha Jayarathne <ujayarat@robarts.ca>
#version 440 core

in vec2 tcoordVC;
layout(location = 0) out vec4 color;

uniform sampler2D _volume, _mask, _foreground, _foreground_grad;

// Parameters for keyhole
uniform float x0,y0; // Center of the keyhole. Eventually this will be in 3D. This is normalized. 
uniform float radius; // Radius of the keyhole. This is normalized. 
uniform float gamma; // gamma variable to control distance fall off
uniform int use_mask_texture, use_hard_edges, mode; 
uniform float aspect_ratio, alpha;
uniform float d1;

float calc_mask(vec2, float, float, float, float, float, float, int);

void main(void)
{
  vec4 volume = texture2D(_volume, tcoordVC);
  vec4 foreground = texture2D(_foreground, tcoordVC);
  vec4 foreground_grad = texture2D(_foreground_grad, tcoordVC);
  vec3 inside, outside, surface;

  // Convert the edge map to gray-scale.
  float gray = 0.299*foreground_grad.r + 0.58*foreground_grad.g + 0.114*foreground_grad.b;
  float factor = 5;

  if(mode == 0) // Just background image
  {
      color.rgb = foreground.rgb;
      color.a = 1.0;
  }
  else if(mode == 1) // Alpha blending
  {
      if( volume.rgb == vec3(0, 0, 0) ) // Don't do alpha-blending for the background
      {
          color.rgb = foreground.rgb;
          color.a = 1.0;
      }
      else
      {
          color.rgb = alpha*volume.rgb + (1-alpha)*foreground.rgb;
          color.a = 1.0;
      }
  }
  else if(mode == 2)  // Additive blending
  {  
      color.rgb = volume.rgb + foreground.rgb;
      color.a = 1.0;
  }
  else if(mode == 3) // Keyhole blending
  {   
	  float mask;
      if( use_mask_texture == 0)
      {
          // Compute the keyhole mask based on x0,y0, radius and gamma
          mask = calc_mask( tcoordVC, x0, y0, d1, radius, aspect_ratio, gamma, use_hard_edges);
      }
      else
      {
         mask = texture2D(_mask, tcoordVC).r;
      }
	  
	  float opacity = gray*factor + mask;
	  surface = (1-mask)*vec3(gray*factor, gray*factor, gray*factor) + mask*foreground.rgb;
      // Blend textures to get the final effect. 

      color.rgb = max(volume.rgb*(1-opacity), surface);
      color.a = 1.0;
  }
}

// Calculates the mask
float calc_mask(vec2 tcoord, float x0, float y0, float d1, float r, float ratio, float gamma, int t)
{
    float x = tcoord.x - x0;
    float y = tcoord.y - y0;

	float distance = sqrt(ratio*x*x + y*y/ratio);
	
	if(distance <= d1)
	{
		return 0;
	}
	else if (d1<distance && distance <=r)
	{
		float val = pow((distance-d1)/(r-d1),2);
		return 1-exp(-val/0.25);
	}
	else
	{
		return 1.0;
	}
}