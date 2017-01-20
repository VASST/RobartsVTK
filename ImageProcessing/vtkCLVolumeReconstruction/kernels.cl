#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

//#define PRINT
#define PRINT_RES0 3
#define PRINT_RES1 30
#define PRINT_RES2 6000

#define BSCAN_WINDOW 4 // must be >= 4 if PT
#define PT_OR_DW 1 // 0=PT (Probe Trajectory), 1=DW (Distance Weighted)

#define COMPOUND_AVG 0
#define COMPOUND_MAX 1
#define COMPOUND_IFEMPTY 2
#define COMPOUND_OVERWRITE 3

#define COMPOUND_METHOD COMPOUND_AVG

#define pixel_pos_c(i,c) (pixel_pos[(i)*3 + (c)])
#define pos_matrix_a(x,y) (pos_matrix[y*4 + x])
#define inrange(x,a,b) ((x) >= (a) && (x) <= (b))
#define volume_a(x,y,z) (volume[(x) + (y)*volume_w + (z)*volume_w*volume_h])

__kernel void round_off_translate(__global float * pixel_pos,
																	float volume_spacing,
																	int mask_size,
																	float origo_x,
																	float origo_y,
																	float origo_z) {
	int n = get_global_id(0);
	if (n >= mask_size) return;

	pixel_pos_c(n,0) = (int)(pixel_pos_c(n,0)/volume_spacing) - origo_x;
	pixel_pos_c(n,1) = (int)(pixel_pos_c(n,1)/volume_spacing) - origo_y;
	pixel_pos_c(n,2) = (int)(pixel_pos_c(n,2)/volume_spacing) - origo_z;
}

__kernel void fill_volume(__global float * pixel_pos,
													__global unsigned char * pixel_ill,
													int mask_size,
													__global unsigned char * volume,
													int volume_n,
													int volume_h,
													int volume_w) {

	int n = get_global_id(0);
	if (n >= mask_size) return;

	int x = pixel_pos_c(n,0);
	int y = pixel_pos_c(n,1);
	int z = pixel_pos_c(n,2);
	if (inrange(x,0,volume_w) && inrange(y,0,volume_h) && inrange(z,0,volume_n))
		volume_a(x,y,z) = pixel_ill[n];
}

__kernel void fill_holes(__global float * pixel_pos,
												 __global char * hole_ill,
												 int mask_size,
												 __global unsigned char * volume,
												 int volume_n,
												 int volume_h,
												 int volume_w) {
	// Assumes no black ultrasound input data

	int n = get_global_id(0);
	if (n >= mask_size) return;

	int x = pixel_pos_c(n,0);
	int y = pixel_pos_c(n,1);
	int z = pixel_pos_c(n,2);

	#define kernel_size 5
	#define half_kernel (kernel_size/2)
	//#define cutoff (kernel_size*kernel_size*kernel_size/3)

	//for (int a = -half_kernel; a <= half_kernel; a++)
		//for (int b = -half_kernel; b <= half_kernel; b++)
			//for (int c = -half_kernel; c <= half_kernel; c++)
	for (int h = 0; h < 5; h++) {
		int a = -(1+h); int b = -(1+h); int c = -(1+h); // TODO: Set to direction of probe trajectory
		//int a = (1+h); int b = (1+h); int c = (1+h); // TODO: Set to direction of probe trajectory
		if (inrange(x+a,0,volume_w) && inrange(y+b,0,volume_h) && inrange(z+c,0,volume_n)) {
			hole_ill[sizeof(unsigned char)*mask_size*h + n] = 0;
			if (volume_a(x+a,y+b,z+c) == 0) {
				int sum = 0;
				int sum_counter = 0;
				for (int i = -half_kernel; i <= half_kernel; i++) {
					for (int j = -half_kernel; j <= half_kernel; j++) {
						for (int k = -half_kernel; k <= half_kernel; k++) {
							if (inrange(x+a+i,0,volume_w) && inrange(y+b+j,0,volume_h) && inrange(z+c+k,0,volume_n)) {
								if (volume_a(x+a+i,y+b+j,z+c+k) != 0) {
									sum += volume_a(x+a+i,y+b+j,z+c+k);
									sum_counter++;
								}
							}
						}
					}
				}
				//if (sum_counter > cutoff) {
					volume_a(x+a,y+b,z+c) = sum/(float)sum_counter;
					hole_ill[sizeof(unsigned char)*mask_size*h + n] = sum/(float)sum_counter;
				//}
			}
		}	
	}
}

__kernel void transform(__global float * pixel_pos,
												__global float * pos_matrix, // Access violation when set to __constant
												int mask_size) {
	
	int n = get_global_id(0);
	if (n >= mask_size) return;

	float sum0, sum1, sum2;
	for (int y = 0; y < 3; y++) {
		float sum = 0;
		for (int x = 0; x < 3; x++) 
			sum += pos_matrix_a(x,y)*pixel_pos_c(n,x);
		sum += pos_matrix_a(3,y);
		if (y==0) sum0=sum; else if (y==1) sum1=sum; else sum2=sum;
	}
	pixel_pos_c(n,0) = sum0; pixel_pos_c(n,1) = sum1; pixel_pos_c(n,2) = sum2;
}

//#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable // warning: unknown action for '#pragma OPENCL' - ignored
//#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable // warning: unknown action for '#pragma OPENCL' - ignored

/*__kernel void fill_pixel_ill_pos(__global unsigned char * bscan,
																 __constant unsigned char * mask,
																 __global float * pixel_pos,
																 __global unsigned char * pixel_ill,
																 __constant int mask_size,
																 __constant int bscan_h,
																 __constant int bscan_w,
																 __constant float bscan_spacing_x,
																 __constant float bscan_spacing_y,
																 __global int * global_mask_counter) { // arg not used

	int mask_counter = 0;
	for (int y = 0; y < bscan_h; y++) {
		for (int x = 0; x < bscan_w; x++) {
			char foo = 1 << (y*bscan_w + x)%8;
			char mask_bit = mask[(x + y*bscan_w)/8] & foo;
			if (mask_bit != 0) {
				pixel_ill[mask_counter] = bscan[x + y*bscan_w];
				pixel_pos[mask_counter*3+0] = 0;//atom_inc(global_mask_counter);
				pixel_pos[mask_counter*3+1] = x*bscan_spacing_x;
				pixel_pos[mask_counter*3+2] = y*bscan_spacing_y;
				mask_counter++;
			}
		}
	}
}*/

__kernel void build_ray_dirs(__global unsigned char * volume,
														 int volume_w,
														 int volume_h,
														 int volume_n,
														 __global float4 * ray_dirs,
														 int bitmap_w,
														 int bitmap_h,
														 float4 camera_pos,
														 float4 camera_lookat,
														 __global float * printings) {
	int n = get_global_id(0);
	if (n >= bitmap_w*bitmap_h) return;

	int ray_x = n%bitmap_w;
	int ray_y = n/bitmap_w;
	
	float4 camera_forward = normalize(camera_lookat - camera_pos);
	float4 temp_up = {0, 1, 0, 0};
	float4 camera_right = normalize(cross(temp_up, camera_forward));
	float4 camera_up = normalize(cross(camera_right, camera_forward));

	float fov_hor = 45/2;
	float fov_ver = fov_hor*bitmap_h/(float)bitmap_w;
	fov_hor = fov_hor/180.0f*3.14f;
	fov_ver = fov_ver/180.0f*3.14f;
	
	float4 step_forward = camera_forward;
	//float temp = 1/cos(fov_hor);
	//temp *= sin(fov_hor);
	float temp = (ray_x-bitmap_w/2)/(float)(bitmap_w/2);
	float4 step_right = temp * fov_hor * camera_right;
	//temp = 1/cos(fov_ver);
	//temp *= sin(fov_ver);
	temp = (ray_y-bitmap_h/2)/(float)(bitmap_h/2);
	float4 step_up = temp * fov_ver * camera_up;
	float4 ray_dir = normalize(step_forward + step_right + step_up);
	
	ray_dirs[n] = ray_dir;

	//printings[n*3+0] = ray_dir.x;
	//printings[n*3+1] = ray_dir.y;
	//printings[n*3+2] = ray_dir.z;
}

__kernel void cast_rays(__global unsigned char * volume,
												int volume_w,
												int volume_h,
												int volume_n,
												__global float4 * ray_dirs,
												__global unsigned char * bitmap,
												int bitmap_w,
												int bitmap_h,
												float4 camera_pos,
												float4 camera_lookat,
												__global float * printings) {
	int n = get_global_id(0);
	if (n >= bitmap_w*bitmap_h) return;

	int ray_x = n%bitmap_w;
	int ray_y = n/bitmap_w;
	
	float4 ray_dir = ray_dirs[n];

	#define step_size 1.0f
	#define transparent_level 54
	#define transparency_ajustment 0.4f
	#define ray_strength_cutoff (255/10.0f)

	unsigned char accum = 0;
	float t = 0;
	
	float4 volume_0 = {0, 0, 0, 0};
	float4 volume_1 = {volume_w-1, volume_h-1, volume_n-1, 0};

	float4 foo0 = (volume_0 - camera_pos)/ray_dir;
	float4 foo1 = (volume_1 - camera_pos)/ray_dir;
	foo0 = min(foo0, foo1);
	t = max(foo0.x, max(foo0.y, foo0.z)) + 2;

	float4 t_pos = camera_pos + t*ray_dir;

	if (t_pos.x > 0 && t_pos.x < volume_w-1 &&
			t_pos.y > 0 && t_pos.y < volume_h-1 &&
			t_pos.z > 0 && t_pos.z < volume_n-1) {
		float ray_strength = 255;
		unsigned char voxel;
		float transparency;
		while(true) {
			if (t_pos.x < 0 || t_pos.x > volume_w-1 ||
					t_pos.y < 0 || t_pos.y > volume_h-1 ||
					t_pos.z < 0 || t_pos.z > volume_n-1) {
				break;				
			}
			if (ray_strength < ray_strength_cutoff) break;

			voxel = volume_a((int)t_pos.x, (int)t_pos.y, (int)t_pos.z);
			if (voxel < transparent_level) voxel = 0;

			transparency = min((1 - voxel/255.0f) + transparency_ajustment, 1.0f);
			
			accum += ray_strength * (1-transparency);
			ray_strength *= transparency;
			
			t += step_size;
			t_pos = camera_pos + t*ray_dir;
		}
	} else { // miss
		accum = ((ray_x+ray_y)%2)*150;
	}

	bitmap[n] = accum;
	
	//printings[n*3+0] = ray_dir.x;
	//printings[n*3+1] = ray_dir.y;
	//printings[n*3+2] = ray_dir.z;
	//printings[n] = t;
}

#define NULL 0
#define distance_pp(v, plane) (plane.x*v.x + plane.y*v.y + plane.z*v.z + plane.w)/sqrt(plane.x*plane.x + plane.y*plane.y + plane.z*plane.z)
#define bscans_queue_a(n, x, y) bscans_queue[(n)*bscan_w*bscan_h + (y)*bscan_w + (x)]

typedef struct {
  float4 corner0;
  float4 cornerx;
  float4 cornery;
} plane_pts;

__kernel void adv_fill_voxels(__global float4 * intersections, 
															__global unsigned char * volume,
															float volume_spacing, 
															int volume_w, 
															int volume_h, 
															int volume_n, 
															__global float4 * x_vector_queue, 
															__global float4 * y_vector_queue, 
															__global plane_pts * plane_points_queue,
															__global float4 * bscan_plane_equation_queue,
															float bscan_spacing_x,
															float bscan_spacing_y,
															int bscan_w,
															int bscan_h,
															__global unsigned char * mask,
															__global unsigned char * bscans_queue,
															__global float * bscan_timetags_queue,
															int intersection_counter) {

	int i = get_global_id(0);

	if (i >= intersection_counter) return;

	float4 intrs0 = intersections[i*2 + 0]/volume_spacing;
	float4 intrs1 = intersections[i*2 + 1]/volume_spacing;

	int x0 = min(intrs0.x,intrs1.x);
	int x1 = max(x0+1.0f, max(intrs0.x,intrs1.x));
	int y0 = min(intrs0.y,intrs1.y);
	int y1 = max(y0+1.0f, max(intrs0.y,intrs1.y));
	int z0 = min(intrs0.z,intrs1.z);
	int z1 = max(z0+1.0f, max(intrs0.z,intrs1.z));

	int safety = 0;
	for (int z = z0; z <= z1; z++) {
		for (int y = y0; y <= y1; y++) {
			for (int x = x0; x <= x1; x++) {
				float4 voxel_coord = {x*volume_spacing,y*volume_spacing,z*volume_spacing,0};
				if (inrange(x, 0, volume_w) && inrange(y, 0, volume_h) && inrange(z, 0, volume_n)) {
					float contribution = 0;
					if (PT_OR_DW) { // DW
						float dists[BSCAN_WINDOW];
						unsigned char bilinears[BSCAN_WINDOW];
						bool valid = true;
						float G = 0;
						for (int n = 0; n < BSCAN_WINDOW; n++) {
							int q_idx = n;

							float4 normal = { bscan_plane_equation_queue[q_idx].x, bscan_plane_equation_queue[q_idx].y, bscan_plane_equation_queue[q_idx].z, 0 };

								float dist0 = fabs(distance_pp(voxel_coord, bscan_plane_equation_queue[q_idx]));
							float4 p0 = voxel_coord + -dist0*normal - plane_points_queue[q_idx].corner0;
								float px0 = dot(p0, x_vector_queue[q_idx]) / bscan_spacing_x;
							float py0 = dot(p0, y_vector_queue[q_idx]) / bscan_spacing_y;
							float xa = px0 - floor(px0);
							float ya = py0 - floor(py0);
							int xa0 = (int)px0;
							int ya0 = (int)py0;

							bool valid0 = false;

							if (inrange(xa0, 0, bscan_w) && inrange(ya0, 0, bscan_h) && inrange(xa0 + 1, 0, bscan_w) && inrange(ya0 + 1, 0, bscan_h)) {
								if (mask[xa0 + ya0*bscan_w] != 0 && mask[xa0 + 1 + (ya0 + 1)*bscan_w] != 0 && mask[xa0 + 1 + ya0*bscan_w] != 0 && mask[xa0 + (ya0 + 1)*bscan_w] != 0) {
									bilinears[n] = bscans_queue_a(q_idx, xa0, ya0)*(1 - xa)*(1 - ya) + bscans_queue_a(q_idx, xa0 + 1, ya0)*xa*(1 - ya) + bscans_queue_a(q_idx, xa0, ya0 + 1)*(1 - xa)*ya + bscans_queue_a(q_idx, xa0 + 1, ya0 + 1)*xa*ya;
									valid0 = true;
								}
							}

							valid &= valid0;
							dists[n] = dist0;
							if (dist0 == 0)
								continue;

							G += 1 / dists[n];
							contribution += bilinears[n] / dists[n];
						}

						if (!valid) continue;

						if (G != 0)
							contribution /= G;
						
					} else { // PT
						// Find virtual plane time stamp:
						float dists[4];
						bool valid = true;
						for (int n = 0; n < 4; n++) {
							int q_idx = BSCAN_WINDOW/2-2+n;

							float4 normal = {bscan_plane_equation_queue[q_idx].x, bscan_plane_equation_queue[q_idx].y, bscan_plane_equation_queue[q_idx].z, 0};

							float dist0 = fabs(distance_pp(voxel_coord, bscan_plane_equation_queue[q_idx]));
							float4 p0 = voxel_coord + -dist0*normal - plane_points_queue[q_idx].corner0;
							float px0 = dot(p0, x_vector_queue[q_idx])/bscan_spacing_x;
							float py0 = dot(p0, y_vector_queue[q_idx])/bscan_spacing_y;
							float xa = px0-floor(px0);
							float ya = py0-floor(py0);
							int xa0 = (int)px0;
							int ya0 = (int)py0;

							bool valid0 = false;
							float bilinear0;

							if (inrange(xa0, 0, bscan_w) && inrange(ya0, 0, bscan_h) && inrange(xa0+1, 0, bscan_w) && inrange(ya0+1, 0, bscan_h))
								if (mask[xa0 + ya0*bscan_w] != 0 && mask[xa0+1 + (ya0+1)*bscan_w] != 0 && mask[xa0+1 + ya0*bscan_w] != 0 && mask[xa0 + (ya0+1)*bscan_w] != 0)
									valid0 = true;

							dists[n] = dist0;

							valid &= valid0;
						}
						if (!valid) continue;
						float G = dists[1] + dists[2];
						float t = dists[2]/G*bscan_timetags_queue[BSCAN_WINDOW/2-1] + dists[1]/G*bscan_timetags_queue[BSCAN_WINDOW/2];

						// Cubic interpolate 4 bscan plane equations, corner0s and x- and y-vectors:
						float4 v_plane_eq = {0,0,0,0};
						float4 v_corner0 = {0,0,0,0};
						float4 v_x_vector = {0,0,0,0};
						float4 v_y_vector = {0,0,0,0};
						for (int k = 0; k < 4; k++) {
							int q_idx = BSCAN_WINDOW/2-2+k;
							float phi = 0;
							float a = -1/2.0f;
							float abs_t = fabs((t-bscan_timetags_queue[q_idx]))/(bscan_timetags_queue[1]-bscan_timetags_queue[0]);
							if (inrange(abs_t, 0, 1))
								phi = (a+2)*abs_t*abs_t*abs_t - (a+3)*abs_t*abs_t + 1;
							else if (inrange(abs_t, 1, 2))
								phi = a*abs_t*abs_t*abs_t - 5*a*abs_t*abs_t + 8*a*abs_t - 4*a;
							v_plane_eq += bscan_plane_equation_queue[q_idx]*phi;
							v_corner0 += phi*plane_points_queue[q_idx].corner0;
							v_x_vector += phi*x_vector_queue[q_idx];
							v_y_vector += phi*y_vector_queue[q_idx];
						}

						// Find 2D coordinates on virtual plane:
						float4 p0 = voxel_coord - v_corner0;
						float px0 = dot(p0, v_x_vector)/bscan_spacing_x;
						float py0 = dot(p0, v_y_vector)/bscan_spacing_y;
						float xa = px0-floor(px0);
						float ya = py0-floor(py0);
						int xa0 = (int)px0;
						int ya0 = (int)py0;

						// Distance weight 4 bilinears:
						float F = 0;
						for (int n = 0; n < 4; n++) {
							int q_idx = BSCAN_WINDOW/2-2+n;
							float bilinear0 = bscans_queue_a(q_idx,xa0,ya0)*(1-xa)*(1-ya) + bscans_queue_a(q_idx,xa0+1,ya0)*xa*(1-ya) + bscans_queue_a(q_idx,xa0,ya0+1)*(1-xa)*ya + bscans_queue_a(q_idx,xa0+1,ya0+1)*xa*ya;
							F += 1/dists[n];
							contribution += bilinear0/dists[n];
						}
						contribution /= F;
					}

					if (COMPOUND_METHOD == COMPOUND_AVG)
						if (volume_a(x,y,z) != 0) volume_a(x,y,z) = (volume_a(x,y,z) + contribution)/2;	else volume_a(x,y,z) = contribution;
					if (COMPOUND_METHOD == COMPOUND_MAX)
						if (contribution > volume_a(x,y,z)) volume_a(x,y,z) = contribution;
					if (COMPOUND_METHOD == COMPOUND_IFEMPTY)
						if (volume_a(x,y,z) == 0) volume_a(x,y,z) = contribution;
					if (COMPOUND_METHOD == COMPOUND_OVERWRITE)
						volume_a(x,y,z) = contribution;
				}

			}
		}
	}
}

__kernel void trace_intersections(__global float4 * intersections, 
																	int volume_w, 
																	int volume_h, 
																	int volume_n, 
																	float volume_spacing, 
																	__global float4 * bscan_plane_equation_queue,
																	int axis) {

	float4 Rd = {axis == 0, axis == 1, axis == 2, 0};

	int iter_end[3] = {(axis != 0)*volume_w+(axis==0), (axis != 1)*volume_h+(axis==1), (axis != 2)*volume_n+(axis==2)};

	int n = get_global_id(0);

	if (n >= iter_end[0]*iter_end[1]*iter_end[2]) return;

	int x = (axis != 0);
	int y = (axis != 1);
	int z = (axis != 2);

	if (axis == 0) {
		y = n%volume_h;
		z = n/volume_h;
	}
	if (axis == 1) {
		x = n%volume_w;
		z = n/volume_w;
	}
	if (axis == 2) {
		x = n%volume_w;
		y = n/volume_w;
	}

	bool invalid = false;
	for(int f = 0; f < 2; f++) {
		int i = f==0 ? BSCAN_WINDOW/2-1 : BSCAN_WINDOW/2; // Fill voxels between two middle bscans
		//int i = f==0 ? BSCAN_WINDOW/2-BSCAN_WINDOW/4-1 : BSCAN_WINDOW/2+BSCAN_WINDOW/4; // Alternatively fill voxels between BSCAN_WINDOW/2 middle bscans
		//int i = f==0 ? 0 : BSCAN_WINDOW-1; // Alternatively fill voxels between first and last bscan
		float4 Pn = {bscan_plane_equation_queue[i].x, bscan_plane_equation_queue[i].y, bscan_plane_equation_queue[i].z, 0};
		float4 R0 = {x*volume_spacing, y*volume_spacing, z*volume_spacing, 0};
		float Vd = dot(Pn, Rd);
		float V0 = -(dot(Pn, R0) + bscan_plane_equation_queue[i].w);
		if (Vd == 0)
		{
			invalid = true;
			continue;
		}
		float t = V0 / Vd;

		float4 intersection = R0 + t*Rd;
		intersections[n*2 + f] = intersection;
	}
}