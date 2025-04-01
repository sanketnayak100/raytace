#include <iostream>
#include <ray_funcs.h>	

__host__ void cuda_trace(int num_threads, tira::camera cam, tira::triangle* tri, spheres* s1, spheres bound_sph, lights* l1, plane p, int res_x, int res_y, int tri_count, int num_sph, int num_lights, int num_planes, int num_mesh, float* image_mat)
{
	tira::camera *device_cam;
	tira::triangle* device_tri;
	spheres *device_sph, *dev_bound_sph; 
	lights* device_lights;
	plane* dev_p;
	float* dev_image_mat;
	cudaError_t err;

	//malloc

	err = cudaMalloc((void**)&device_cam, sizeof(tira::camera));
	HANDLE_ERROR(err);

	err = cudaMalloc((void**)&device_tri, tri_count * sizeof(tira::triangle));
	HANDLE_ERROR(err);

	err = cudaMalloc(&device_sph, num_sph * sizeof(spheres));
	HANDLE_ERROR(err);

	err = cudaMalloc(&dev_bound_sph, sizeof(spheres));
	HANDLE_ERROR(err);

	err = cudaMalloc(&device_lights, num_lights * sizeof(lights));
	HANDLE_ERROR(err);

	err = cudaMalloc(&dev_p, sizeof(plane));
	HANDLE_ERROR(err);

	err = cudaMalloc(&dev_image_mat, res_x * res_y * 3 * sizeof(float));
	HANDLE_ERROR(err);

	//memcpy

	err = cudaMemcpy(dev_image_mat, image_mat, res_x * res_y * 3 * sizeof(float), cudaMemcpyHostToDevice);
	HANDLE_ERROR(err);

	err = cudaMemcpy(device_cam, &cam, sizeof(tira::camera), cudaMemcpyHostToDevice);
	HANDLE_ERROR(err);

	err = cudaMemcpy(device_sph, s1, num_sph * sizeof(spheres), cudaMemcpyHostToDevice);
	HANDLE_ERROR(err);

	err = cudaMemcpy(dev_bound_sph, &bound_sph, sizeof(spheres), cudaMemcpyHostToDevice);
	HANDLE_ERROR(err);

	err = cudaMemcpy(device_lights, l1, num_lights * sizeof(lights), cudaMemcpyHostToDevice);
	HANDLE_ERROR(err);

	err = cudaMemcpy(dev_p, &p, sizeof(plane), cudaMemcpyHostToDevice);
	HANDLE_ERROR(err);

	err = cudaMemcpy(device_tri, tri, tri_count * sizeof(tira::triangle), cudaMemcpyHostToDevice);
	HANDLE_ERROR(err);

	dim3 blockSize(16, 16); // Each block has 16x16 threads
	dim3 gridSize(63,63); // Number of blocks needed

	raytrace_loop_cuda << < gridSize,blockSize >>>(device_cam, device_tri, device_sph, dev_bound_sph, device_lights, dev_p, res_x, res_y, tri_count, num_sph, num_lights, num_planes, num_mesh, dev_image_mat);
	err = cudaDeviceSynchronize();
	HANDLE_ERROR(err);

	err = cudaMemcpy(image_mat, dev_image_mat, res_x * res_y * 3 * sizeof(float), cudaMemcpyDeviceToHost);
	HANDLE_ERROR(err);

	cudaFree(dev_image_mat);
	cudaFree(device_cam);
	cudaFree(device_tri);
	cudaFree(device_sph);
	cudaFree(dev_bound_sph);
	cudaFree(device_lights);
	cudaFree(dev_p);
	//cudaMemcpy(y, x, sizeof(int), cudaMemcpyDeviceToHost);
}