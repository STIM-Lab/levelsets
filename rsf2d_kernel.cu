
#include<iostream>
#include "cuda_runtime.h"
#include<string>
#include<vector>
#include<fstream>
#include <cuda.h>
#include "device_launch_parameters.h"
#include <cuda_runtime_api.h>
#include <device_functions.h>


# define PI  3.14159265358979323846

static void HandleError(cudaError_t err, const char* file, int line) {
	if (err != cudaSuccess) {
		std::cout << cudaGetErrorString(err) << "in" << file << "at line" << line;
	}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ )) 



__global__ void Eout_Ein_calculation(float* Eo_gpu, float* Ei_gpu, float* fo_gpu, float* fi_gpu, float* input_img_gpu, float* input, float* fout, float* fin, int img_w, int img_h,  int sigma) {

	
	int i = blockDim.y * blockIdx.y + threadIdx.y;	// calculate row index, point to the output  //width 
	int j = blockDim.x * blockIdx.x + threadIdx.x;	// calculate column index, point to the output //height

	float s1 = 0;
	float s2 = 0;

	for (int u = -sigma; u <= sigma; u++)
	{
		for (int v = -sigma; v <= sigma; v++)
		{
			if (((i + u) >= 0) && ((i + u) < img_h) && ((j + v) >= 0) && ((j + v) < img_w))
			{
				s1 = (s1 + ((1 / pow(((2 * sigma) + 1), 2)) * (input_img_gpu[((i + u) * img_w) + (j + v)] - fo_gpu[(i * img_w) + j]) * (input_img_gpu[((i + u) * img_w) + (j + v)] - fo_gpu[(i * img_w) + j])));
				s2 = (s2 + ((1 / pow(((2 * sigma) + 1), 2)) * (input_img_gpu[((i + u) * img_w) + (j + v)] - fi_gpu[(i * img_w) + j]) * (input_img_gpu[((i + u) * img_w) + (j + v)] - fi_gpu[(i * img_w) + j])));
			}
		}
	}
	Eo_gpu[i * img_w + j] = s1;
	Ei_gpu[i * img_w + j] = s2;

	

}

// convolution on device
__global__ void Convolution__on_device(float* out, float* img, float* kernel, int img_w,  int out_w, int out_h, int K) {
	size_t i = blockDim.y * blockIdx.y + threadIdx.y;
	size_t j = blockDim.x * blockIdx.x + threadIdx.x;

	// i and j being smaller than output's width and height, manage the edges perfectly
	if (i >= out_h || j >= out_w) return;

	float conv = 0;
	for (int ki = 0; ki < K; ki++)
		for (int kj = 0; kj < K; kj++)
			conv += img[(i + ki) * img_w + j + kj] * kernel[ki * K + kj];

	out[i * out_w + j] = conv;

}



void adddevice_convolution(float* y_output, float* in_img,  int img_w, int img_h, float sigma, float* gkernel , unsigned int k_size) {


	cudaDeviceProp props;
	HANDLE_ERROR(cudaGetDeviceProperties(&props, 0));


	// allocating the output array for pixels after convolution along y axis
	int y_height = img_h - k_size + 1;
	int y_width = img_w - k_size + 1;
	int y_size = y_height * y_width ;

	
	float* gkernel_gpu;
	float* input_img_gpu;
	float* gpu_output_y;
	size_t bytes = (img_w * img_h) * sizeof(float);
	

	HANDLE_ERROR(cudaMalloc(&gkernel_gpu, k_size * k_size * sizeof(float)));
	HANDLE_ERROR(cudaMalloc(&input_img_gpu, bytes));  							    //allocate memory on device
	HANDLE_ERROR(cudaMalloc(&gpu_output_y, y_size * sizeof(float)));  				//allocate memory on device


	HANDLE_ERROR(cudaMemcpy(input_img_gpu, in_img, bytes, cudaMemcpyHostToDevice));     //copy the array from main memory to device
	HANDLE_ERROR(cudaMemcpy(gkernel_gpu, gkernel, k_size * k_size * sizeof(float), cudaMemcpyHostToDevice));     //copy the array from main memory to device


	size_t blockDim = sqrt(props.maxThreadsPerBlock);
	dim3 threads(blockDim, blockDim);
	dim3 blocks(img_w / threads.x +1, img_h / threads.y +1);

	Convolution__on_device << < blocks, threads >> > (gpu_output_y, input_img_gpu, gkernel_gpu, img_w, y_width, img_h, k_size);
	HANDLE_ERROR(cudaMemcpy(y_output, gpu_output_y, y_size * sizeof(float), cudaMemcpyDeviceToHost));

	
	
	cudaFree(gpu_output_y);
	cudaFree(gkernel_gpu);
	cudaFree(input_img_gpu); 


}

void adddevice(float* input, float* fout, float* fin, float* Eo, float* Ei, int img_w, int img_h, int sigma) {
	
	
	cudaDeviceProp props;
	HANDLE_ERROR(cudaGetDeviceProperties(&props, 0));


	size_t blockDim = sqrt(props.maxThreadsPerBlock);
	dim3 threads(blockDim, blockDim);
	dim3 blocks(img_w / threads.x , img_h / threads.y );


	float* Eo_gpu;
	float* Ei_gpu;
	float* fo_gpu;
	float* fi_gpu;
	float* input_img_gpu;
	size_t bytes = (img_w * img_h) * sizeof(float);
	HANDLE_ERROR(cudaMalloc(&input_img_gpu, bytes));  							    //allocate memory on device
	HANDLE_ERROR(cudaMalloc(&fo_gpu,  bytes));  							//allocate memory on device
	HANDLE_ERROR(cudaMalloc(&fi_gpu, bytes));  							//allocate memory on device
	HANDLE_ERROR(cudaMalloc(&Eo_gpu, bytes));  							//allocate memory on device
	HANDLE_ERROR(cudaMalloc(&Ei_gpu, bytes));  							//allocate memory on device

	HANDLE_ERROR(cudaMemcpy(input_img_gpu, input, bytes, cudaMemcpyHostToDevice));     //copy the array from main memory to device
	HANDLE_ERROR(cudaMemcpy(fo_gpu, fout, bytes, cudaMemcpyHostToDevice));     //copy the array from main memory to device
	HANDLE_ERROR(cudaMemcpy(fi_gpu, fin, bytes, cudaMemcpyHostToDevice));     //copy the array from main memory to device

	Eout_Ein_calculation << < blocks, threads >> > (Eo_gpu, Ei_gpu, fo_gpu, fi_gpu, input_img_gpu, input, fout, fin, img_w, img_h,  sigma);
	

	HANDLE_ERROR(cudaMemcpy(Eo , Eo_gpu, bytes, cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(Ei , Ei_gpu, bytes, cudaMemcpyDeviceToHost));


	cudaFree(Eo_gpu);
	cudaFree(Ei_gpu);
	cudaFree(fo_gpu);
	cudaFree(fi_gpu);
	cudaFree(input_img_gpu);


}

