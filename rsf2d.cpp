#include <vector>
#include <array>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include<string>
#include <cstdlib>
#include<stdio.h>
#include "tira/image.h"
#include "tira/image/colormap.h"
#include<complex>
#include<cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_runtime_api.h> 
#include<Windows.h>
#include<boost/program_options.hpp>
#include <stack>



using namespace std;






/// <summary>
/// Save a numpy array given a 2D grid
/// </summary>
/// <param name="filename">Filename used to store the .npy array</param>
/// <param name="data">Pointer to the data</param>
/// <param name="X">Size of the X dimension</param>
/// <param name="Y">Size of the Y dimension</param>
/*void save_numpy_2D(std::string filename, float* data, unsigned long X, unsigned long Y) {
	cnpy::npy_save(filename.c_str(), &data[0], { X,Y }, "w");
}

void save_numpy_2D_npz(std::string  filename, std::string filename2, float* data, unsigned long X, unsigned long Y) {
	cnpy::npz_save(filename.c_str(), filename2.c_str(), &data[0], { X,Y }, "a");
}
*/

bool debug = false;



//heaviside function
tira::image<float> heaviside(tira::image<float> array)
{
	tira::image<float> output = array;
	for (int xi = 0; xi < array.width(); xi++) {
		for (int yi = 0; yi < array.height(); yi++)
		{
			float epsilon = 0.2;
			float heaviside = (0.5 * (1 + (2 / 3.14159265358979323846) * atan(array(xi, yi) / epsilon)));
			output(xi, yi) = heaviside;
		}

	}
	return output;
}


//derivative of heaviside function
tira::image<float> deri_heaviside(tira::image<float> array1)
{

	tira::image<float> output = array1;
	for (int xi = 0; xi < array1.width(); xi++) {
		for (int yi = 0; yi < array1.height(); yi++)
		{
			float epsilon = 0.2;
			float deri_heaviside = (1 / 3.14159265358979323846) * (epsilon / ((epsilon * epsilon) + (array1(xi, yi) * array1(xi, yi))));
			output(xi, yi) = deri_heaviside;
		}
	}
	return output;
}



//calculate gradient along dx
tira::image<float> function_dxdy(tira::image<float> dxdy)
{
	tira::image<float> output = dxdy;
	for (int j = 0; j < dxdy.width(); j++)
	{
		for (int i = 0; i < dxdy.height(); i++)
		{
			int j_left = j - 1;
			int j_right = j + 1;
			if (j_left < 0)
			{
				j_left = 0;
				j_right = 1;
				double dist_grad = dxdy(j_left, i) - dxdy(j_right, i);
				
			}

			else if (j_right >= dxdy.width())
			{
				j_right = dxdy.width() - 1;
				j_left = j_right - 1;
				double dist_grad = dxdy(j_right, i) - dxdy(j_left, i);
				
			}


			double dist_grad = (dxdy(j_right, i) - dxdy(j_left, i)) / 2.0f;
			output(j, i) = dist_grad;
		}
	}

	return output;
}


// calculate gradient along dy
tira::image<float> function_dydx(tira::image<float> dxdy)
{
	tira::image<float> output = dxdy;
	for (int j = 0; j < dxdy.width(); j++)
	{
		for (int i = 0; i < dxdy.height(); i++)
		{
			int i_left = i - 1;
			int i_right = i + 1;
			if (i_left < 0)
			{
				i_left = 0;
				i_right = 1;
				double dist_grad = dxdy(j, i_right) - dxdy(j, i_left);
				
			}

			else if (i_right >= dxdy.height())
			{
				i_right = dxdy.height() - 1;
				i_left = i_right - 1;
				double dist_grad = dxdy(j, i_right) - dxdy(j, i_left);
				
			}


			double dist_grad = (dxdy(j, i_right) - dxdy(j, i_left)) / 2.0f;
			
			output(j, i) = dist_grad;
		}
	}

	return output;
}


// calculating distance field
tira::image<float> dist(tira::image<float> levelset) {


	std::vector<float>distGrid;
	distGrid.resize(levelset.height() * levelset.width());

	const int height = levelset.height();
	const int width = levelset.width();
	const int row = width;

	for (int y = 0; y < levelset.height(); y++)
	{
		for (int x = 0; x < levelset.width(); x++)
		{
			distGrid[y * width + x] = levelset(x, y);
		}

	}

	std::vector<float>frozenCells;
	frozenCells.resize(levelset.height() * levelset.width());

	for (int i = 0; i < distGrid.size(); i++)
	{
		if (distGrid[i] == 0) {
			frozenCells[i] = true;
		}
		else {
			frozenCells[i] = false;
		}

	}


	const int NSweeps = 4;



	//// sweep directions { start, end, step }
	const int dirX[NSweeps][3] = { {0, width - 1, 1} , {width - 1, 0, -1}, {width - 1, 0, -1}, {0, width - 1, 1} };
	const int dirY[NSweeps][3] = { {0, height - 1, 1}, {0, height - 1, 1}, {height - 1, 0, -1}, {height - 1, 0, -1} };
	double aa[2];
	double d_new, a, b;
	int s, ix, iy, gridPos;
	const double h = 1.0, f = 1.0;

	for (s = 0; s < NSweeps; s++) {

		for (iy = dirY[s][0]; dirY[s][2] * iy <= dirY[s][1]; iy += dirY[s][2]) {
			for (ix = dirX[s][0]; dirX[s][2] * ix <= dirX[s][1]; ix += dirX[s][2]) {

				gridPos = iy * row + ix;

				if (!frozenCells[gridPos]) {

					// === neighboring cells (Upwind Godunov) ===


					if (iy == 0 || iy == (height - 1)) {                    // calculation for ymin
						if (iy == 0) {
							aa[1] = distGrid[(iy + 1) * row + ix];
						}
						if (iy == (height - 1)) {
							aa[1] = distGrid[(iy - 1) * row + ix];
						}
					}
					else {
						
						aa[1] = std::min(distGrid[(iy - 1) * row + ix], distGrid[(iy + 1) * row + ix]);
					}

					if (ix == 0 || ix == (width - 1)) {                    // calculation for xmin
						if (ix == 0) {
							aa[0] = distGrid[iy * row + (ix + 1)];
						}
						if (ix == (width - 1)) {
							aa[0] = distGrid[iy * row + (ix - 1)];
						}
					}
					else {
						
						aa[0] = std::min(distGrid[iy * row + (ix - 1)], distGrid[iy * row + (ix + 1)]);
					}

					a = aa[0]; b = aa[1];
					d_new = (fabs(a - b) < f * h ? (a + b + sqrt(2.0 * f * f * h * h - (a - b) * (a - b))) * 0.5 : std::fminf(a, b) + f * h);

					distGrid[gridPos] = d_new;
				}
			}
		}
	}

	for (int y = 0; y < levelset.height(); y++)
	{
		for (int x = 0; x < levelset.width(); x++)
		{
			levelset(x, y) = distGrid[y * width + x];
		}

	}

	return levelset;
}


//calculating signed distance field
tira::image<float> sdf(tira::image<float> distance) {
	

	std::vector<float>SDF;
	SDF.resize(distance.height() * distance.width());

	int width = distance.width();
	int height = distance.height();

	for (int y = 0; y < distance.height(); y++)
	{
		for (int x = 0; x < distance.width(); x++)
		{
			SDF[y * width + x] = distance(x, y);
		}

	}

	std::vector <float> frozenCells;
	frozenCells.resize(distance.height() * distance.width());

	for (int i = 0; i < frozenCells.size(); i++)
	{
		if (SDF[i] == 0) {
			frozenCells[i] = true;
		}
		else {
			frozenCells[i] = false;
		}

	}

	for (int i = 0; i < SDF.size(); i++) {
		SDF[i] = -1 * SDF[i];
	}

	//cout << "STARTED \n";
	double val; int gridPos;
	const int row = distance.width();
	const int nx = distance.width() - 1;
	const int ny = distance.height() - 1;
	int ix = 0, iy = 0;

	std::stack<std::tuple<int, int>> stack = {};

	std::tuple<int, int> idsPair;

	// find the first unfrozen cell
	gridPos = 0;

	while (frozenCells[gridPos]) {
		ix += (ix < nx ? 1 : 0);
		iy += (iy < ny ? 1 : 0);
		gridPos = row * iy + ix;
	}
	stack.push({ ix, iy });

	// a simple pixel flood
	while (stack.size()) {
		idsPair = stack.top();
		stack.pop();
		ix = std::get<0>(idsPair);
		iy = std::get<1>(idsPair);
		gridPos = row * iy + ix;
		if (!frozenCells[gridPos]) {
			val = -1.0 * SDF[gridPos];
			SDF[gridPos] = val;
			frozenCells[gridPos] = true; // freeze cell when done
			if (ix > 0) {
				stack.push({ ix - 1, iy });
			}
			if (ix < nx) {
				stack.push({ ix + 1, iy });
			}
			if (iy > 0) {
				stack.push({ ix, iy - 1 });
			}
			if (iy < ny) {
				stack.push({ ix, iy + 1 });
			}
		}
	}


	for (int y = 0; y < distance.height(); y++)
	{
		for (int x = 0; x < distance.width(); x++)
		{
			distance(x, y) = SDF[y * width + x];
		}

	}
	return distance;
}


float normaldistance(float x, float sigma)
{
	float y = 1.0f / (sigma * sqrt(2 * 3.14159));
	float ex = -(x * x) / (2 * sigma * sigma);
	return y * exp(ex);
}

float sigma;
int T;
float dt;
float v1;
float N = 255 * 255;
float v = N * v1;
float e;
std::string centerline;
std::string input_image;
std::string ouput_filename;

void adddevice(float* input, float* fout, float* fin, float* Eo, float* Ei, int img_w, int img_h, int sigma);
void adddevice_convolution(float* y_output, float* in_img, int img_w, int img_h, float sigma, float* gkernel, unsigned int k_size);


int main(int argc, char** argv) {

	// Declare the supported options.
	boost::program_options::options_description desc("Allowed options");
	desc.add_options()
		("help", "produce help message")
		("sigma", boost::program_options::value<float >(&sigma)->default_value(5.0), "blur kernel for the fitting term (in pixels)")
		("T", boost::program_options::value<int>(&T)->default_value(13.0), "total number of time steps")
		("dt", boost::program_options::value<float >(&dt)->default_value(0.1), "time step size")
		("v", boost::program_options::value<float >(&v1)->default_value(0.01), "weight for the curvature term")
		("e", boost::program_options::value<float >(&e)->default_value(1.0), "weight for the fitting term")
		("phi0", boost::program_options::value<std::string>(&centerline)->default_value("circle_23.bmp"), "initial level set (npy)")
		("input", boost::program_options::value<std::string>(&input_image)->default_value("circle_square_200_100.bmp"), "input image filename")
		("ouput", boost::program_options::value<std::string>(&ouput_filename)->default_value("tira.npy"), "output level set (npy)")
		("debug", "output all debugging details")
		;
	boost::program_options::variables_map vm;
	boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(desc).style(
		boost::program_options::command_line_style::unix_style ^ boost::program_options::command_line_style::allow_short
	).run(), vm);
	boost::program_options::notify(vm);

	if (vm.count("help")) {
		std::cout << desc << std::endl;
		return 1;
	}

	if (vm.count("debug")) {
		debug = true;
	}

	// loading input image and centerlines
	//tira::image<float>I1("circle_23.bmp");
	tira::image<float>I1(centerline);
	tira::image<float> centerlines = I1.channel(0);
	tira::image<float>I2(input_image);
	tira::image<float>input = I2.channel(0);

	int file_suffix = 0;

	int width = input.width();
	int height = input.height();


	// initializing all 2D arrays
	tira::image<float>initial_phi(input.width(), input.height());
	tira::image<float> HD_out(input.width(), input.height());
	tira::image<float> I_out(input.width(), input.height());
	tira::image<float> HD_in(input.width(), input.height());
	tira::image<float> I_in(input.width(), input.height());
	tira::image<float>Gradientmagnitude_phi(input.width(), input.height());
	tira::image<float>d2PHI_dx2_1(input.width(), input.height());
	tira::image<float>d2PHI_dy2_1(input.width(), input.height());
	tira::image<float> normalized_DIV(input.width(), input.height());
	tira::image<float> DIV(input.width(), input.height());
	tira::image<float>fout_x;
	tira::image<float>fin_x;
	tira::image<float>Eout(input.width(), input.height());
	tira::image<float>Ein(input.width(), input.height());
	tira::image<float>derivative_heaviside(input.width(), input.height());
	tira::image<float>dPHI_dx(input.width(), input.height());
	tira::image<float>dPHI_dy(input.width(), input.height());
	tira::image<float>d2PHI_dx2(input.width(), input.height());
	tira::image<float>d2PHI_dy2(input.width(), input.height());
	tira::image<float>d2PHI_dx_n_n(input.width(), input.height());
	tira::image<float>d2PHI_dy_n_n(input.width(), input.height());
	tira::image<float>fitting_term(input.width(), input.height());
	tira::image<float>smoothing_term(input.width(), input.height());
	tira::image<float>Eout_Ein(input.width(), input.height());
	tira::image<float>regularization_term(input.width(), input.height());
	




	//creating gaussian kernel
	//float sigma = 5;
	unsigned int size = (sigma * 7);
	float dx = 1.0f;
	float start = -(float)(size - 1) / 2.0f;

	tira::image<float>K({ size, size });
	for (size_t yi = 0; yi < size; yi++)
	{
		float gy = normaldistance(start + dx * yi, sigma);

		for (size_t xi = 0; xi < size; xi++)
		{
			float gx = normaldistance(start + dx * xi, sigma);
			K(xi, yi) = gx * gy;
		}

	}

	int k_size = (7 * sigma);

	std::vector<float>image;
	image.resize(input.height() * input.width());

	initial_phi = dist(centerlines);
	initial_phi = sdf(initial_phi);




	// initialization of cropped images
	tira::image<float>input_crop;
	unsigned int e = (size - 1) / 2;
	short w = input.width() - (size - 1);
	short h = input.height() - (size - 1);
	input_crop = input.crop(e, e, w, h);
	tira::image<float>fout_x1(input_crop.width(), input_crop.height());
	tira::image<float>fin_x1(input_crop.width(), input_crop.height());
	tira::image<float>I_out_blurred(input_crop.width(), input_crop.height());
	tira::image<float>HD_out_blurred(input_crop.width(), input_crop.height());
	tira::image<float>I_in_blurred(input_crop.width(), input_crop.height());
	tira::image<float>HD_in_blurred(input_crop.width(), input_crop.height());


	

	// creating timer for GPU
	cudaEvent_t c_start;
	cudaEvent_t c_stop;
	cudaEventCreate(&c_start);
	cudaEventCreate(&c_stop);
	cudaEventRecord(c_start, NULL);





	// starting the for loop
   //int T = 13;
	for (int t = 0; t < T; t++)
	{
		//applying heaviside function on initial phi
		HD_out = heaviside(initial_phi);


		// multiplying  heaviside_image with input_image
		for (int i = 0; i < HD_out.width(); i++)
		{
			for (int j = 0; j < HD_out.height(); j++)
			{
				I_out(i, j) = input(i, j) * HD_out(i, j);
			}
		}


		//inverse of heaviside image
		for (int i = 0; i < HD_in.width(); i++)
		{
			for (int j = 0; j < HD_in.height(); j++)
			{
				HD_in(i, j) = 1 - HD_out(i, j);
			}
		}


		// multiplying  inverse of heaviside_image with input_image
		for (int i = 0; i < HD_out.width(); i++)
		{
			for (int j = 0; j < HD_out.height(); j++)
			{
				I_in(i, j) = input(i, j) * HD_in(i, j);
			}
		}


		// calculating divergence
		dPHI_dx = function_dxdy(initial_phi);
		dPHI_dy = function_dydx(initial_phi);



		for (int i = 0; i < HD_out.width(); i++)
		{
			for (int j = 0; j < HD_out.height(); j++)
			{
				Gradientmagnitude_phi(i, j) = sqrt((dPHI_dx(i, j) * dPHI_dx(i, j)) + (dPHI_dy(i, j) * dPHI_dy(i, j)));
			}
		}

		for (int i = 0; i < HD_out.width(); i++)
		{
			for (int j = 0; j < HD_out.height(); j++)
			{
				d2PHI_dx2_1(i, j) = dPHI_dx(i, j) / Gradientmagnitude_phi(i, j);
			}
		}

		for (int i = 0; i < HD_out.width(); i++)
		{
			for (int j = 0; j < HD_out.height(); j++)
			{
				d2PHI_dy2_1(i, j) = dPHI_dy(i, j) / Gradientmagnitude_phi(i, j);
			}
		}


		d2PHI_dx2 = function_dxdy(d2PHI_dx2_1);
		d2PHI_dy2 = function_dydx(d2PHI_dy2_1);

		d2PHI_dx_n_n = function_dxdy(dPHI_dx);
		d2PHI_dy_n_n = function_dydx(dPHI_dy);


		for (int i = 0; i < HD_out.width(); i++)
		{
			for (int j = 0; j < HD_out.height(); j++)
			{
				DIV(i, j) = d2PHI_dx_n_n(i, j) + d2PHI_dy_n_n(i, j);
			}
		}

		for (int i = 0; i < HD_out.width(); i++)
		{
			for (int j = 0; j < HD_out.height(); j++)
			{
				normalized_DIV(i, j) = d2PHI_dx2(i, j) + d2PHI_dy2(i, j);
			}
		}

		//std::cout << "___________________________________________GPU VERSION________________________________________________________" << std::endl;

		// fout_x calculation

		adddevice_convolution(I_out_blurred.data(), I_out.data(), input.width(), input.height(), sigma, K.data(), k_size);
		adddevice_convolution(HD_out_blurred.data(), HD_out.data(), input.width(), input.height(), sigma, K.data(), k_size);

		for (int i = 0; i < I_out_blurred.width(); i++)
		{
			for (int j = 0; j < I_out_blurred.height(); j++)
			{
				fout_x1(i, j) = I_out_blurred(i, j) / HD_out_blurred(i, j);
			}

		}

		fout_x = fout_x1.border_replicate((k_size - 1) / 2);

		//fin_x calculation

		adddevice_convolution(I_in_blurred.data(), I_in.data(), input.width(), input.height(), sigma, K.data(), k_size);
		adddevice_convolution(HD_in_blurred.data(), HD_in.data(), input.width(), input.height(), sigma, K.data(), k_size);

		for (int i = 0; i < I_in_blurred.width(); i++)
		{
			for (int j = 0; j < I_in_blurred.height(); j++)
			{
				fin_x1(i, j) = I_in_blurred(i, j) / HD_in_blurred(i, j);
			}

		}

		fin_x = fin_x1.border_replicate((k_size - 1) / 2);



		//Eout and Ein calculation

		int sigma1 = sigma;

		adddevice(input.data(), fout_x.data(), fin_x.data(), Eout.data(), Ein.data(), input.width(), input.height(), sigma1);



		derivative_heaviside = deri_heaviside(initial_phi);

		//calculating final phi (phi_n)

		for (int i = 0; i < Eout_Ein.width(); i++)
		{
			for (int j = 0; j < Eout_Ein.height(); j++)
			{
				Eout_Ein(i, j) = e * (Eout(i, j) - Ein(i, j));
			}
		}

		for (int i = 0; i < Eout_Ein.width(); i++)
		{
			for (int j = 0; j < Eout_Ein.height(); j++)
			{
				fitting_term(i, j) = derivative_heaviside(i, j) * Eout_Ein(i, j) * dt;
			}
		}

		for (int i = 0; i < Eout_Ein.width(); i++)
		{
			for (int j = 0; j < Eout_Ein.height(); j++)
			{
				smoothing_term(i, j) = v * dt * derivative_heaviside(i, j) * normalized_DIV(i, j);
			}
		}

		for (int i = 0; i < Eout_Ein.width(); i++)
		{
			for (int j = 0; j < Eout_Ein.height(); j++)
			{
				regularization_term(i, j) = dt * (DIV(i, j)) - dt * (normalized_DIV(i, j));
			}
		}


		for (int i = 0; i < Eout_Ein.width(); i++)
		{
			for (int j = 0; j < Eout_Ein.height(); j++)
			{
				initial_phi(i, j) = initial_phi(i, j) - fitting_term(i, j) + smoothing_term(i, j) + regularization_term(i, j);
			}
		}



		for (int i = 0; i < initial_phi.width(); i++)
		{
			for (int j = 0; j < initial_phi.height(); j++)
			{
				image[j * width + i] = initial_phi(i, j);
			}
		}

		
		file_suffix++;

		//save_numpy_2D_npz(ouput_filename, "arr1" + std::to_string(file_suffix), &image[0], input.height(), input.width());

	}

	// timer ends
	cudaEventRecord(c_stop, NULL);
	cudaEventSynchronize(c_stop);
	float time_difference_gpu;
	cudaEventElapsedTime(&time_difference_gpu, c_start, c_stop);
	std::cout << "It takes " << time_difference_gpu << " ms to calculate " << std::endl;
}



