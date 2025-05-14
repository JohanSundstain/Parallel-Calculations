#include <iostream>
#include <memory>
#include <cmath>
#include <string>
#include <iomanip>
#include <fstream>
#include <nvtx3/nvToolsExt.h>
#include <cublas_v2.h>
#include <openacc.h>
#include <cuda_runtime.h>
#include <boost/program_options.hpp>

namespace po = boost::program_options;
double cpuSecond() 
{
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return static_cast<double>(ts.tv_sec) + static_cast<double>(ts.tv_nsec) * 1e-9;
}


#define OFFSET(x, y, m) (((x) * (m)) + (y))

void initialize(double* A, double* Anew, int m, int n) 
{
    std::memset(A, 0, n * m * sizeof(double));
    std::memset(Anew, 0, n * m * sizeof(double));
	
	int ind = OFFSET(1, 1, m);
	A[ind] = 10;
	Anew[ind] = 10;

	ind = OFFSET(1, m-2, m);
	A[ind] = 20;
	Anew[ind] = 20;

	ind = OFFSET(n-2, 1, m);
	A[ind] = 30;
	Anew[ind] = 30;

	ind = OFFSET(n-2, m-2, m);
	A[ind] = 20;
	Anew[ind] = 20;

}

void show_matrix(double* Matx, int n, int m)
{
	for (int i = 0; i < m; i++)
	{
		std::cout << "[";
		for (int  j = 0; j < n; j++)
		{
			std::cout << std::scientific              // научный формат
                      << std::setprecision(2)         // 2 знака после точки (можно 3 или 4)
                      << std::setw(8)                // ширина поля, чтобы всё ровно стояло
                      << Matx[i*m+j] << " ";
		}
		std::cout << "]\n";
	}
}

void save_matrix_to_csv(const std::string& filename,
                        const double* Matx, int n, int m) {
    std::ofstream file(filename);
    if (!file.is_open())
	{
        std::cerr << "Ошибка: не удалось открыть файл " << filename << "\n";
        return;
    }

    for (int i = 0; i < m; i++)
	 {
        for (int j = 0; j < n; j++)
		{
            file << std::setprecision(10) << Matx[i*m+j]; 
            if (j != n - 1)
                file << ",";
        }
        file << "\n";
    }

    file.close();
}

int main(int argc, char** argv) 
{
    int m, n, iter_max, experemnts = 10;
    double tol;
	bool show;

    // Command-line options
    po::options_description desc("Jacobi OpenACC options");
    desc.add_options()
        ("help,h", "show help message")
        ("m", po::value<int>(&m)->default_value(4096), "number of columns (m)")
        ("n", po::value<int>(&n)->default_value(4096), "number of rows (n)")
        ("iters,i", po::value<int>(&iter_max)->default_value(1000), "max iterations")
		("s",po::bool_switch(&show), "Show result to console")
        ("tol,t", po::value<double>(&tol)->default_value(1e-6), "tolerance");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 0;
    }
    po::notify(vm);

    std::cout << "Jacobi relaxation: " << n << " x " << m << " mesh\n";

	double avg_time = 0.0;
	double start = 0.0;
	int iter;
	double error = 1.0;
	double end = 0.0;

 	std::unique_ptr<double[]> A(new double[m * n]);
	std::unique_ptr<double[]> Anew(new double[m * n]);
	double* A_ptr = A.get();
	double* Anew_ptr = Anew.get();

	cublasHandle_t handle;
	cudaStream_t stream;
	cublasStatus_t stat = cublasCreate(&handle);
	if (stat != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "Can't create handle\n";
        return 1;
    }
	cudaStreamCreate(&stream); 
	cublasSetStream(handle, stream);

	double* Errors_ptr = nullptr;
	cudaMalloc((void**)&Errors_ptr, m * n * sizeof(double));

	for (int i = 0; i < experemnts; i++)
	{
		error = 1.0;
		start = cpuSecond();
		nvtxRangePushA("init");
		initialize(A_ptr, Anew_ptr, m, n);
 	   	nvtxRangePop();

		#pragma acc enter data copyin(A_ptr[0:m*n], Anew_ptr[0:m*n])
		{
			for (iter = 0; iter < iter_max; iter++)
			{
				nvtxRangePushA("calc");
				#pragma acc parallel loop collapse(2) deviceptr(Errors_ptr)
				for (int j = 1; j < n - 1; ++j) 
				{
					for (int i = 1; i < m - 1; ++i)
					{
						int idx = j*m + i;
						double val = 0.25 * (A_ptr[idx+1] + A_ptr[idx-1] + A_ptr[idx+m] + A_ptr[idx-m]);
						Anew_ptr[idx] = val;
						Errors_ptr[idx] = std::fabs(val - A_ptr[idx]);
					}
				}

				int idx = 0;
				double* cublas_ptr = (double*) acc_deviceptr(Errors_ptr);
				cublasIdamax(handle, m*n, Errors_ptr, 1, &idx);
				int max_idx = idx - 1;
				cudaMemcpyAsync(&error, Errors_ptr + max_idx, sizeof(double), cudaMemcpyDeviceToHost, stream);
				cudaStreamSynchronize(stream);
				nvtxRangePop();
   				if (error <= tol) break;
    			std::swap(A_ptr, Anew_ptr);

			}
			end = cpuSecond();
			std::cout << end-start << std::endl;
			avg_time += (end - start);
		}
	}

	std::cout << "Avg time " << avg_time / static_cast<double>(experemnts) << ", " << iter << ", " << error << std::endl;
   
	#pragma acc exit data copyout(A_ptr[0:m*n])

	if (show)
	{
		show_matrix(A_ptr, n, m);
	}

	save_matrix_to_csv("output.csv", A_ptr, n, m);
	
	cublasDestroy(handle);
	cudaFree(Errors_ptr);
    return 0;
}
