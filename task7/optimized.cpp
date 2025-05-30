#include <iostream>
#include <memory>
#include <cmath>
#include <string>
#include <iomanip>
#include <fstream>
#include <cublas_v2.h>
//#include <nvtx3/nvToolsExt.h>
#include <boost/program_options.hpp>

class cuBLAS_handler
{
private:
	cublasHandle_t handler;
	cublasStatus_t stat;
public:
	cuBLAS_handler()
	{
		this->stat = cublasCreate(&this->handler);
	}

	cublasStatus_t get_status()
	{
		return this->stat;
	}

	~cuBLAS_handler()
	{
		cublasDestroy(this->handler);
	}

	cublasHandle_t& get_handle()
	{
		return this->handler;
	}

};

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
	// m - columns
	// n - rows
    std::memset(A, 0, n * m * sizeof(double));
    std::memset(Anew, 0, n * m * sizeof(double));

	int ind;
	double a, b;
	double step;

	a = 10, b = 20;
	step = (b - a) / (m - 1);
	for (int i = 0; i < m; i++)
	{
		ind = OFFSET(0, i, m);
		Anew[ind] = A[ind] = a + step * i;
	}

	a = 30, b = 20;
	step = (b - a) / (m - 1);
	for (int i = 0; i < m; i++)
	{
		ind = OFFSET(n-1, i, m);
		Anew[ind] = A[ind] = a + step * i;
	}

	a = 10, b = 30;
	step = (b - a) / (n - 1);
	for (int i = 0; i < n; i++)
	{
		ind = OFFSET(i, 0, m);
		Anew[ind] = A[ind] = a + step * i;
	}

	a = 20, b = 20;
	step = (b - a) / (n - 1);
	for (int i = 0; i < n; i++)
	{
		ind = OFFSET(i, m-1, m);
		Anew[ind] = A[ind] = a + step * i;
	}
}

void show_matrix(double* Matx, int m, int n)
{
	// m - column
	// n - rows
	for (int i = 0; i < n; i++)
	{
		std::cout << "[";
		for (int  j = 0; j < m; j++)
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
	int iter = 0;
	double error = 1.0;
	double end = 0.0;

	// =========================================================
	// cuBLAS variables for sub matrix C = α * op(A) + β * op(B),
	// ========================================================
	cuBLAS_handler handler;
	cublasStatus_t stat = handler.get_status();
	if (stat != CUBLAS_STATUS_SUCCESS) {
		std::cerr << "Can't create handle\n";
		return 1;
	}


 	std::unique_ptr<double[]> A(new double[m * n]);
	std::unique_ptr<double[]> Anew(new double[m * n]);
	std::unique_ptr<double[]> Error(new double[m* n]);
	double* A_hptr = A.get();
	double* Anew_hptr = Anew.get();
	double* Error_hptr = Error.get();

	#pragma acc data create(Error_hptr[0:m*n], A_hptr[0:m*n], Anew_hptr[0:m*n])
	{
		for (int i = 0; i < experemnts; i++)
		{
			iter = 0;
			error = 1.0;
			start = cpuSecond();
			//nvtxRangePushA("init");
			initialize(A_hptr, Anew_hptr, m, n);
			#pragma acc update device(A_hptr[0:m*n], Anew_hptr[0:m*n])
		// nvtxRangePop();		
			for (iter = 1; iter <= iter_max; iter++)
			{
				//nvtxRangePushA("calc");
				if (iter % 1000 == 0)
				{
					#pragma acc parallel loop collapse(2) present(A_hptr, Anew_hptr)
					for (int j = 1; j < n - 1; ++j) 
					{
						for (int i = 1; i < m - 1; ++i)
						{
							int idx = j*m + i;
							double val = 0.25 * (A_hptr[idx+1] + A_hptr[idx-1] + A_hptr[idx+m] + A_hptr[idx-m]);
							Anew_hptr[idx] = val;
						}
					}
						
					int idx;
					#pragma acc host_data use_device(Anew_hptr, A_hptr, Error_hptr)
					{
						double alpha = 1;
						double beta = -1;
						// C = alpha * A + beta * B
						cublasDgeam(handler.get_handle(), CUBLAS_OP_N, CUBLAS_OP_N, m, n, &alpha, Anew_hptr, m, &beta, A_hptr, m, Error_hptr, m);
						cublasIdamax(handler.get_handle(), m*n, Error_hptr, 1, &idx);
					}
					idx = idx - 1;
					double* max_elem_hptr = &Error_hptr[idx];
					#pragma acc update host(max_elem_hptr[0])
					error = max_elem_hptr[0];
				}
				else
				{	
					#pragma acc parallel loop collapse(2) present(A_hptr, Anew_hptr)
					for (int j = 1; j < n - 1; ++j) 
					{
						for (int i = 1; i < m - 1; ++i)
						{
							int idx = j*m + i;
							double val = 0.25 * (A_hptr[idx+1] + A_hptr[idx-1] + A_hptr[idx+m] + A_hptr[idx-m]);
							Anew_hptr[idx] = val;
						}
					}
				}
				//nvtxRangePop();
				if (error <= tol)
				{
					break;
				}
				//nvtxRangePushA("swap");
				std::swap(A_hptr, Anew_hptr);
				//nvtxRangePop();
			}
			end = cpuSecond();
			std::cout << end-start << std::endl;
			avg_time += (end - start);
		}
		#pragma acc update host(A_hptr[0:n*m])
	}
	std::cout << "Avg time " << avg_time / static_cast<double>(experemnts) << ", " << iter << ", " << error << std::endl;
	
	if (show)
	{
		show_matrix(A_hptr, n, m);
	}

	save_matrix_to_csv("output.csv", A_hptr, n, m);

    //nvtxRangePop();

    return 0;
}
