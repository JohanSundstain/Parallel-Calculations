#include <iostream>
#include <memory>
#include <cmath>
#include <string>
#include <iomanip>
#include <fstream>
#include <boost/program_options.hpp>
#include <cub/cub.cuh>  
#include <cuda_runtime.h>
#include <cfloat>
#include <nvtx3/nvToolsExt.h>

#define OFFSET(x, y, m) (((x) * (m)) + (y))
#define BLOCK_DIM_2D_X 16
#define BLOCK_DIM_2D_Y 16
#define BLOCK_DIM_1D 256

inline int grid(int elems, int block)
{
	return (elems + block - 1)/block;
}

template <typename T>
struct CudaDeleter
{
	void operator()(T* _dptr)
	{
		std::cout << "Free " << _dptr  << std::endl;
		cudaFree(_dptr);

	}
};

template <typename T>
struct CudaHostDeleter
{
	void operator()(T* _dptr)
	{
		std::cout << "Free " << _dptr  << std::endl;
		cudaFreeHost(_dptr);

	}
};

namespace po = boost::program_options;
double cpuSecond() 
{
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return static_cast<double>(ts.tv_sec) + static_cast<double>(ts.tv_nsec) * 1e-9;
}

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

__global__ void calcNextCU(double *__restrict__ A, double *__restrict__ Anew, int dim)
{

	 __shared__ double tile[BLOCK_DIM_2D_X + 2][BLOCK_DIM_2D_Y + 2];

    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;

    int lx = threadIdx.x + 1;
    int ly = threadIdx.y + 1;

    if (gx < dim && gy < dim)
        tile[ly][lx] = A[gy * dim + gx];

    if (threadIdx.x == 0 && gx > 0)
        tile[ly][0] = A[gy * dim + (gx - 1)];
    if (threadIdx.x == blockDim.x - 1 && gx < dim - 1)
        tile[ly][blockDim.x + 1] = A[gy * dim + (gx + 1)];

    if (threadIdx.y == 0 && gy > 0)
        tile[0][lx] = A[(gy - 1) * dim + gx];
    if (threadIdx.y == blockDim.y - 1 && gy < dim - 1)
        tile[blockDim.y + 1][lx] = A[(gy + 1) * dim + gx];

    __syncthreads();  

    if (gx > 0 && gx < dim - 1 && gy > 0 && gy < dim - 1)
	{
        double up    = tile[ly - 1][lx];
        double down  = tile[ly + 1][lx];
        double left  = tile[ly][lx - 1];
        double right = tile[ly][lx + 1];
        Anew[gy * dim + gx] = 0.25 * (up + down + left + right);
    }
}

__global__ void absSubMaxReduceCU(double*__restrict__ A, double*__restrict__ B, double* output, int size)
{
    typedef cub::BlockReduce<double, BLOCK_DIM_1D> BlockReduceT;
    __shared__ typename BlockReduceT::TempStorage temp_storage;

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    double thread_val = (i < size) ? std::fabs(A[i] - B[i]) : -DBL_MAX;

    double block_max = BlockReduceT(temp_storage).Reduce(thread_val, cub::Max());

    if (threadIdx.x == 0) output[blockIdx.x] = block_max;
}

__global__ void maxReduceCU(double* __restrict__ input, double* __restrict__ output, int size)
{
    typedef cub::BlockReduce<double, BLOCK_DIM_1D> BlockReduceT;
    __shared__ typename BlockReduceT::TempStorage temp_storage;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double thread_data = (i < size) ? input[i] : -DBL_MAX;

    double block_max = BlockReduceT(temp_storage).Reduce(thread_data, cub::Max());

    if (threadIdx.x == 0) output[blockIdx.x] = block_max;
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
			std::cout           // научный формат
                      << std::setprecision(3)         // 2 знака после точки (можно 3 или 4)
                      << std::setw(5)                // ширина поля, чтобы всё ровно стояло
                      << Matx[i*m+j] << " ";
		}
		std::cout << "]\n";
	}
}

void save_matrix_to_csv(const std::string& filename, const double* Matx, int n, int m)
{
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
    int n, iter_max, experemnts;
    double tol;
	bool show = false;

	cudaSetDevice(3); 
	int device;
    cudaGetDevice(&device);

    int sharedMemPerBlock;
    cudaDeviceGetAttribute(&sharedMemPerBlock, cudaDevAttrMaxSharedMemoryPerBlock, device);

    std::cout << "Максимальный объем shared memory на блок: "
              << sharedMemPerBlock << " байт" << std::endl;


    // Command-line options
    po::options_description desc("Jacobi OpenACC options");
    desc.add_options()
        ("help,h", "show help message")
        ("n", po::value<int>(&n)->default_value(1024), "number of columns and rows (m)")
		("e", po::value<int>(&experemnts)->default_value(10), "number of experements")
        ("iters,i", po::value<int>(&iter_max)->default_value(1000000), "max iterations")
		("s",po::bool_switch(&show), "Show result to console")
        ("tol,t", po::value<double>(&tol)->default_value(1e-6), "tolerance");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 0;
    }
    po::notify(vm);

    std::cout << "Jacobi relaxation: " << n << " x " << n << " mesh\n";

	// ============
	// CONST VARS
	// ============
	const int ALL_ELEMENTS = n * n;
	const int NEEDED_BYTES = ALL_ELEMENTS * sizeof(double);

	// ============
	// VARIABLES
	// ============
	double avg_time = 0.0, start = 0.0, end = 0.0, error = 1.0;
	int iter = 0;
	
	dim3 matrixBlock(BLOCK_DIM_2D_X, BLOCK_DIM_2D_Y);
	dim3 matrixGrid(grid(n, matrixBlock.x), grid(n, matrixBlock.y));

	dim3 linearBlock(BLOCK_DIM_1D);
	dim3 linearGrid(grid(ALL_ELEMENTS, linearBlock.x));

	// ================
	// CPU pinned pointers
	// ================
	double* A_hptr, *Anew_hptr, *buffer_hptr;
	cudaMallocHost(&A_hptr, NEEDED_BYTES);
	cudaMallocHost(&Anew_hptr, NEEDED_BYTES);
	cudaMallocHost(&buffer_hptr, linearGrid.x * sizeof(double));

 	std::unique_ptr<double, CudaHostDeleter<double>> A(A_hptr);
 	std::unique_ptr<double, CudaHostDeleter<double>> Anew(Anew_hptr);
 	std::unique_ptr<double, CudaHostDeleter<double>> Buffer(buffer_hptr);

	// ===============
	// GPU pointers
	// ===============
	double* A_dptr, *Anew_dptr, *Block_max_dptr;
	cudaMalloc(&A_dptr, NEEDED_BYTES);
	cudaMalloc(&Anew_dptr, NEEDED_BYTES);
	cudaMalloc(&Block_max_dptr,  linearGrid.x * sizeof(double));

	std::unique_ptr<double, CudaDeleter<double>> Agpu(A_dptr); 
	std::unique_ptr<double, CudaDeleter<double>> Anewgpu(Anew_dptr); 
	std::unique_ptr<double, CudaDeleter<double>> Block_max(Block_max_dptr); 

	// ==================
	// Initialize arrays 
	// ==================
	initialize(A_hptr, Anew_hptr, n, n);
	
	cudaGraph_t graph;
	cudaGraphExec_t instance;
	bool graph_created = false;

	double* buffer = new double[linearGrid.x];

	for (int i = 0; i < experemnts; i++)
	{
		iter = 0;
		error = 1.0;

		nvtxRangePushA("copy");
		cudaMemcpy(A_dptr, A_hptr, NEEDED_BYTES, cudaMemcpyHostToDevice);
		cudaMemcpy(Anew_dptr, Anew_hptr, NEEDED_BYTES, cudaMemcpyHostToDevice);

		if (!graph_created)
		{
			cudaStream_t capture_stream;
			cudaStreamCreate(&capture_stream);
			
			cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeGlobal);

			for (int i = 0; i < 500; i++)
			{
				calcNextCU<<<matrixGrid, matrixBlock, 0, capture_stream>>>(A_dptr, Anew_dptr, n);
				calcNextCU<<<matrixGrid, matrixBlock, 0, capture_stream>>>(Anew_dptr, A_dptr, n);
			}

			cudaStreamEndCapture(capture_stream, &graph);
			cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
			cudaStreamDestroy(capture_stream);

			graph_created = true;
		}

 	    nvtxRangePop();

		start = cpuSecond();
		for (iter = 0; iter < iter_max; iter+=1000)
		{
        	// Запускаем граф
			cudaGraphLaunch(instance, 0);


			//std::swap(A_dptr, Anew_dptr);

			//nvtxRangePushA("Calc");
			//calcNextCU<<<matrixGrid, matrixBlock>>>(A_dptr, Anew_dptr, n);
			//calcNextCU<<<matrixGrid, matrixBlock>>>(Anew_dptr, A_dptr, n);
			//std::swap(A_dptr, Anew_dptr);
			//nvtxRangePop();

			if (iter % 2000 == 0)
			{
				nvtxRangePushA("Sub & reduce");
				absSubMaxReduceCU<<<linearGrid, linearBlock>>>(Anew_dptr, A_dptr, Block_max_dptr, ALL_ELEMENTS);
				nvtxRangePop();

				nvtxRangePushA("Memcpy");
				cudaMemcpy(buffer, Block_max_dptr, sizeof(double) * linearGrid.x, cudaMemcpyDeviceToHost);
				nvtxRangePop();

				nvtxRangePushA("CPU find max");
				error = *std::max_element(buffer, buffer+linearGrid.x);
				nvtxRangePop();

				//current_size = num_blocks;
				//while (current_size > 1) 
				//{
					//int num_blocks = grid(current_size, linearBlock.x);
					//maxReduceCU<<<num_blocks, linearBlock.x>>>(in, out, current_size);
					//current_size = num_blocks;
					//std::swap(in, out);
				//}

				//cudaMemcpy(&error, in, sizeof(double), cudaMemcpyDeviceToHost);

				if (error <= tol)
					break;

			}
		}

		end = cpuSecond();
		std::cout << end-start << std::endl;
		avg_time += (end - start);
	}

	std::cout << "Avg time " << avg_time / static_cast<double>(experemnts) << ", " << iter << ", " << error << std::endl;

	cudaMemcpy(A_hptr, A_dptr, NEEDED_BYTES, cudaMemcpyDeviceToHost);
	if (show)
	{
		show_matrix(A_hptr, n, n);
	}

	save_matrix_to_csv("output.csv", A_hptr, n, n);

    //nvtxRangePop();
	if (graph_created)
	{
		cudaGraphExecDestroy(instance);
		cudaGraphDestroy(graph);
	}
    return 0;
}
