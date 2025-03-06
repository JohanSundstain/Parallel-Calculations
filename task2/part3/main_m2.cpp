#include <iostream>
#include <memory>
#include <sstream>
#include <numeric>
#include <iomanip>
#include <thread>
#include <cmath>
#include <vector>
#include <unistd.h>
#include <fstream>
#include <time.h>
#include <omp.h>

const double tau = 0.01;
const double eps = 10.e-5;
const int max_iters = 1000000;

double cpuSecond()
{
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ((double)ts.tv_sec + (double)ts.tv_nsec * 1.e-9);
}

double l2norm(const std::vector<double>& vec)
{
	return std::sqrt(std::accumulate(vec.begin(),vec.end(), 0.0, 
	[](double prev, double elem){return prev + elem * elem; }));
}

void draw_matrix(const std::vector<double>& vec, int rows, int columns)
{
	size_t n = vec.size();
	for (int row = 0; row < rows; row++) 
	{
		for (int column = 0; column < columns; column++)
		{
			std::cout << "|" << vec[row * rows + column];  
		}
		std::cout << "|\n";
	}
}


double simple_iter_method(size_t n, size_t k)
{	
	// m - lines, n - rows, k - number of threads
	std::vector<double> a, b, x, x_next, diff, norms(k);
	a.reserve(n*n);
	b.reserve(n);
	x.reserve(n);
	x_next.reserve(n);
	diff.reserve(n);

	double time = 0.0;
	double norm_b = sqrt(n * (n + 1) * (n + 1));
	double norm = 10.e+6;
	int iters = 0.0;
	bool refresh = true, leave = false;

	#pragma omp parallel num_threads(k)
    {	
        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();
        int items_per_thread = n / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? (n - 1) : (lb + items_per_thread - 1);

	    for (int i = lb; i <= ub; i++)
	    {
  	    	for (int j = 0; j < n; j++)
		  	{ 	
   	 	    	a[i * n + j] =  i == j ? 2.0 : 1.0;
		  	}
			b[i] = n + 1;
			x_next[i] = 0.0;
			x[i] = 0.0;
    	}
		#pragma omp barrier

		#pragma omp single
		{
			time = cpuSecond();
		}

		double sum;
		do
		{
			sum = 0.0;
			norms[threadid] = 0.0;
			for (int i = lb; i <= ub; i++)
   	    	{
				diff[i] = 0.0;
         	  	for (int j = 0; j < n; j++)
				{
        	        diff[i] += a[i * n + j] * x[j];
				}
				diff[i] -= b[i];
			}

			for (int i = lb; i <= ub; i++)
			{
				x_next[i] = x[i] - (tau * diff[i]);
				sum += diff[i] * diff[i]; 
			}
			
			norms[threadid] += sum;

			#pragma omp barrier
			
			#pragma omp single
			{
				if (iters == max_iters)
				{
					std::cout << "Over max iters\n";
					leave = true;
				}
				norm = l2norm(norms);
				refresh =  norm < eps ? false : true;
				iters++;
			}
			#pragma omp barrier		

			if (leave)
			{
				break;
			}

			if (refresh)
			{
				for (int i = lb; i <= ub; i++)
				{
					x[i] = x_next[i];
				}
			}

			#pragma omp barrier	
		}
		while (norm > eps);
    }
	time = cpuSecond() - time;
	return time;
}

void write_result(std::string content, std::string file_name)
{
	std::ofstream out(file_name, std::ios::app);
	out << content << std::endl;
	out.close();
}

double experement(int times, int n, int threads)
{
	double avg_time = 0.0;
	for (int i = 0; i < times; i++)
	{
		avg_time += simple_iter_method(n, threads);
	}
	
	return avg_time / static_cast<double>(times);
}

int main(int argc, char *argv[])
{
	std::cout << std::scientific << std::setprecision(1);

	std::string title = "iters;threads;time;\n";
	std::stringstream content;
	content << title;
	int times = 3;
	int n = 20000;
	int threads = std::thread::hardware_concurrency();
	double apt = 0.0;
	for (int i = 1; i <= threads; i++)
	{
		apt = experement(times, n, i); 
		std::cout << "Avg time: " << apt << " by "<< i <<std::endl;
		content << n*n << ";" << i << ";" << apt <<";\n";
	}
	
	write_result(content.str(), "exp_40x40k_m2.txt");
	return 0;
}