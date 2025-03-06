#include <iostream>
#include <memory>
#include <sstream>
#include <vector>
#include <fstream>
#include <time.h>
#include <omp.h>


template <typename T>
struct SimpleAllocator {
    using value_type = T;

    SimpleAllocator() = default;

    template <typename U>
    constexpr SimpleAllocator(const SimpleAllocator<U>&) noexcept {}

    T* allocate(std::size_t n) 
	{
        return static_cast<T*>(::operator new(n * sizeof(T)));
    }

    void deallocate(T* p, std::size_t) noexcept
	{
        ::operator delete(p);
    }
};


double cpuSecond()
{
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ((double)ts.tv_sec + (double)ts.tv_nsec * 1.e-9);
}

double get_serial_time(size_t m, size_t n)
{
	std::vector<double> a;
	std::vector<double> b;
	std::vector<double> c;
	a.reserve(m*n);
	b.reserve(n);
	c.reserve(m);
	double time = 0.0;
	
    for (int i = 0; i < m; i++)
    {
        c[i] = 0.0;
        for (int j = 0; j < n; j++)
            c[i] += a[i * n + j] * b[j];
    }

	for (size_t i = 0; i < m; i++)
    {
    	for (size_t j = 0; j < n; j++)
	  	{
 	    	a[i * n + j] = i + j;
	  	}
		c[i] = 0.0;
   	}
	
    for (size_t j = 0; j < n; j++)
	{
		b[j] = j;
	}

	time = cpuSecond();
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
		{
   	    	c[i] += a[i * n + j] * b[j];
		}
    }
	time = cpuSecond() - time;
	return time;
}

double get_parallel_time(size_t m, size_t n, size_t k)
{	
	// m - lines, n - rows, k - number of threads
	std::vector<double> a;
	std::vector<double> b;
	std::vector<double> c;
	a.reserve(m*n);
	b.reserve(n);
	c.reserve(m);
	double time = 0.0;
	#pragma omp parallel num_threads(k)
    {	
        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();
        int items_per_thread = m / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? (m - 1) : (lb + items_per_thread - 1);
		
		// init arrays 
	    for (size_t i = lb; i <= ub; i++)
	    {
  	    	for (size_t j = 0; j < n; j++)
		  	{
   	 	    	a[i * n + j] = i + j;
		  	}
			c[i] = 0.0;
    	}

	    for (size_t j = 0; j < n; j++)
		{
			b[j] = j;
		}

		#pragma omp barrier

		#pragma omp single
		{
			time = cpuSecond();
		}
		
        for (int i = lb; i <= ub; i++)
        {
            for (int j = 0; j < n; j++)
                c[i] += a[i * n + j] * b[j];
        }
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

double avg_ser_time(size_t m, size_t n, size_t times)
{
	double avg_time_ser = 0.0;

	for (size_t i = 0; i < times; i++)
	{
		avg_time_ser += get_serial_time(m, n);
	}

	return avg_time_ser / static_cast<double>(times); 
}

double avg_par_time(size_t m, size_t n, size_t times, size_t threads)
{
	double avg_time_par = 0.0;

	for (size_t i = 0; i < times; i++)
	{
		avg_time_par += get_parallel_time(m, n, threads);
	}

	return avg_time_par / static_cast<double>(times);
}

int main(int argc, char *argv[])
{
	size_t N = 20000, M = 20000;
	size_t times = 10;

	std::vector<size_t> threads = {2, 4, 7, 8, 16, 20, 40};
	double ast = 0.0;
	double apt = 0.0;
	double boost = 0.0;
	std::stringstream content;
	std::string title = "iters;serial;boost;threads;\n";
	content  << title;

	double all_time = cpuSecond();
	ast = avg_ser_time(N, M, times);
	for (size_t i : threads)
	{
		apt = avg_par_time(N, M, times, i);
		boost = ast / apt;
		content << N*M << ";" << ast << ";" << boost << ";" << i << ";\n";
	}

	write_result(content.str(), "exp_20x20k.txt");
	content.str("");
	content << title;

	N = 40000; M = 40000;
	ast = avg_ser_time(N, M, times);
	for (size_t i : threads)
	{
		apt = avg_par_time(N, M, times, i);
		boost = ast / apt;
		content << N*M << ";" << ast << ";" << boost << ";" << i << ";\n";
	}
	write_result(content.str(), "exp_40x40k.txt");
	all_time = cpuSecond() - all_time;
	std::cout << "Time of all experements: " << all_time << std::endl; 

	return 0;
}