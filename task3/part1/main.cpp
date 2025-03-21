#include <iostream>
#include <thread>
#include <vector>
#include <memory>
#include <fstream>
#include <sstream>

double cpuSecond()
{
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ((double)ts.tv_sec + (double)ts.tv_nsec * 1.e-9);
}

double start, end;

void matrix_mult_ser(std::shared_ptr<double[]> a,std::shared_ptr<double[]> b,std::shared_ptr<double[]> c, size_t n)
{
    for (int i = 0; i < n; i++)
    {
        c[i] = 0.0;
        for (int j = 0; j < n; j++)
		{
            c[i] += a[i * n + j] * b[j];
		}
    }

	for (size_t i = 0; i < n; i++)
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

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
		{
   	    	c[i] += a[i * n + j] * b[j];
		}
    }
}


void matrix_mult_par(std::shared_ptr<double[]> a,std::shared_ptr<double[]> b,std::shared_ptr<double[]> c, size_t n, size_t num_of_threads, size_t curr_thread)
{	
	double time = 0.0;
	int nthreads = num_of_threads;
    int threadid = curr_thread; 
	int items_per_thread = n / nthreads;
   	int lb = threadid * items_per_thread;
	int ub = (threadid == nthreads - 1) ? (n - 1) : (lb + items_per_thread - 1);

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
			
    for (int i = lb; i <= ub; i++)
    {
        for (int j = 0; j < n; j++)
		{
            c[i] += a[i * n + j] * b[j];
		}
    }
}


int main()
{
	std::vector<size_t> num_of_threads = {2, 4, 7, 8, 16, 20, 40};
	std::vector<std::thread> threads;
	std::stringstream content;

	auto on_start = [&] ()
	{
		start = cpuSecond();
	};

	auto on_end = [&] ()
	{
		end = cpuSecond() - start;
	};

	size_t n = 40000;
	size_t times = 10;

	// serial region
	{
		std::shared_ptr<double[]> a(new double[n*n]);
		std::shared_ptr<double[]> b(new double[n]);
		std::shared_ptr<double[]> c(new double[n]);
		double avg_time = 0.0;
		for (size_t i = 0; i < times; i++)
		{
			on_start();
			matrix_mult_ser(a, b, c, n);	
			on_end();
			avg_time += end;
			std::cout << "End: " << end << std::endl;
		}

		content << avg_time / static_cast<double>(times) << ";";
	}

	// parallel region
	for (size_t th : num_of_threads)
	{
		std::shared_ptr<double[]> a(new double[n*n]);
		std::shared_ptr<double[]> b(new double[n]);
		std::shared_ptr<double[]> c(new double[n]);

		double avg_time = 0.0;
		for (size_t t = 0; t < times; t++)
		{
			on_start();
			for (size_t i = 0; i < th; i++)
			{
				threads.emplace_back(matrix_mult_par, a, b, c, n, th, i);
			}

			for (auto& thread : threads)
			{
				thread.join();
			}
			on_end();
			avg_time += end;
			std::cout << "End: " << end << std::endl;
			threads.clear();
		}

		content << avg_time / times << ";"; 
	}

	std::ofstream out("res.txt", std::ios::app);
	out << content.str();
	out.close();

	return 0;
}