#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include <time.h>
#include <omp.h>

typedef double (*run_code)(int, int);

const double a = -4.0;
const double b = 4.0;

double cpuSecond()
{
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ((double)ts.tv_sec + (double)ts.tv_nsec * 1.e-9);
}

double func(double x)
{
    return exp(-x * x);
}

double integrate(double (*func)(double), double a, double b, int n)
{
    double h = (b - a) / n;
    double sum = 0.0;

    for (int i = 0; i < n; i++)
        sum += func(a + h * (i + 0.5));

    sum *= h;

    return sum;
}

double integrate_omp(double (*func)(double), double a, double b, int n, int threads)
{
    double h = (b - a) / static_cast<double>(n);
    double sum = 0.0;
	std::vector<double> array_of_result(threads);

	#pragma omp parallel num_threads(threads)
    {
		double local_sum = 0.0;
        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();
        int items_per_thread = n / threads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? (n - 1) : (lb + items_per_thread - 1);

        for (int i = lb; i <= ub; i++)
		{
            local_sum += func(a + h * (i + 0.5));
		}
		array_of_result[threadid] = local_sum;	
    }

	for (int i = 0; i < threads; i++)
	{
		sum += array_of_result[i];
	}

	sum *= h;

    return sum;
}

double run_serial(int nsteps, int threads)
{
    double t = cpuSecond();
    double res = integrate(func, a, b, nsteps);
    t = cpuSecond() - t;
	std::cout << "Serial error: " << (fabs(res - sqrt(M_PI))) << std:: endl;

    return t;
}


double run_parallel(int nsteps, int threads)
{
    double t = cpuSecond();
    double res = integrate_omp(func, a, b, nsteps, threads);
    t = cpuSecond() - t;
	std::cout << "Parallel error: " << (fabs(res - sqrt(M_PI))) << std:: endl;

    return t;
}


void write_result(std::string content, std::string file_name)
{
	std::ofstream out(file_name, std::ios::app);
	out << content << std::endl;
	out.close();
}


double experement(int nsteps, int times, int threads, run_code func)
{
	double avg_time = 0.0;
	for (int i = 0; i < times; i++)
	{
		avg_time += func(nsteps, threads);
	}

	return avg_time / static_cast<double>(times);
}

int main(int argc, char **argv)
{
	int times = 30;
	std::cout << std::fixed << std::setprecision(8);
	std::vector<int> threads = {2, 4, 7, 8, 16, 20, 40};
	double ast = 0.0;
	double apt = 0.0;
	double boost = 0.0;
	int nsteps = 40000000;

	std::stringstream content;
	std::string title = "iters;serial;boost;threads;\n";
	content  << title;

	double all_time = cpuSecond();
	ast = experement(nsteps, times, 1, run_serial);
	for (auto i : threads)
	{
		apt = experement(nsteps, times, i, run_parallel);
		boost = ast / apt;
		content << nsteps << ";" << ast << ";" << boost << ";" << i << ";\n";
	}

	write_result(content.str(), "exp_4000k.txt");
	content.clear();
	content.str("");

	all_time = cpuSecond() - all_time;
	std::cout << "Time of all experements: " << all_time << std::endl; 
    return 0;
}