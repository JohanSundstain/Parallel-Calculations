#ifndef CLIENT
#define CLIENT

#include <type_traits>
#include <random>
#include <future>
#include <iostream>
#include <memory>
#include <limits>
#include <fstream>
#include <map>
#include <sstream>
#include <iterator>

#include "Server.cpp"

/// @brief Функция генерации числа из равномерного разпределения
/// @tparam T тип генерации 
/// @param min Нижняя граница числа (включительно)
/// @param max Верхняя граница числа (включительно)
/// @return Возвращает случайное число из диапазона
template <typename T>
T uniform(T min, T max)
{
	std::random_device rd;
	std::mt19937 gen(rd());

	if constexpr (std::is_same<T, double>::value)
	{
		std::uniform_real_distribution<double> dist(min, max);
		return dist(gen);
	}
	else if constexpr (std::is_same<T, int>::value)
	{
		std::uniform_int_distribution<int> dist(min, max);
		return dist(gen);
	}
	else
	{
		throw std::runtime_error("This type is not supported\n");
	}

}


enum class ClientType { Sin = 0, Sqrt = 1, Pow = 2 };

template <typename T>
class Client
{
public:
	Client(ClientType type)
	{
		this->ctype = type;
		file_name = "out";

		switch (ctype)
		{
			case ClientType::Sin:
			{
				file_name += "_sin.txt";
				task = [](T x, T y) {return std::sin(x); };
				break;
			}
			case ClientType::Sqrt:
			{
				file_name += "_sqrt.txt"; 
				task = [](T x, T y) {return std::sqrt(x); };
				break;
			}
			case ClientType::Pow:
			{
				file_name += "_pow.txt";
				task = [](T x, T y) {return std::pow(x,y); };
				break;
			}

			default: break;
		}
	}

	~Client(){}

	void set_server(std::shared_hptr<Server> server)
	{
		this->server = server;
	}

	void operator()()
	{
		int iterations = uniform<int>(5, 10000);

		for (int i = 0; i < iterations; i++)
		{
			size_t id = this->get_id();
			T arg1 = uniform<T>(0, 100);
			T arg2 = uniform<T>(0, 100);
			auto result = this->server->add_task(task, arg1, arg2);
			auto answer = result.get();
			logger << id << " " << arg1 << " " << arg2 << " " << answer << std::endl;
		}

		write_to_file();
	}

private:
	ClientType ctype;
	std::string file_name;
	std::stringstream logger;
	std::shared_hptr<Server> server;
	std::function<T(T, T)> task;
	static size_t uniq_id;
	static std::mutex gen_mutex;


	static size_t get_id()
	{
		std::lock_guard<std::mutex> lock(gen_mutex);
		return uniq_id++;
	}

	void write_to_file()
	{		
		std::ofstream out(this->file_name);
		out << logger.str();
		out.close();
	}
};
template <typename T>
size_t Client<T>::uniq_id = 0;

template <typename T>
std::mutex Client<T>::gen_mutex;

#endif