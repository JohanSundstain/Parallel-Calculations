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
		// TO DO 
		// delete throw construction
		throw std::runtime_error("This type is not supported\n");
	}

}

/// @brief Структура для записи ответов с сервера в файл 
/// @tparam T 
template <typename T>
struct record
{
	std::pair<T, T> args; /// Аргументы
	T result; /// Резульат функции над агрументами
};


enum class ClientType { Sin = 0, Sqrt = 1, Pow = 2 }; /// Тип клиента

template <typename T>
class Client
{
public:
	/// @brief Конструктор в который передаётся тип клиента
	/// @param type Тип клиента
	Client(ClientType type)
	{
		this->ctype = type;
		file_name = "out";
		switch (ctype)
		{
			case ClientType::Sin:  file_name += "_sin.txt"; break;
			case ClientType::Sqrt: file_name += "_sqrt.txt"; break;
			case ClientType::Pow:  file_name += "_pow.txt"; break;
			default: break;
		}
	}

	~Client(){}

	/// @brief Получает shared_ptr на сервер, для взаимодействия
	/// @param server Сервер
	void set_server(std::shared_ptr<Server<T>> server)
	{
		this->server = server;
	}

	/// @brief Перегрузка оператора () для создания функционального класса
	void operator()()
	{
		int iterations = uniform<int>(5, 10000);
		std::vector<size_t> ids;
		size_t workers = server->get_num_workers();

		/// Создание ассинхронного метода с отложенным вычислением
		std::future<void> ask_results = std::async(std::launch::deferred, &Client::get_results, this, std::ref(ids));

		for (int i = 0; i < iterations; i++)
		{
			std::pair<T, T> p = this->generate_args();
			/// Генерация порта, по которому будет обращение
			int port = uniform<int>(0, static_cast<int>(workers-1));
			size_t id = this->server->add_task(p, static_cast<size_t>(this->ctype), port);
			ids.push_back(id);
			records[id].args = p;
		}

		ask_results.get();

		write_to_file();
	}

private:
	ClientType ctype; /// Тип клиента
	std::string file_name; /// Имя файла для каждого клиента, куда он сохраняет ответы
	std::map<size_t, record<T>> records; /// Хеш таблица записей ключ - идентификатор задачи, значение - структура запись
	std::shared_ptr<Server<T>> server; /// Указатель на сервер

	/// @brief Генерация случайных аргементов
	/// @return Возвращает пару случайных аргументов
	std::pair<T, T> generate_args()
	{
		std::pair<T, T> p;
		p.first = uniform<T>(0, 10);
		p.second = uniform<T>(0, 10);
		return p;
	}

	/// @brief Записывает ответы по в Хеш таблицу
	/// @param ids Идентификаторы задач, резульататы которых нужно получить
	void get_results(std::vector<size_t>& ids)
	{
		T result;
		for (int i = 0; i < ids.size(); i++)
		{
			result = this->records[ids[i]].result = this->server->request_result(ids[i]);
		}
	}

	/// @brief Запись ответов в файл
	void write_to_file()
	{
		std::stringstream ss;
		for (auto it = records.begin(); it != records.end(); it++)
		{
			ss << it->first << " "
				<< it->second.args.first << " "
				<< it->second.args.second << " "
				<< it->second.result << "\n";
		}
		
		std::ofstream out(this->file_name);
		out << ss.str();
		out.close();
	}
};

#endif