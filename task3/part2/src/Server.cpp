#ifndef SERVER
#define SERVER

#include <iostream>
#include <cmath>
#include <ctime>
#include <mutex>
#include <thread>
#include <queue>
#include <condition_variable>
#include <cstring>
#include <vector>
#include <cstddef>

/**
 * @brief Структура для хранения передаваемых параметров
 */
template <typename T>
struct param
{
	size_t id; /// Идентификатор
	size_t type; /// Тип вычисляемой функции (0 - sin, 1 - sqrt, 2 - pow)
	std::pair<T, T> args; /// Аргументы для функций
};

/**
 * @brief Структура для неблокирующего доступа потоков к результатам вычесления
 * @  packed 
 */
struct alignas(64) field
{
	bool packed; /// Флаг для опредления готовности ответа
	char data[63]; /// Поле для данных
};


template <typename T>
class Server
{
public:

	/// @brief 
	/// @param num_workers Количество потоков в пуле 
	Server(size_t num_workers)
	{
		this->num_workers = num_workers;
		this->global_id = 0;

		this->ports.resize(this->num_workers);

		for (int i = 0; i < this->num_workers; i++)
		{
			this->cvs.push_back(std::make_shared<std::condition_variable>());
			this->mutexes.push_back(std::make_shared<std::mutex>());
		}

		this->results = std::shared_ptr<field[]>{ new field[30000] };
		for (int i = 0; i < 30000; i++)
		{
			this->results[i].packed = false;
		}
	}

	~Server()
	{
		if (!to_stop)
		{
			this->stop();
		}
	}

	/// @brief Запуск сервера и запуск потоков с функцией listen
	void start()
	{
		this->to_stop = false;
		for (size_t worker = 0; worker < this->num_workers; worker++)
		{
			workers.emplace_back(&Server::listen, this, worker);
		}
	}
	
	/// @brief Остановка сервера и остановка всех потоков
	void stop()
	{
		this->to_stop = true;
		for (std::shared_ptr<std::condition_variable> cv : cvs)
		{
			cv->notify_one();
		}

		for (std::thread& t : workers)
		{
			t.join();
		}
	}

	/// @brief Функция возврата количества потоков 
	/// @return Количество потоков 
	size_t get_num_workers()
	{
		return this->num_workers;
	}

	/// @brief Функция добавления задачи по заданному порту для сервера 
	/// @param args Аргументы для функци 
	/// @param type Тип функции
	/// @param port На какой порт отправляется задача
	/// @return Идентификатор задачи
	size_t add_task(std::pair<T, T> args, size_t type, int port)
	{
		std::unique_lock<std::mutex> lock(*mutexes[port]);
		size_t id = this->generate_id();
		this->ports[port].push({ id, type, args });
		cvs[port]->notify_one();
		return id;
	}

	/// @brief Возвращает результат задачи по идентификатору 
	/// @param task_id Идентификатор задачи
	/// @return Возвращает резултат задачи
	T request_result(size_t task_id)
	{
		T data;

		while (!results[task_id].packed)
		{
			std::unique_lock<std::mutex> lock(request_mutex);
			cv_return.wait(lock, [&] {return results[task_id].packed; });
		}

		std::memcpy(&data, results[task_id].data, sizeof(T));
		return data;
	}

private:
	size_t global_id; /// Глобальный идентификатор для сервера
	bool to_stop; /// Флаг для остановки
 
	size_t num_workers; /// Количество потоков
	std::vector<std::thread> workers; /// Вектор потоков
	std::vector<std::shared_ptr<std::mutex>> mutexes; /// Вектор мьютексов для каждого порта
	std::vector<std::queue<param<T>>> ports; /// Вектор очередей для каждого порта
	std::vector<std::shared_ptr<std::condition_variable>> cvs; /// Вектор условных переменный для отслеживания работы на каждом порту
	std::shared_ptr<field[]> results; /// Массив структур результата

	std::mutex request_mutex, id_generator; /// Мьютекс для возврата задачи, для генерации идентификатора
	std::condition_variable cv_return; /// Условная переменная для отслеживания готовности данных для возврата

	// Server's api
	T func_sin(T x)
	{
		return std::sin(x);
	}

	T func_sqrt(T x)
	{
		return std::sqrt(x);
	}

	T func_pow(T x, T y)
	{
		return std::pow(x, y);
	}

	/// @brief Генерирует уникальный идентификатор для задачи
	/// @return Уникальный идентификатор
	size_t generate_id()
	{
		std::lock_guard<std::mutex> lock(id_generator);
		size_t old = global_id;
		global_id++;
		return old;
	}

	/// @brief Функция для каждого потока. Поток в цикле ожидает задачи от клиента на указанный порт.
	/// @param port Порт с которым работает поток
	void listen(int port)
	{
		param<T> p;
		T result;
		size_t type;
		while (true)
		{
			std::unique_lock<std::mutex> lock(*mutexes[port]);
			cvs[port]->wait(lock, [&] { return this->to_stop || !this->ports[port].empty(); });
			
			if (to_stop) break;

			p = this->ports[port].front();
			this->ports[port].pop();
			type = p.type;

			switch (type)
			{
				case 0: result = this->func_sin(p.args.first); break;
				case 1: result = this->func_sqrt(p.args.first); break;
				case 2: result = this->func_pow(p.args.first, p.args.second); break;
				default: break;
			}
			/// копируем данные результата в область данных структуры field
			std::memcpy(this->results[p.id].data, &result, sizeof(T));
			this->results[p.id].packed = true;
			cv_return.notify_all();
		}
	}
};

#endif