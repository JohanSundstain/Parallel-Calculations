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
#include <memory>
#include <functional>
#include <future>

class Server
{
public:
	Server(size_t num_workers)
	{
		this->num_threads = num_workers;
		this->to_stop = true;
	}

	~Server()
	{
		if (!to_stop)
		{
			this->stop();
		}
	}

	/// @brief Создает пул потоков, которые вызывают функцию listen
	void start()
	{
		this->to_stop = false;
		for (size_t worker = 0; worker < this->num_threads; worker++)
		{
			this->thread_pool.emplace_back(&Server::listen, this);
		}
	}

	/// @brief Останавливает потоки
	void stop()
	{
		this->to_stop = true;

		this->cv_ready_task.notify_all();

		for (std::thread& t : this->thread_pool)
		{
			t.join();
		}
	}

	/// @brief Функция принимает фунцию с произвольным количеством аргументов
	/// @tparam Func - Тип переданной функции
	/// @tparam ...Args - Универсальные ссылки на передаваемые аргументы для функции
	/// @param func Универсальная ссылка на функцию
	/// @param ...args Аргументы функции
	/// @return Возвращает объект future вычисляемого типа
	template <typename Func, typename... Args>
	auto add_task(Func&& func, Args&&... args) -> std::future<decltype(func(args...))>
	{
		using return_type = decltype(func(args...));

		/// Создаем shared_ptr на задачу, передаем бинд функции и её переданных аргументов
		/// Пробрасываем forward для правильной обработки rvalue и lvalue
		auto task = std::make_shared<std::packaged_task<return_type()>>(
			std::bind(std::forward<Func>(func), std::forward<Args>(args)...)
		);

		std::future<return_type> result = task->get_future();

		{
			/// Кладем в очередь лямбда выражение с захваченным контекстом 
			/// При вызове этой лямбда функции вызовется функция нашей задачи
			std::lock_guard<std::mutex> lock(queue_mutex);
			task_queue.push([task]() {(*task)(); });
		}

		cv_ready_task.notify_one();
		return std::move(result);
	}

private:
	bool to_stop;
	size_t num_threads;
	std::vector<std::thread> thread_pool;
	std::queue<std::function<void()>> task_queue;
	std::mutex queue_mutex;
	std::condition_variable cv_ready_task;

	/// @brief Фоновая функция для пула потоков
	void listen()
	{
		std::function<void()> task;

		while (true)
		{
			{
				std::unique_lock<std::mutex> lock(queue_mutex);
				this->cv_ready_task.wait(lock, [&] { return this->to_stop || !task_queue.empty(); });

				if (to_stop) break;

				task = std::move(task_queue.front());
				task_queue.pop();
			}
			task();
		}
	}
};

#endif