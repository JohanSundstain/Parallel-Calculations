#include <iostream>
#include <thread>

#include "Server.cpp"
#include "Client.cpp"


struct alignas(64) field2
{
	bool packed;
	char data[63];
};


int main()
{
	Client<double> c{ ClientType::Sin};
	Client<double> c1{ ClientType::Sqrt };
	Client<double> c2{ ClientType::Pow };
	std::shared_ptr<Server> server = std::make_shared<Server>(10);

	c.set_server(server);
	c1.set_server(server);
	c2.set_server(server);

	server->start();

	std::thread tc{ std::ref(c) };
	std::thread tc1{ std::ref(c1) };
	std::thread tc2{ std::ref(c2) };

	tc.join();
	tc1.join();
	tc2.join();

	server->stop();
	return 0;
}