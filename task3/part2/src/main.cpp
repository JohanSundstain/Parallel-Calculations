#include <iostream>
#include <thread>

#include "Server.cpp"
#include "Client.cpp"

int main()
{
	Client<double> c{ ClientType::Sin};
	Client<double> c1{ ClientType::Sqrt };
	Client<double> c2{ ClientType::Pow };
	std::shared_ptr<Server<double>> server = std::make_shared<Server<double>>(10);

	c.set_server(server);
	c1.set_server(server);
	c2.set_server(server);

	server->start();

	std::thread tc{ c };
	std::thread tc1{ c1 };
	std::thread tc2{ c2 };

	tc.join();
	tc1.join();
	tc2.join();

	server->stop();
	return 0;
}