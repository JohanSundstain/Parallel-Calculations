#include <iostream>
#include <cmath>
#include <memory>

#ifdef DOUBLE
using array_type = double;
#else
using array_type = float;
#endif

constexpr size_t size = 10000000;
constexpr array_type step = static_cast<array_type>((M_PI*2)/size); 

int main()
{
    std::unique_ptr<array_type> array {new array_type[size]};
    array_type* ptr = array.get();
	std::cout << step << std::endl;

	array_type sum = 0;
    for (size_t i = 0; i < size; i++)
    {
        ptr[i] = sin(i*step);
        sum += ptr[i];
    }

    std::cout << sum << std::endl;

    return 0;
}