#include "cuda_runtime.h"
#include <stdio.h>

__global__ void HellWorld()
{
	printf("Hello world, %d, %d\n", blockIdx.x, threadIdx.x);
}

int main()
{
	HellWorld<<<2,5>>>();
	cudaDeviceSynchronize();
	return 0;
}