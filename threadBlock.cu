#include <iostream>
#include <cuda_runtime.h>

int main()
{
	// Confirming cuda device property
    cudaDeviceProp dProp;
    cudaGetDeviceProperties(&dProp, 0);
    std::cout << "Device: " << dProp.name << std::endl;
    std::cout << "Maximum number of threads per block:" << dProp.maxThreadsPerBlock << std::endl;
    std::cout << "Max dimension size of a thread block (x,y,z): (" << dProp.maxThreadsDim[0] << "," 
        << dProp.maxThreadsDim[1] << "," << dProp.maxThreadsDim[2]  << ")"<<  std::endl;
    std::cout << "Max dimension size of a grid size    (x,y,z): (" << dProp.maxGridSize[0] << ","
        << dProp.maxGridSize[1] << "," << dProp.maxGridSize[2] << ")" << std::endl;
    return 0;
}