
#include <stdio.h>
#include <stdlib.h>

__global__ void helloKernel(){

  int thread = threadIdx.x;
  int block  = blockIdx.x;

  printf("hello from thread %d in block %d\n", thread, block);

}


int main(int argc, char **argv){

  int B = 3; // number of thread-blocks
  int T = 4; // number of threads per thread-block

  helloKernel <<< B, T >>> ();

  cudaDeviceSynchronize(); 

  return 0;
}
