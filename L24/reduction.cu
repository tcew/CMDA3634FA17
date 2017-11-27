
// compile with: nvcc -arch sm_60 -o reduction reduction.cu
// run: ./reduction

#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"

// use this later to define number of threads in thread block
#define BSIZE 256

__global__ void partialReduction_v0(int N, 
				    float *c_a,
				    float *c_result){

  // shared memory array
  __shared__ float s_a[BSIZE];

  // find thread number in thread-block
  int t = threadIdx.x;
  
  // find block number
  int b = blockIdx.x;

  // choose an array index for this thread to read
  int n = t + b*blockDim.x;

  // check is this index in bounds
  float a = 0;
  if(n<N)
    a = c_a[n];
  
  // store the entry in shared memory
  s_a[t] = a;

  // block until all threads have written to the shared memory
  __syncthreads();
  if(t<BSIZE/2) s_a[t] = s_a[t] + s_a[t+(BSIZE/2)];

  __syncthreads();
  if(t<BSIZE/4) s_a[t] = s_a[t] + s_a[t+(BSIZE/4)];

  __syncthreads();
  if(t<BSIZE/8) s_a[t] = s_a[t] + s_a[t+(BSIZE/8)];

  __syncthreads();
  if(t<BSIZE/16) s_a[t] = s_a[t] + s_a[t+(BSIZE/16)];

  __syncthreads();
  if(t<BSIZE/32) s_a[t] = s_a[t] + s_a[t+(BSIZE/32)];

  __syncthreads();
  if(t<BSIZE/64) s_a[t] = s_a[t] + s_a[t+(BSIZE/64)];

  __syncthreads();
  if(t<BSIZE/128) s_a[t] = s_a[t] + s_a[t+(BSIZE/128)];

  __syncthreads();
  if(t<BSIZE/256) s_a[t] = s_a[t] + s_a[t+(BSIZE/256)];

  if(t==0)
    c_result[b] = s_a[0];
}

__global__ void partialReduction_v1(const int N, 
				    const float * __restrict__ c_a,
				    float * __restrict__ c_result){
  
  // shared memory array
  __shared__ float s_a[BSIZE];

  // find thread number in thread-block
  int t = threadIdx.x;
  
  // find block number
  int b = blockIdx.x;

  // choose an array index for this thread to read
  int n = t + b*blockDim.x;

  // check is this index in bounds
  float a = 0;
  if(n<N)
    a = c_a[n];
  
  // store the entry in shared memory
  s_a[t] = a;

  // block until all threads have written to the shared memory
  __syncthreads();
  if(t<BSIZE/2) s_a[t] = s_a[t] + s_a[t+(BSIZE/2)];

  __syncthreads();
  if(t<BSIZE/4) s_a[t] = s_a[t] + s_a[t+(BSIZE/4)];

  __syncthreads();
  if(t<BSIZE/8) s_a[t] = s_a[t] + s_a[t+(BSIZE/8)];

  __syncthreads();
  if(t<BSIZE/16) s_a[t] = s_a[t] + s_a[t+(BSIZE/16)];

  __syncthreads();
  if(t<BSIZE/32) s_a[t] = s_a[t] + s_a[t+(BSIZE/32)];

  __syncthreads();
  if(t<BSIZE/64) s_a[t] = s_a[t] + s_a[t+(BSIZE/64)];

  __syncthreads();
  if(t<BSIZE/128) s_a[t] = s_a[t] + s_a[t+(BSIZE/128)];

  __syncthreads();
  if(t<BSIZE/256) s_a[t] = s_a[t] + s_a[t+(BSIZE/256)];

  if(t==0)
    c_result[b] = s_a[0];
}

// use "warp synchronization"
__global__ void partialReduction_v2(const int N, 
				    const float * __restrict__ c_a,
				    float * __restrict__ c_result){
  
  // shared memory array
  __volatile__ __shared__ float s_a[BSIZE];

  // find thread number in thread-block
  int t = threadIdx.x;
  
  // find block number
  int b = blockIdx.x;

  // choose an array index for this thread to read
  int n = t + b*blockDim.x;

  // check is this index in bounds
  float a = 0;
  if(n<N)
    a = c_a[n];
  
  // store the entry in shared memory
  s_a[t] = a;

  // block until all threads have written to the shared memory
  __syncthreads();
  if(t<BSIZE/2) s_a[t] = s_a[t] + s_a[t+(BSIZE/2)];

  __syncthreads();
  if(t<BSIZE/4) s_a[t] = s_a[t] + s_a[t+(BSIZE/4)];

  __syncthreads();
  if(t<BSIZE/8) s_a[t] = s_a[t] + s_a[t+(BSIZE/8)];

  //  __syncthreads();
  if(t<BSIZE/16) s_a[t] = s_a[t] + s_a[t+(BSIZE/16)];

  //  __syncthreads();
  if(t<BSIZE/32) s_a[t] = s_a[t] + s_a[t+(BSIZE/32)];

  //  __syncthreads();
  if(t<BSIZE/64) s_a[t] = s_a[t] + s_a[t+(BSIZE/64)];

  //  __syncthreads();
  if(t<BSIZE/128) s_a[t] = s_a[t] + s_a[t+(BSIZE/128)];

  //  __syncthreads();
  if(t<BSIZE/256) s_a[t] = s_a[t] + s_a[t+(BSIZE/256)];

  if(t==0)
    c_result[b] = s_a[0];
}

// issue more loads per thread, and use threads
__global__ void partialReduction(const int N, 
				 const float * __restrict__ c_a,
				 float * __restrict__ c_result){
  
  // shared memory array
  __volatile__ __shared__ float s_a[BSIZE];

  // find thread number in thread-block
  int t = threadIdx.x;
  
  // find block number
  int b = blockIdx.x;

  // choose an array index for this thread to read
  int n = t + b*blockDim.x;

  // check is this index in bounds
  float a = 0;
  while(n<N){
    a += c_a[n];
    n += blockDim.x*gridDim.x;
  }
    
  // store the entry in shared memory
  s_a[t] = a;

  // block until all threads have written to the shared memory
  __syncthreads();
  if(t<BSIZE/2) s_a[t] = s_a[t] + s_a[t+(BSIZE/2)];

  __syncthreads();
  if(t<BSIZE/4) s_a[t] = s_a[t] + s_a[t+(BSIZE/4)];

  __syncthreads();
  if(t<BSIZE/8) s_a[t] = s_a[t] + s_a[t+(BSIZE/8)];

  //  __syncthreads();
  if(t<BSIZE/16) s_a[t] = s_a[t] + s_a[t+(BSIZE/16)];

  //  __syncthreads();
  if(t<BSIZE/32) s_a[t] = s_a[t] + s_a[t+(BSIZE/32)];

  //  __syncthreads();
  if(t<BSIZE/64) s_a[t] = s_a[t] + s_a[t+(BSIZE/64)];

  //  __syncthreads();
  if(t<BSIZE/128) s_a[t] = s_a[t] + s_a[t+(BSIZE/128)];

  //  __syncthreads();
  if(t<BSIZE/256) s_a[t] = s_a[t] + s_a[t+(BSIZE/256)];

  if(t==0)
    c_result[b] = s_a[0];
}


int main(int argc, char **argv){

  int N = atoi(argv[argc-1]);

  // host array
  float *h_a = (float*) malloc(N*sizeof(float));
  float *h_result = (float*) malloc(N*sizeof(float));

  int n;
  for(n=0;n<N;++n){
    h_a[n] = 1;
    h_result[n] = 0;
  }

  // allocate device array
  float *c_a, *c_result;

  cudaMalloc(&c_a, N*sizeof(float));
  cudaMalloc(&c_result, N*sizeof(float));

  // copy data from host to device
  cudaMemcpy(c_a, h_a, N*sizeof(float), cudaMemcpyHostToDevice);

  // must zero
  cudaMemcpy(c_result, h_result, N*sizeof(float), cudaMemcpyHostToDevice);

  // choose number of threads in thread-block
  dim3 B(BSIZE,1,1);

  // choose number of thread-blocks
  int Nblocks = (N+BSIZE-1)/BSIZE;

  int Nblocks1 = (Nblocks+11)/12;
  dim3 G1(Nblocks1,1,1);
  
  printf("Nblocks1 = %d\n", Nblocks1);

  // launch reduction kernel
  partialReduction <<< G1, B >>> (N, c_a, c_result);
  
  cudaMemcpy(h_result, c_result,  Nblocks1*sizeof(float), cudaMemcpyDeviceToHost);
  
  // print out partial sums
  float res = 0;
  for(n=0;n<Nblocks1;++n){
    printf("%f\n", h_result[n]);
    res += h_result[n];
  }

  printf("res = %f\n", res);
  
  return 0;
}
