#include<stdio.h>
#include<stdlib.h>
#include "cuda.h"

__global__ void addVectors(int N, double *a, double *b, double *c) {

  int thread = threadIdx.x;
  int block = blockIdx.x;
  int blockSize = blockDim.x;

  int id = block*blockSize + thread;

  __shared__ double s_a[32];
  __shared__ double s_b[32];
  __shared__ double s_c[32];

  //populate shared memory cache
  if (id<N) {
    s_a[thread] = a[id];
    s_b[thread] = b[id];
  }

  __syncthreads(); //make sure all threads have written to cache

  //perform the addition
  s_c[thread] = s_a[thread] + s_b[thread];
  
  if (id<N) {
    c[id] = s_c[thread];
  }
}

int main (int argc, char** argv) {

  int N = 100000; //vector size

  int Nthreads = 32; //number of threads per block
  int Nblocks = (N+Nthreads-1)/Nthreads; //WARNING integer division here

  double *h_a, *h_b, *h_c; //host vectors

  //allocate vectors
  h_a = (double *) malloc(N*sizeof(double));
  h_b = (double *) malloc(N*sizeof(double));
  h_c = (double *) malloc(N*sizeof(double));

  double *d_a, *d_b, *d_c; //device arrays

  //allocate memory on device
  cudaMalloc(&d_a, N*sizeof(double));
  cudaMalloc(&d_b, N*sizeof(double));
  cudaMalloc(&d_c, N*sizeof(double));

  //populate our vectors a and b
  for (int n=0;n<N;n++) {
    h_a[n] = n;
    h_b[n] = N-n;
  }

  //copy a and b to device
  cudaMemcpy(d_a,h_a,N*sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(d_b,h_b,N*sizeof(double),cudaMemcpyHostToDevice);

  // c = a+b
  // for (int n=0;n<N;n++) {
  //   c[n] = a[n] + b[n];
  // }

  //add the vectors on the device
  addVectors<<<Nblocks,Nthreads>>>(N,d_a,d_b,d_c);

  cudaMemcpy(h_c,d_c,N*sizeof(double),cudaMemcpyDeviceToHost);

  int printId = 0;
  printf("c[%d] = %f \n", printId, h_c[printId]);

  //free up device memory
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  //free up memory
  free(h_a);
  free(h_b);
  free(h_c);
}

