/* 

To compile:

   gcc -O3 -o mandelbrot mandelbrot.c png_util.c -I. -lpng -lm

To create an image with 4096 x 4096 pixels (last argument will be used to set number of threads):

    ./mandelbrot 4096 4096 1

*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "png_util.h"

// Q2a: add include for CUDA header file here:

#define MXITER 1000

typedef struct {
  
  double r;
  double i;
  
}complex_t;

// return iterations before z leaves mandelbrot set for given c
int testpoint(complex_t c){

  int iter;
  complex_t z;
  double temp;
  
  z = c;
  
  for(iter=0; iter<MXITER; iter++){  
    temp = (z.r*z.r) - (z.i*z.i) + c.r;
    
    z.i = z.r*z.i*2. + c.i;
    z.r = temp;
    
    if((z.r*z.r+z.i*z.i)>4.0){
      return iter;
    }
  }
  return iter; 
}

// perform Mandelbrot iteration on a grid of numbers in the complex plane
// record the  iteration counts in the count array
void mandelbrot(int Nre, int Nim, complex_t cmin, complex_t dc, float *count){ 

  // Q2c: replace this loop with a CUDA kernel
  for(int n=0;n<Nim;++n){
    for(int m=0;m<Nre;++m){
      complex_t c;

      c.r = cmin.r + dc.r*m;
      c.i = cmin.i + dc.i*n;
      
      count[m+n*Nre] = (float) testpoint(c);
    }
  }
}

int main(int argc, char **argv){

  // to create a 4096x4096 pixel image [ last argument is placeholder for number of threads ] 
  // usage: ./mandelbrot 4096 4096 32 

  int Nre = atoi(argv[1]);
  int Nim = atoi(argv[2]);
  int Nthreads = atoi(argv[3]);

  // Q2b: set the number of threads per block and the number of blocks here:
  
  // storage for the iteration counts
  float *count;
  count = (float*) malloc(Nre*Nim*sizeof(float));

  // Parameters for a bounding box for "c" that generates an interesting image
  const float centRe = -.759856, centIm= .125547;
  const float diam  = 0.151579;

  complex_t cmin; 
  complex_t cmax;
  complex_t dc;

  cmin.r = centRe - 0.5*diam;
  cmax.r = centRe + 0.5*diam;
  cmin.i = centIm - 0.5*diam;
  cmax.i = centIm + 0.5*diam;

  //set step sizes
  dc.r = (cmax.r-cmin.r)/(Nre-1);
  dc.i = (cmax.i-cmin.i)/(Nim-1);

  clock_t start = clock(); //start time in CPU cycles

  // compute mandelbrot set
  mandelbrot(Nre, Nim, cmin, dc, count); 
  
  // copy from the GPU back to the host here

  clock_t end = clock(); //start time in CPU cycles
  
  // print elapsed time
  printf("elapsed = %f\n", ((double)(end-start))/CLOCKS_PER_SEC);

  // output mandelbrot to png format image
  FILE *fp = fopen("mandelbrot.png", "w");

  printf("Printing mandelbrot.png...");
  write_hot_png(fp, Nre, Nim, count, 0, 80);
  printf("done.\n");

  free(count);

  exit(0);
  return 0;
}  