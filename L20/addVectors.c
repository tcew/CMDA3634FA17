#include<stdio.h>
#include<stdlib.h>

int main (int argc, char** argv) {

	int N = 100000; //vector size

  double *a, *b, *c; //vectors

  //allocate vectors
  a = (double *) malloc(N*sizeof(double));
  b = (double *) malloc(N*sizeof(double));
  c = (double *) malloc(N*sizeof(double));

  //populate our vectors a and b
  for (int n=0;n<N;n++) {
    a[n] = n;
    b[n] = N-n;
  }

  // c = a+b
  for (int n=0;n<N;n++) {
    c[n] = a[n] + b[n];
  }

  int printId = 0;
  printf("c[%d] = %f \n", printId, c[printId]);

  //free up memory
	free(a);
  free(b);
  free(c);
}

