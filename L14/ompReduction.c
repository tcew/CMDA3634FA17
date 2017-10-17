
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char **argv){
  int threadCount = 20;

  omp_set_num_threads(threadCount);

  // still serial here
  int i = 6;

  int v[4], badness, veryBad = 0;
  
  // fork the program
#pragma omp parallel reduction(+:veryBad)
  {
    // stuff in this scope gets executed by all OpenMP threads
    int rank = omp_get_thread_num();
    int size = omp_get_num_threads();

    int foo = veryBad;
    
    if((rank+1)%2==0)
      printf("hello world from rank %d/%d\n",
	     rank, size);

    // good parallel
    v[rank] = rank;
    
    // "bad" parallel - expect badness
    badness = rank;

    veryBad = foo + rank;
    
  }

  int n;
  for(n=0;n<threadCount;++n){
    printf("v[%d]=%d\n", n, v[n]);
  }
  
  printf("veryBad = %d\n", veryBad);

  exit(0);
  return 0;

}
