
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char **argv){
  int threadCount = 20;

  omp_set_num_threads(threadCount);

  // still serial here
  int i = 6;

  int v[4], veryBad = 0;
  
  // fork the program
#pragma omp parallel firstprivate(i)
  {
    // stuff in this scope gets executed by all OpenMP threads
    int rank = omp_get_thread_num();
    int size = omp_get_num_threads();

    i = i + rank;
    
    // good parallel
#pragma omp critical
    {
      // force threads to take turn based on rank
      v[rank] = rank;
      //      printf("v[%d]=%d\n", rank, v[rank]);
      veryBad += rank;
      printf("veryBad = %d (from rank %d)\n",
	     veryBad, rank);
      printf("i=%d\n", i);
    }
    

  }

  printf("veryBad = %d\n", veryBad);

  exit(0);
  return 0;

}
