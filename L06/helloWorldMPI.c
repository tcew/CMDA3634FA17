#include <stdlib.h>
#include <stdio.h>
#include "mpi.h"

int main(int argc, char **argv){

  /* initialize MPI as soon as
     you enter your main function */
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD,
		&rank);
  MPI_Comm_size(MPI_COMM_WORLD,
		&size);
  
  printf("hello from process %0d/%0d\n",
	 rank, size);

  // only rank 0 does this
  int i;
  double j=0;
  for(i=0;i<100;++i){
    j += i/(double)(rank+1);
  }
  printf("rank %d computed j=%lf\n",
	 rank, j);

  /* all instances of this MPI
     program must enter the
     "finalize" function before
     exiting */
  
  MPI_Finalize();
  exit(0);
  return 0;
}
