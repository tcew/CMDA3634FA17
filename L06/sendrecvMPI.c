#include <stdlib.h>
#include <stdio.h>
#include "mpi.h"

int main(int argc, char **argv){

  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD,	&size);

  int  dataCount = 10, n;
  int sourceRank = 0, destRank = 1;
  int tag = 999;

  if(rank==sourceRank){

    int *dataOut =
      (int*) malloc(dataCount*sizeof(int));

    for(n=0;n<dataCount;++n){
      dataOut[n] = 2*n;
    }

    MPI_Send(dataOut,
	     dataCount,
	     MPI_INT,
	     destRank,
	     tag,
	     MPI_COMM_WORLD);

    free(dataOut);
  }

  if(rank==destRank){
    /* receive message */
    MPI_Status status;
    
    int *dataIn =
      (int*) malloc(dataCount*sizeof(int));

    MPI_Recv(dataIn,
	     dataCount,
	     MPI_INT,
	     sourceRank,
	     tag,
	     MPI_COMM_WORLD,
	     &status);

    printf("rank %d got this message \n",
	   rank);
    for(n=0;n<dataCount;++n){
      printf("%d\n", dataIn[n]);
    }
    
    free(dataIn);
  }
  
  MPI_Finalize();
  exit(0);
  return 0;
}
