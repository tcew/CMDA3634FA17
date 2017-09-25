#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

int main(int argc, char **argv){
  MPI_Status status;
  int message, rank, size, tag = 999;

  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD,
		&rank);

  MPI_Comm_size(MPI_COMM_WORLD,
		&size);
  
  if(rank==0 && rank+1<size)
    MPI_Send(&message, 1, MPI_INT, rank+1, tag, MPI_COMM_WORLD);
  else{
    if(rank-1>=0)
      MPI_Recv(&message, 1, MPI_INT, rank-1, tag, MPI_COMM_WORLD, &status);
    if(rank+1<size)
      MPI_Send(&message, 1, MPI_INT, rank+1, tag, MPI_COMM_WORLD);
  }
  
  
  if(rank==size-1 && rank-1>=0)
    MPI_Send(&message, 1, MPI_INT, rank-1, tag, MPI_COMM_WORLD);
  else{
    if(rank+1<size)
      MPI_Recv(&message, 1, MPI_INT, rank+1, tag, MPI_COMM_WORLD, &status);
    if(rank>0)
      MPI_Send(&message, 1, MPI_INT, rank-1, tag, MPI_COMM_WORLD);
  }
  
  if(rank==0) printf("process 0 got through barrier\n");
  
  MPI_Finalize();
  exit(0);
  return 0;
}
