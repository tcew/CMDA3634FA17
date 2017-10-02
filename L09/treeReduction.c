
#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

/* sum up the val variables over all processes */
double treeReduction(double val){

  int rank, size, tag = 999;
  
  double *sendBuffer =
    (double*) malloc(sizeof(double));
  double *recvBuffer =
    (double*) malloc(sizeof(double));

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  
  sendBuffer[0] = val;
  
  int alive = 1;
  while(alive<size) // round up to nearest power of 2
    alive *= 2;
  
  while(alive>1){

    if(rank>=alive/2){
      int dest = rank - (alive/2); 
      // MPI_Send to rank - (alive/2);
      if(dest>=0)
	MPI_Send(sendBuffer,
		 1,
		 MPI_DOUBLE,
		 dest,
		 tag,
		 MPI_COMM_WORLD);
    }
    
    if(rank<(alive/2)){
      MPI_Status status;
      int source = rank + (alive/2); // round up
      
      //      MPI_Recv from rank + (alive/2);
      if(source<size)
	MPI_Recv(recvBuffer,
		 1,
		 MPI_DOUBLE,
		 source,
		 tag,
		 MPI_COMM_WORLD,
		 &status);
      // add incoming data to our data;
      sendBuffer[0] += recvBuffer[0];
    }
    
    alive = alive/2;
  }
  return sendBuffer[0];
  
}

int main(int argc, char **argv){

  MPI_Init(&argc, &argv);

  double val = 1;

  double reducedVal = treeReduction(val);

  int rank;
  
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  printf("rank %d got %lf\n", rank, reducedVal);

  MPI_Finalize();

  exit(0);
  return 0;
}
  
