
#include <stdlib.h>
#include <stdio.h>
#include "mpi.h"


void barrier(){

  int rank, size, alive, tag = 999;

  int sizeBuffer = 1;
  int *sendBuffer = (int*) calloc(sizeBuffer, sizeof(int));
  int *recvBuffer = (int*) calloc(sizeBuffer, sizeof(int));

  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  /* root initializes message */
  if(rank==0)
    sendBuffer[0] = 10;
  
  /* number of "alive" processes */
  alive = 1;

  /* keep doing tree stuff until all processes are alive */
  while(alive<size){

    if(rank < alive){
      /* sender */
      int dest = rank + alive;

      /* is this a legal send */
      if(dest<size)
	MPI_Send(sendBuffer, sizeBuffer, MPI_INT, dest, tag, MPI_COMM_WORLD);
    }

    if(rank >= alive && rank < 2*alive){
      /* receiver */
      int source = rank - alive;

      /* is this a legal recv */
      if(source >= 0)
	MPI_Recv(recvBuffer, sizeBuffer, MPI_INT, source, tag, MPI_COMM_WORLD,
		 &status);
    }

    alive *= 2;

  }

}
