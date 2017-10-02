
#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

int main(int argc, char **argv){

  int rank, size;

  MPI_Init(&argc, &argv);
  
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);


  if(rank==0){
    int dest = 1;
    int count = 1;
    int tag = 999;
    int *sendBuffer = (int*) malloc(count*sizeof(int));
    sendBuffer[0] = 911;

    MPI_Send(sendBuffer, count, MPI_INT, dest, tag, MPI_COMM_WORLD);
  }

  if(rank==1){

    MPI_Status status;

    MPI_Probe(MPI_ANY_SOURCE,
	      MPI_ANY_TAG,
	      MPI_COMM_WORLD, &status);

    int source = status.MPI_SOURCE;
    int tag = status.MPI_TAG;

    int count;
    MPI_Get_count(&status, MPI_INT, &count);

    int *recvBuffer = (int*) malloc(count*sizeof(int));
    MPI_Recv(recvBuffer, count, MPI_INT, source, tag, MPI_COMM_WORLD,
	     &status);

    printf("received %d from rank %d with tag %d\n",
	   recvBuffer[0], source, tag);
    
  }
  
  MPI_Finalize();
  return 0;

}
