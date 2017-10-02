
#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

void deadlockV1(){
  MPI_Status status;
  int rank, tag= 99;

  double *recvBuffer = (double*) malloc(sizeof(double));
  double *sendBuffer = (double*) malloc(sizeof(double));
  
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);  

  // rank 0 and 1 try to send to each other
  if(rank==0){
    int dest = 1, source = 1;
    MPI_Recv(recvBuffer, 1, MPI_DOUBLE, source, tag, MPI_COMM_WORLD,
	     &status);
    MPI_Send(sendBuffer, 1, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD);
  }
  
  if(rank==1){
    int dest = 0, source = 0;
    MPI_Recv(recvBuffer, 1, MPI_DOUBLE, source, tag, MPI_COMM_WORLD,
	     &status);
    MPI_Send(sendBuffer, 1, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD);
  }
}

void undeadlockV1(){
  MPI_Status status;
  int rank, tag= 99;

  double *recvBuffer = (double*) malloc(sizeof(double));
  double *sendBuffer = (double*) malloc(sizeof(double));
  
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);  

  // rank 0 and 1 try to send to each other
  if(rank==0){
    int dest = 1, source = 1;
    MPI_Recv(recvBuffer, 1, MPI_DOUBLE, source, tag, MPI_COMM_WORLD,
	     &status);
    MPI_Send(sendBuffer, 1, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD);
  }
  
  if(rank==1){
    int dest = 0, source = 0;
    MPI_Send(sendBuffer, 1, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD);
    MPI_Recv(recvBuffer, 1, MPI_DOUBLE, source, tag, MPI_COMM_WORLD,
	     &status);

  }
}

void undeadlockV2(){
  MPI_Status status;
  MPI_Request sendRequest, recvRequest;
  
  int rank, tag= 99;

  double *recvBuffer = (double*) malloc(sizeof(double));
  double *sendBuffer = (double*) malloc(sizeof(double));
  
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);  

  // rank 0 and 1 try to send to each other
  if(rank==0){
    int dest = 1, source = 1;
    MPI_Irecv(recvBuffer, 1, MPI_DOUBLE, source, tag, MPI_COMM_WORLD,
	     &recvRequest);
    MPI_Isend(sendBuffer, 1, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD,
	      &sendRequest);
  }
  
  if(rank==1){
    int dest = 0, source = 0;
    MPI_Irecv(recvBuffer, 1, MPI_DOUBLE, source, tag, MPI_COMM_WORLD,
	      &recvRequest);
    MPI_Isend(sendBuffer, 1, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD,
	      &sendRequest);
  }

  /* block until message sent */
  MPI_Wait(&sendRequest, &status);

  /* block until message received */
  MPI_Wait(&recvRequest, &status);
}


int main(int argc, char **argv){

  MPI_Init(&argc, &argv);

  //  deadlockV1();
  //  undeadlockV1();
  undeadlockV2();
  
  MPI_Finalize();
  
  exit(0);
  return 0;
}
  
