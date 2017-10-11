#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

int main(int argc, char **argv){

  long long int test = 0;
  long long int Ninside = 0; // number of random points inside 1/4 circle

  double newPi = 0, estPi = 0;
  double tol = 1e-8;

  int rank, size;
  
  long long int *sendBuffer
    = (long long int*) malloc(sizeof(long long int));
  long long int *recvBuffer
    = (long long int*) malloc(sizeof(long long int));
  int count = 1;
  
  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // seed random number generator so that each process
  // is running a different sequence of tests
  srand48(rank);
  
  do{
    int n, Ninnertests=10000/size;

    estPi = newPi;

    for(n=0;n<Ninnertests;++n){
      ++test;
      double x = drand48();
      double y = drand48();
      
      if(x*x+y*y<1){
	++Ninside;
      }
    }

    sendBuffer[0] = Ninside;
    
    MPI_Allreduce(sendBuffer, recvBuffer, count,
                  MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);

    newPi = 4.*recvBuffer[0]/(double)(test*size);
    printf("newPi = %lf\n", newPi);
  }while(fabs(newPi-estPi)>tol);

  if(rank==0)
    printf("estPi = %lf\n", 4.*newPi);

  MPI_Finalize();
  
  return 0;
}
