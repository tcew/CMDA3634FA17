#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "mpi.h"

typedef struct{
  // Network Properties

  // Number of Nodes in whole network
  int nodes;

  // Maximum number of connections each node can have
  int maxConnects;

  // Number of connections stored in struct
  int totalConnects;

  // Number of nodes this processor is responsible for
  int locNodes;

  // Relaxation constant
  double d;

  // Array pointers

  // Number of outbound connections array pointer
  int* connectCount;
  // Sources array pointers
  int* source;
  // Destinations array pointers
  int* dest;


  // Book keeping for send and recieve buffers
  int* sendCount; // Size of send
  int* recvCount; // nuber of recvs the process is to expect
  int* sendOffset; // Managing one array for all outgoing data
  int* recvOffset; // Managing one array for all incoming data
  int* recvLoc; // Companion of recvBuffer. Contins local node destinantion of received data

  int totalSends; // Total sends this processor needs to make
  int totalRecvs; // Total recvs this processor has to make

  // PageRank array pointers
  double* oldPageRank;
  double* newPageRank;

}Network;


Network mpi_networkReader(char* filename){

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if(rank == 0){
    /* If root, read from data and distribute to all processors.
       Each processor should have global variables like number of nodes/max connects,
       every connection where one of its nodes are involved
       and the number of outbound connections of each of its nodes. See assignment
       for spliting up node ownership and local indexing information.
    */


  }
  else{
    /* Recieve data from root */

  }
 
}



/* Book keeping method. Tabulates counts and sets up data transfers */
// Some order assumed: recv buffer contains chuncks of recv data in ascending rank order.                   
// That is, buffer looks like [Data from 0, Data from 1, ..., Data from n].                                 
// It is assumed that the order source/destination data is stored is the same on all processors.            
// This assumption saves us having to send companion destination data  
Network mpi_tally(Network net){
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  int source, dest;
  int sourceRank, destRank;
  int j;

  // Send count arrays so each process knows
  // How many sends/recvs needed with each other node
  net.sendCount = (int*)calloc(size, sizeof(int));
  net.recvCount = (int*)calloc(size, sizeof(int));
  
  // Tabulate size of sends/recvs this process needs to make each update
  for(j=0; j<net.totalConnects;j++){
    source = net.source[j];
    dest = net.dest[j];
    // Remember: (global node number) mod (number of processors) gives 
    // the processor the node belongs to. 
    sourceRank = source%size;
    // Similar to compute which rank destiination belongs to
    destRank = dest%size;
    if(sourceRank==rank && destRank!=rank){
      net.sendCount[destRank]++;
    }
    if(destRank==rank && sourceRank!=rank){
      net.recvCount[sourceRank]++;
    }
    
  }

  // Count total number of send/recvs this processor needs to make
  net.totalSends = 0;
  net.totalRecvs = 0;
  for(j=0; j<size; j++){
    net.totalSends += net.sendCount[j];
    net.totalRecvs += net.recvCount[j];
  }

  // Book keeping for sends and recevs
  // Sets offset markers to divide send/recv buffers
  // Effectivly reserves enough space for a buffer
  // to/from each processor
  net.sendOffset = (int*) calloc(size, sizeof(int));
  net.recvOffset = (int*) calloc(size, sizeof(int));
  for(j=0;j<size-1;j++){
    net.sendOffset[j+1] =  net.sendOffset[j]+net.sendCount[j];
    net.recvOffset[j+1] =  net.recvOffset[j]+net.recvCount[j];
  }

  // Book keeping for predetermining node destination of 
  // incoming data 
  net.recvLoc = (int*) calloc(net.totalRecvs, sizeof(int));
  int *count = (int*) calloc(size, sizeof(int));
  // Iterate through all connections once again.
  for(j=0; j<net.totalConnects;j++){
    source = net.source[j];
    dest = net.dest[j];
    destRank = dest%size;
    sourceRank = source%size;
    if(destRank==rank && sourceRank!=rank){
      *(net.recvLoc+(net.recvOffset[sourceRank]+count[sourceRank])) =  dest/size; 
      // store local node index of where data is going. We already know the data is on the right node
      count[sourceRank]++;
    }
  }

  return net;

}


Network  mpi_updatePageRank(Network net){

  // MPI variables
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Varriables
  int j, i, k;
  MPI_Status status;
  int tag = 999;
  int source; // current source
  int dest; // current destination
  int destRank; // Prosessor rank we need to send to
  int sourceRank; // Processor that owns source
  int marker;
  double PR;
  double L;

  // Buffers and counter needed for data transfer
  int *count = (int*) calloc(size, sizeof(int));
  double *recvBuffer = (double*) calloc(net.totalRecvs, sizeof(double));
  double *sendBuffer = (double*) calloc(net.totalSends, sizeof(double));

  /* For all connections */
  for(j=0;j<net.totalConnects;j++){
    source = net.source[j];
    dest = net.dest[j];
    destRank = dest%size;
    sourceRank = source%size;

    // If source belongs to this node, do something with the data
    if(sourceRank==rank){
      PR = net.oldPageRank[source/size];
      L = (double) net.connectCount[source/size];
      if(destRank == rank){
	// If dest also belongs to this node, update its PageRank
	net.newPageRank[dest/size] += net.d*(PR/L);
      }
      else{
	// If dest does not belong to this processor, prepare data to be sent
	*(sendBuffer+(net.sendOffset[destRank]+count[destRank])) = net.d*(PR/L);
	count[destRank]++;	
      }
    }

  }

  marker = 0;
  // Send / Rcv
  // For all processors, loops over whose turn it is to send
  for(j=0; j<size; j++){
    if(j==rank){
      // Send to everyone
      for(i=0; i < size; i++){
	if(i!=j && net.sendCount[i] > 0){ // sendCount to self will be zero, no worries
	  MPI_Send(sendBuffer+net.sendOffset[i], net.sendCount[i], MPI_DOUBLE, 
		   i, tag, MPI_COMM_WORLD); 
	}
	
      }
    }
    if(j!=rank && net.recvCount[j] > 0){
      // Recieve from j
      MPI_Recv(recvBuffer+marker, net.recvCount[j], MPI_DOUBLE, j, tag,
	       MPI_COMM_WORLD, &status);
      marker+=net.recvCount[j]; // move marker so we dont overwrite data on next recv
    }
    
  }
  int loc; // local node number of where recv data is going
  // Read through recved data and update PageRank of nodes the data belongs to
  for(j=0; j < net.totalRecvs; j++){
    loc = net.recvLoc[j];
    net.newPageRank[loc] += recvBuffer[j];  
  }
  
  /* Update complete. Free buffers */ 
  free(sendBuffer);
  free(recvBuffer);
  free(count);

  return net;
}



double mpi_computeDiff(Network net){
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  /* Compute local sum of square differences and performa a reduction
     to sum over all processors, then take square root. 
  */

}


Network mpi_computePageRank(Network net){

  double tol = 1e-6;
  double diff = 1;
  double* temp;
  int k;
  double nodes = (double) net.nodes;
  double constant = (1-net.d)/nodes;
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);


  /* Initialize old PR vlaues */
  for(k=0;k<net.locNodes;k++){
    net.oldPageRank[k] = 1/nodes;
  }

  while(diff > tol){
    /* Set nodes new PR values to constant */
    for(k=0;k<net.locNodes;k++){
      net.newPageRank[k] = constant;
    }

    /* Update PR based on connections */
    net = mpi_updatePageRank(net);
    
    /* Compute norm of difference  */ 
    diff = mpi_computeDiff(net);
    
    /* Have root node print diff */
    if(rank==0){
      printf("diff = %f\n", diff);
    }
    
    /* Switch pointers so updated values in oldPR array             
       Gets us ready to run again */
    temp = net.oldPageRank;
    net.oldPageRank = net.newPageRank;
    net.newPageRank = temp;
  }

  return net;
}

/* Free all (heap) arrays allocated. Make Valgrind happy */
void networkDestructor(Network net){
  free(net.connectCount);
  free(net.source);
  free(net.dest);
  free(net.sendCount);
  free(net.recvCount);
  free(net.sendOffset);
  free(net.recvOffset);
  free(net.oldPageRank);
  free(net.newPageRank);
  free(net.recvLoc);
}



int main(int argc, char** argv){


  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Reads the filename of the data file
  char* filename = argv[1];


  //Start your code
  int j;
  int k = 0;
  
  // Read in data, distribute to all processors
  Network net = mpi_networkReader(filename);
  net.d = 0.85;

  // Tabulation and book keeping for data exchange
  net = mpi_tally(net);

  net = mpi_computePageRank(net);
 
 for(j=0; j< net.nodes;j++){
   // If node belongs to this processor, print its PageRank  
   if(rank == j%size){ 
     printf("PageRank of node %d is: %f\n", j, net.oldPageRank[j/size]);
   }
 }
 
 // Program complete. Deconstruct network and finalize
 networkDestructor(net);
 MPI_Finalize();

 return 0;
}
