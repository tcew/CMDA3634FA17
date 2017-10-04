#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv){

  long long int test = 0;
  long long int Ninside = 0; // number of random points inside 1/4 circle

  double newPi = 0, estPi = 0;
  double tol = 1e-8;
  
  do{
    int n, Ninnertests=10000;

    estPi = newPi;

    for(n=0;n<Ninnertests;++n){
      ++test;
      double x = drand48();
      double y = drand48();
      
      if(x*x+y*y<1){
	++Ninside;
      }
    }
    newPi = Ninside/(double)test;
    printf("newPi = %lf\n", newPi);
  }while(fabs(newPi-estPi)>tol);

  printf("estPi = %lf\n", 4.*newPi);

  return 0;
}
