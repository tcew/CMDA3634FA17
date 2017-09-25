#include <math.h>
#include <stdio.h>

// define a new variable type
typedef struct {
  float x;
  float y;
  float z;
} point;


point pointCreate(float x, float y, float z){

  point p;
  p.x = x;
  p.y = y;
  p.z = z;

  return p;
}

float pointDistanceFromOrigin(point p){

  float d = sqrt(p.x*p.x + p.y*p.y
		 + p.z*p.z);
  
  return d;
}

void pointPrint(point p){
  printf("point: %f,%f,%f\n", p.x, p.y, p.z);
}

void pointZero(point* pt_p){

  // opt 1:
  (*pt_p).x = 0;
  (*pt_p).y = 0;
  (*pt_p).z = 0;

  // opt 2:
  pt_p->x = 0;
  pt_p->y = 0;
  pt_p->z = 0;

  // opt 3:
  pt_p[0].x = 0;
  pt_p[0].y = 0;
  pt_p[0].z = 0;
}


int main(){


  point p = pointCreate(1.2, 2.3, 3.1);

  float d = pointDistanceFromOrigin(p);

  printf("d = %f\n", d);

  pointPrint(p);
  
  pointZero(&p);

  pointPrint(p);
  
}
