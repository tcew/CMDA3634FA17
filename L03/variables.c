#include <stdio.h>

/* every C program must have a main function */
int main(){

  int a; // reserved 4 bytes on the "stack"
  int b;
  double c; // reserve 8 bytes on stack
  a = 3;
  b = 6;
  c = 1.2;

  printf("a=%d\n", a);
  printf("b=%d\n", b);
  printf("a+b=%d\n", a+b);
  printf("a-b=%d\n", a-b);
  printf("a/b=%d\n", a/b);
  printf("a*b=%d\n", a*b);

  printf("c=%lf\n", c);
  printf("b=%d\n", b);
  printf("c+b=%lf\n", c+b);
  printf("c-b=%lf\n", c-b);
  printf("c/b=%lf\n", c/b);
  printf("c*b=%lf\n", c*b);
  
  return 0;
  
  return 0;
  
}
