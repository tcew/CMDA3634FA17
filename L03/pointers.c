#include <stdio.h>

/* every C program must have a main function */
int main(){

  int a; // reserved 4 bytes on the "stack"
  int b;
  double c; // reserve 8 bytes on stack

  /* create a pointer variable */
  int* pt_a;
  int* pt_b;
  
  pt_a = &a; // & finds the address of a variable
  pt_b = &b;

  *pt_a = 4;

  printf("a = %d\n", a);
  printf("pt_a = %p\n", pt_a);
  printf("pt_b = %p\n", pt_b);

  *(pt_a+1) = 6;
  printf("b = %d\n", b);
  return 0;
}
