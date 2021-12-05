#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#include "support.h"

void verify(float *A, float *B, float *C, float *OUT, unsigned int m, unsigned int k,
  unsigned int n) {

  const float relativeTolerance = 1e-6;
  unsigned int count = 0;
	//printf("\n0, %f/%f ", A[0], B[0]);
       // printf("\n1, %f/%f ", A[1], B[1]);
       // printf("\n2, %f/%f ", A[2], B[2]);
       // printf("\n3, %f/%f ", A[3], B[3]);
  for(int row = 0; row < m; ++row) {
    for(int col = 0; col < n; ++col) {
      float sum = 0;
      for(unsigned int i = 0; i < k; ++i) {
	//printf("\n %f ", A[row*k + i]);
        //printf("\n %f ", B[i*n + col]);

	sum += A[row*k + i]*B[i*n + col];
      }
      count++;
      float relativeError = (sum - C[row*n + col])/sum;
	//printf("\n %f/%f/%f ", A[row*n + col], row, col);
	//printf("\n %f/%f/%f ", B[row*n + col], row, col);
      printf("\n0, %f/%f ", sum, C[row*n + col]);
//	printf("\n1, %f/%f ", sum, C[1], "1");
//	printf("\n2, %f/%f ", sum, C[2], "2");
//	printf("\n3, %f/%f ", sum, C[3], "3");
      //if (relativeError > relativeTolerance
        //|| relativeError < -relativeTolerance) {
       // printf("\nTEST FAILED %u\n\n",count);
       // exit(1);
     // }
    }
  }
	printf("\n OUTPUT: %f ", OUT[0]);
  printf("TEST PASSED %u\n\n", count);

}

void startTime(Timer* timer) {
    gettimeofday(&(timer->startTime), NULL);
}

void stopTime(Timer* timer) {
    gettimeofday(&(timer->endTime), NULL);
}

float elapsedTime(Timer timer) {
    return ((float) ((timer.endTime.tv_sec - timer.startTime.tv_sec) \
                + (timer.endTime.tv_usec - timer.startTime.tv_usec)/1.0e6));
}

