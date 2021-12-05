#include <stdio.h>
#include <stdlib.h>
#include "kernel.cu"
#include "support.h"
#include <iostream>
#include <cublas_v2.h>


void cublas_sgemm(int m, int n, int k, const float *A, const float *B, float *C){
	int lda=m,ldb=k,ldc=m;
        const float alf = 1;
        const float bet = 0;
        const float *alpha = &alf;
        const float *beta = &bet;

        // Create a handle for CUBLAS
        cublasHandle_t handle;
        cublasCreate(&handle);

        // Do the actual multiplication
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, alpha, B, ldb, A, lda, beta, C, ldc);

        // Destroy the handle
        cublasDestroy(handle);

}

void cublas_out_sgemm(int m, int n, int k, const float *OW, float *C, float *OUT){
        int lda=m,ldb=k,ldc=m;
        const float alf = 1;
        const float bet = 0;
        const float *alpha = &alf;
        const float *beta = &bet;

        // Create a handle for CUBLAS
        cublasHandle_t handle;
        cublasCreate(&handle);

        // Do the actual multiplication
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, alpha, C, ldb, OW, lda, beta, OUT, ldc);

        // Destroy the handle
        cublasDestroy(handle);

} 

void sigmoid(float *C){
	
}

int main (int argc, char *argv[])
{

    Timer timer;
    cudaError_t cuda_ret;

    // Initialize host variables ----------------------------------------------

    printf("\nSetting up the problem..."); fflush(stdout);
    startTime(&timer);

    float *A_h, *B_h, *C_h, *Bias_h, *OW_h, *OUT_h;
    float *A_d, *B_d, *C_d, *Bias_d, *OW_d, *OUT_d;
    size_t A_sz, B_sz, C_sz, Bias_sz, OW_sz, OUT_sz;
    int len, OUT_len;
    unsigned matArow, matAcol;
    unsigned matBrow, matBcol;
    dim3 dim_grid, dim_block;

    if (argc == 1) {
        matArow = 1000;
        matAcol = matBrow = 1000;
        matBcol = 1000;
    } else if (argc == 2) {
        matArow = atoi(argv[1]);
        matAcol = matBrow = atoi(argv[1]);
        matBcol = atoi(argv[1]);
    } else if (argc == 4) {
        matArow = atoi(argv[1]);
        matAcol = matBrow = atoi(argv[2]);
        matBcol = atoi(argv[3]);
    } else {
        printf("\n    Invalid input parameters!"
      "\n    Usage: ./sgemm-tiled                # All matrices are 1000 x 1000"
      "\n    Usage: ./sgemm-tiled <m>            # All matrices are m x m"
      "\n    Usage: ./sgemm-tiled <m> <k> <n>    # A: m x k, B: k x n, C: m x n"
      "\n");
        exit(0);
    }
   
    A_sz = matArow*matAcol;
    B_sz = matBrow*matBcol;
    C_sz = matArow*matBcol;
    Bias_sz = C_sz;
    OW_sz = matArow*matBcol;
    OUT_sz = matBcol;
    len = C_sz;
    OUT_len = OUT_sz;

    A_h = (float*) malloc( sizeof(float)*A_sz );
    for (unsigned int i=0; i < A_sz; i++) { A_h[i] = (rand()%100)/100.00; }

    B_h = (float*) malloc( sizeof(float)*B_sz );
    for (unsigned int i=0; i < B_sz; i++) { B_h[i] = (rand()%100)/100.00; }

    Bias_h = (float*) malloc( sizeof(float)*Bias_sz);
    for (unsigned int i = 0; i < Bias_sz; i++) { Bias_h[i] = (rand()%100)/100.00; }

    OW_h = (float*) malloc(sizeof(float)*OW_sz);
    for(unsigned int i = 0; i < OW_sz; i++) { OW_h[i] = (rand()%100)/100.00; }

    C_h = (float*) malloc( sizeof(float)*C_sz );

    OUT_h = (float*) malloc(sizeof(float)*OUT_sz);

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    printf("    A: %u x %u\n    B: %u x %u\n    C: %u x %u\n", matArow, matAcol,
        matBrow, matBcol, matArow, matBcol);

    // Allocate device variables ----------------------------------------------

    printf("Allocating device variables..."); fflush(stdout);
    startTime(&timer);

    /*************************************************************************/
    //INSERT CODE HERE
	cudaMalloc((void**) &A_d, sizeof(float)*A_sz);
	cudaMalloc((void**) &B_d, sizeof(float)*B_sz);
	cudaMalloc((void**) &C_d, sizeof(float)*C_sz);
	cudaMalloc((void**) &OW_d, sizeof(float)*OW_sz);
	cudaMalloc((void**) &OUT_d, sizeof(float)*OUT_sz);
    /*************************************************************************/
	
    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy host variables to device ------------------------------------------
    printf("Copying data from host to device..."); fflush(stdout);
    startTime(&timer);
	
    /*************************************************************************/
    //INSERT CODE HERE
	cudaMemcpy(A_d, A_h, sizeof(float)*A_sz, cudaMemcpyHostToDevice);
	cudaMemcpy(B_d, B_h, sizeof(float)*B_sz, cudaMemcpyHostToDevice);
	cudaMemcpy(OW_d, OW_h, sizeof(float)*OW_sz, cudaMemcpyHostToDevice);
    /*************************************************************************/
    
    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Launch kernel using standard sgemm interface ---------------------------
    printf("Launching kernel..."); fflush(stdout);
    startTime(&timer);
    //Hidden Layer Calculation
    basicSgemm(matArow, matBcol, matBrow, A_d, B_d, C_d);
    //cublas_sgemm(matArow, matBcol, matBrow, A_d, B_d, C_d);
    cuda_ret = cudaDeviceSynchronize();
    basicSigmoid(C_d, len);
    cuda_ret = cudaDeviceSynchronize();

    //Output Layer Calcultion
    basicSgemm(matBcol, matBcol, matArow, OW_d, C_d, OUT_d);
    //cublas_out_sgemm(matBcol, matBcol, matArow, OW_d, C_d, OUT_d);
    cuda_ret = cudaDeviceSynchronize();
    basicSigmoid(OUT_d, OUT_len);
    cuda_ret = cudaDeviceSynchronize();

    if(cuda_ret != cudaSuccess) printf("Unable to launch kernel");
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy device variables from host ----------------------------------------
    printf("Copying data from device to host..."); fflush(stdout);
    startTime(&timer);

    /*************************************************************************/
    //INSERT CODE HERE
	cudaMemcpy(C_h, C_d, sizeof(float)*C_sz, cudaMemcpyDeviceToHost);
	cudaMemcpy(OUT_h, OUT_d, sizeof(float)*OUT_sz, cudaMemcpyDeviceToHost);
    /*************************************************************************/

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Verify correctness -----------------------------------------------------

    printf("Verifying results..."); fflush(stdout);
	std::cout << C_sz;
    verify(A_h, B_h, C_h, OUT_h, matArow, matAcol, matBcol);


    // Free memory ------------------------------------------------------------

    free(A_h);
    free(B_h);
    free(C_h);
    free(OW_h);
    free(OUT_h);

    /*************************************************************************/
    //INSERT CODE HERE
	cudaFree(A_d);
	cudaFree(B_d);
	cudaFree(C_d);    
	cudaFree(OW_d);
	cudaFree(OUT_d);
    /*************************************************************************/

    return 0;
}

