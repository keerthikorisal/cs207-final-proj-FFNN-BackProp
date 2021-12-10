#include <stdio.h>
#include <stdlib.h>
#include "kernel.cu"
#include "support.h"
#include <iostream>

void feed_forward(unsigned matArow, unsigned matBcol, unsigned matBrow, int len, int OUT_len, float *OW_d, float *C_d, float *Csig_d, float *OUT_d, float *A_d, float *B_d, cudaError_t cuda_ret){
    	cuda_ret = cudaDeviceSynchronize();
	basicSgemm(matArow, matBcol, matBrow, A_d, B_d, C_d);
    	cuda_ret = cudaDeviceSynchronize();
    	basicSigmoid(C_d, Csig_d, len);
    	cuda_ret = cudaDeviceSynchronize();
    	basicSgemm(matBcol, matBcol, matArow, OW_d, Csig_d, OUT_d);
    	cuda_ret = cudaDeviceSynchronize();
    	basicSigmoid(OUT_d, OUT_d,  OUT_len);
    	cuda_ret = cudaDeviceSynchronize();
}

void back_prop_output(const float *OW_d, float *C_d, float *OW_new_d, float *update_weight_d, float *OW_t_d,  cudaError_t cuda_ret){
	unsigned matArow, matBrow, matBcol;
	matArow = 1; matBrow = 5; matBcol = 1;
	cuda_ret = cudaDeviceSynchronize();
	basicSgemm(matBrow, matBcol, matArow, C_d, update_weight_d, OW_new_d);
	cuda_ret = cudaDeviceSynchronize();
	cuda_ret = cudaDeviceSynchronize();
	basicSub(matBcol, matBrow, matArow, OW_d, OW_new_d, OW_new_d);
	cuda_ret = cudaDeviceSynchronize();
}

void back_prop_hidden(float *B_d, float *update_weight_d, float *A_d, float *OW_d, float *OW_t_d, float *B_new_d, float *B_temp_d, cudaError_t cuda_ret){
	unsigned matArow, matBrow, matBcol;
        matArow = 4; matBrow = 5; matBcol = 1;
        cuda_ret = cudaDeviceSynchronize();
	basicSgemm(matArow, matBcol, matBcol, B_d, update_weight_d, B_temp_d);
	cuda_ret = cudaDeviceSynchronize();
	basicSgemm(matArow, matBrow, matBcol, B_temp_d, OW_d,  B_new_d);
        cuda_ret = cudaDeviceSynchronize();
	basicSub(matBrow, matArow, matBcol, A_d, B_new_d, B_new_d);
	cuda_ret = cudaDeviceSynchronize();
}
 

int main (int argc, char *argv[])
{

    Timer timer;
    cudaError_t cuda_ret;

    // Initialize host variables ----------------------------------------------

    printf("\nSetting up the problem..."); fflush(stdout);
    startTime(&timer);
    
    float *A_h, *B_h, *C_h, *Csig_h, *OW_h, *OUT_h, *OW_new_h, *update_weight_h, *OW_t_h, *B_new_h, *B_temp_h;
    float *A_d, *B_d, *C_d, *Csig_d, *OW_d, *OUT_d, *OW_new_d, *update_weight_d, *OW_t_d, *B_new_d, *B_temp_d;
    size_t A_sz, B_sz, C_sz, Csig_sz, OW_sz, OUT_sz, OW_new_sz, update_sz, OW_t_sz, B_new_sz, B_temp_sz;
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
    
    //Matrix sizes
    A_sz = matArow*matAcol;
    B_sz = matBrow*matBcol;
    C_sz = matArow*matBcol;
    Csig_sz = C_sz;
    OW_sz = matArow*matBcol;
    OUT_sz = matBcol;
    len = C_sz;
    OUT_len = OUT_sz;
    OW_new_sz = OW_sz;
    update_sz = OUT_sz;
    OW_t_sz = A_sz;
    B_new_sz = A_sz;
    B_temp_sz = B_sz;

    A_h = (float*) malloc( sizeof(float)*A_sz );
    for (unsigned int i=0; i < A_sz; i++) { A_h[i] = (rand()%100)/100.00; }

    B_h = (float*) malloc( sizeof(float)*B_sz );
    for (unsigned int i=0; i < B_sz; i++) { B_h[i] = (rand()%100)/100.00; }

    Csig_h = (float*) malloc( sizeof(float)*Csig_sz);
    for (unsigned int i = 0; i < Csig_sz; i++) { Csig_h[i] = (rand()%100)/100.00; }

    OW_h = (float*) malloc(sizeof(float)*OW_sz);
    for(unsigned int i = 0; i < OW_sz; i++) { OW_h[i] = (rand()%100)/100.00; }

    C_h = (float*) malloc( sizeof(float)*C_sz );
    OUT_h = (float*) malloc(sizeof(float)*OUT_sz);
    OW_new_h = (float*) malloc(sizeof(float)*OW_new_sz);
    update_weight_h = (float*) malloc(sizeof(float)*update_sz);	
    OW_t_h = (float*) malloc(sizeof(float)*OW_t_sz);	
    B_new_h = (float*) malloc( sizeof(float)*B_new_sz);
    B_temp_h = (float*) malloc( sizeof(float)*B_temp_sz );

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    printf("    A: %u x %u\n    B: %u x %u\n    C: %u x %u\n", matArow, matAcol,
        matBrow, matBcol, matArow, matBcol);

    // Allocate device variables ----------------------------------------------

    printf("Allocating device variables..."); fflush(stdout);
    startTime(&timer);

    cudaMalloc((void**) &A_d, sizeof(float)*A_sz);
    cudaMalloc((void**) &B_d, sizeof(float)*B_sz);
    cudaMalloc((void**) &C_d, sizeof(float)*C_sz);
    cudaMalloc((void**) &OW_d, sizeof(float)*OW_sz);
    cudaMalloc((void**) &OUT_d, sizeof(float)*OUT_sz);
    cudaMalloc((void**) &Csig_d, sizeof(float)*Csig_sz);
    cudaMalloc((void**) &OW_new_d, sizeof(float)*OW_new_sz);
    cudaMalloc((void**) &update_weight_d, sizeof(float)*update_sz);
    cudaMalloc((void**) &OW_t_d, sizeof(float)*OW_t_sz);
    cudaMalloc((void**) &B_new_d, sizeof(float)*B_new_sz);
    cudaMalloc((void**) &B_temp_d, sizeof(float)*B_temp_sz);
    /*************************************************************************/	

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    // Copy host variables to device ------------------------------------------

    printf("Copying data from host to device..."); fflush(stdout);
    startTime(&timer);
	
    /*************************************************************************/
    //Host to Device
    cudaMemcpy(A_d, A_h, sizeof(float)*A_sz, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, sizeof(float)*B_sz, cudaMemcpyHostToDevice);
    cudaMemcpy(OW_d, OW_h, sizeof(float)*OW_sz, cudaMemcpyHostToDevice);
    /*************************************************************************/
    
    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Feed_forward function --------------------------------------------------
    feed_forward(matArow, matBcol, matBrow, len, OUT_len, OW_d, C_d, Csig_d, OUT_d, A_d, B_d, cuda_ret);


    // Copy device variables from host ----------------------------------------
    //printf("Copying data from device to host..."); fflush(stdout);
    //startTime(&timer);

    /*************************************************************************/
    //Device To Host
    cudaMemcpy(C_h, C_d, sizeof(float)*C_sz, cudaMemcpyDeviceToHost);
    cudaMemcpy(Csig_h, Csig_d, sizeof(float)*Csig_sz, cudaMemcpyDeviceToHost);
    cudaMemcpy(OUT_h, OUT_d, sizeof(float)*OUT_sz, cudaMemcpyDeviceToHost);
    /*************************************************************************/

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
	
    //Back_prop_output---------------------------------------------------------
    update_weight_h[0] = (OUT_h[0] - 1.0f) * 0.05f;
    cudaMemcpy(update_weight_d, update_weight_h, sizeof(float)*update_sz, cudaMemcpyHostToDevice);
    
    printf("\nupdate: %f", update_weight_h[0]);
       
    back_prop_output(OW_d, C_d, OW_new_d, update_weight_d, OW_t_d, cuda_ret);
    
    printf("\nOW_orig: %f", OW_h[1]);
	
    back_prop_hidden(B_d, update_weight_d, A_d, OW_d, OW_t_d, B_new_d, B_temp_d, cuda_ret);
 /*   
    for(int i = 0; i < 100; i++) {
	feed_forward(matArow, matBcol, matBrow, len, OUT_len, OW_new_d, C_d, Csig_d, OUT_d, B_new_d, B_d, cuda_ret);
    	update_weight_h[0] = (OUT_h[0] - 1.0f) * 0.05f;
 	back_prop_output(OW_new_d, C_d, //, update_weight_d, OW_t_d, cuda_ret);
	back_prop_hidden(B_d, update_weight_d, A_d, OW_new_d, OW_t_d, B_new_d, B_temp_d, cuda_ret);
    }
*/
    cudaMemcpy(OW_new_h, OW_new_d, sizeof(float)*OW_new_sz, cudaMemcpyDeviceToHost);
    cudaMemcpy(OW_h, OW_d, sizeof(float)*OW_sz, cudaMemcpyDeviceToHost);
    cudaMemcpy(OW_t_h, OW_t_d, sizeof(float)*OW_t_sz, cudaMemcpyDeviceToHost);
    cudaMemcpy(B_new_h, B_new_d, sizeof(float)*B_new_sz, cudaMemcpyDeviceToHost);
    cudaMemcpy(update_weight_h, update_weight_d, sizeof(float)*update_sz, cudaMemcpyDeviceToHost);	
    cudaMemcpy(B_h, B_d, sizeof(float)*B_sz, cudaMemcpyDeviceToHost);
    cudaMemcpy(B_temp_h, B_temp_d, sizeof(float)*B_temp_sz, cudaMemcpyDeviceToHost);
    cudaMemcpy(C_h, C_d, sizeof(float)*C_sz, cudaMemcpyDeviceToHost);
	
    printf("\nC: %f/%f/%f/%f/%f", C_h[0],C_h[1],C_h[2],C_h[3],C_h[4]);
    printf("\nOW_new: %f/%f/%f/%f/%f", OW_new_h[0],OW_new_h[1],OW_new_h[2],OW_new_h[3],OW_new_h[4]);
    printf("\nOW: %f/%f/%f/%f/%f ", OW_h[0],OW_h[1],OW_h[2],OW_h[3],OW_h[4]);
    for(int i = 0; i < 20; i++) {
	printf("\nA: %f/ B_new: %f ", A_h[i], B_new_h[i]);
    }	
    printf("\nupdate_weight_h: %f/  \n", update_weight_h[0]);
    printf("B: %f/%f/%f/%f/%f \n", B_h[0],B_h[1],B_h[2],B_h[3]);
    printf("B_temp: %f/%f/%f/%f/%f \n", B_temp_h[0],B_temp_h[1],B_temp_h[2],B_temp_h[3]);
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
    free(Csig_h);
    free(OW_new_h);
    free(update_weight_h);
    free(B_new_h);
    free(B_temp_h);
    free(OW_t_h);
    /*************************************************************************/
    //CUDA Free
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);    
    cudaFree(OW_d);
    cudaFree(OUT_d);
    cudaFree(Csig_d);
    cudaFree(OW_new_d);
    cudaFree(update_weight_d);
    cudaFree(B_new_d);
    cudaFree(B_temp_d);
    cudaFree(OW_t_d);
    /*************************************************************************/
    return 0;
}
