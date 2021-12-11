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

void back_prop_output(float *OW_d, float *C_d, float *OW_new_d, float *update_weight_d, float *OW_t_d,  cudaError_t cuda_ret){
	unsigned matArow, matBrow, matBcol;
	matArow = 1; matBrow = 5; matBcol = 1;
	cuda_ret = cudaDeviceSynchronize();
	basicSgemm(matBrow, matBcol, matArow, C_d, update_weight_d, OW_new_d);
	cuda_ret = cudaDeviceSynchronize();
	cuda_ret = cudaDeviceSynchronize();
	basicSub(matBcol, matBrow, matArow, OW_d, OW_new_d, OW_d);
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
	basicSub(matBrow, matArow, matBcol, A_d, B_new_d, A_d);
	cuda_ret = cudaDeviceSynchronize();
}
 

int main (int argc, char *argv[])
{

    Timer timer;
    cudaError_t cuda_ret;

    // Initialize host variables ----------------------------------------------

    printf("\nSetting up the problem..."); fflush(stdout);
    startTime(&timer);
    
    float *A_h, *B_h, *C_h, *Csig_h, *OW_h, *OUT_h, *OW_new_h, *update_weight_h, *OW_t_h, *B_new_h, *B_temp_h, *inputs_h, *targets_h, *tst_input_h;
    float *A_d, *B_d, *C_d, *Csig_d, *OW_d, *OUT_d, *OW_new_d, *update_weight_d, *OW_t_d, *B_new_d, *B_temp_d;
    size_t A_sz, B_sz, C_sz, Csig_sz, OW_sz, OUT_sz, OW_new_sz, update_sz, OW_t_sz, B_new_sz, B_temp_sz, inputs_sz, target_sz, tst_sz;
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
    inputs_sz = 160;
    target_sz = 40;
	tst_sz = 40;
	
	tst_input_h = (float*) malloc(sizeof(float)*tst_sz);
	float test_inputs_h[] = {0.10895f,0.10965f,0.09549f,0.1043f,0.10474f,0.147f,0.10231f,0.14435f,0.14499f,0.1685f,0.14211f,0.14456f,0.14505f,0.169927f,0.13858f,0.16251f,0.162125f,0.17456f,0.15291f,0.16944f,0.16995f,0.1912f,0.16271f,0.17877f,0.1808f,0.20789f,0.177f,0.20681f,0.20935f,0.21867f,0.19123f,0.20071f,0.19931f,0.2003f,0.18058f,0.1935f,0.1935f,0.24927f,0.1935f,0.2458f};

	for(unsigned int i = 0; i < tst_sz; i++){
                tst_input_h[i] = test_inputs_h[i];
        }


    inputs_h = (float*) malloc(sizeof(float)*inputs_sz);
	float input_h[] = {0.23409f,0.2566f,0.233265f,0.24486f,0.24613f,0.281717f,0.23872f,0.28068f,0.28068f,0.28522f,0.25868f,0.28102f,0.28416f,0.29276f,0.17601f,0.21083f,0.2123f,0.222f,0.13331f,0.16343f,0.1726f,0.17468f,0.12446f,0.1335f,0.1335f,0.16088f,0.127693f,0.14375f,0.1445f,0.16528f,0.14258f,0.15426f,0.15627f,0.185f,0.1448f,0.17956f,0.18326f,0.19347f,0.1733f,0.181f,0.18311f,0.18487f,0.13539f,0.13546f,0.13591f,0.16537f,0.1326f,0.16423f,0.17257f,0.17889f,0.155f,0.16872f,0.16914f,0.17345f,0.147394f,0.16751f,0.16751f,0.1884f,0.163252f,0.17407f,0.175f,0.2089f,0.17013f,0.20102f,0.1996f,0.22141f,0.198593f,0.21674f,0.21646f,0.24181f,0.20037f,0.2353f,0.2353f,0.2595f,0.23127f,0.23643f,0.2357f,0.31632f,0.23545f,0.27007f,0.2769f,0.28489f,0.180681f,0.2636f,0.25565f,0.3042f,0.23839f,0.29228f,0.28435f,0.36727f,0.280843f,0.35502f,0.35333f,0.3857f,0.34632f,0.37991f,0.38083f,0.43169f,0.37652f,0.42459f,0.4293f,0.543f,0.42861f,0.53498f,0.5392f,0.58907f,0.46817f,0.54122f,0.55032f,0.57394f,0.492f,0.50136f,0.50631f,0.58766f,0.4958f,0.53606f,0.53969f,0.54925f,0.510527f,0.5222f,0.5222f,0.559974f,0.50344f,0.51959f,0.522134f,0.6149f,0.51611f,0.54858f,0.555f,0.557f,0.46266f,0.53393f,0.54289f,0.648566f,0.54045f,0.60038f,0.605f,0.651099f,0.53836f,0.64978f,0.6508f,0.8065f,0.63613f,0.8001f,0.805f,0.82021f,0.178655f,0.19499f,0.197f,0.23043f,0.18762f,0.22385f,0.22485f,0.22986f,0.20467f,0.20716f,0.2075f,0.25709f,0.19555f,0.25567f};
	for(unsigned int i = 0; i < inputs_sz; i++){
		inputs_h[i] = input_h[i];
		//printf("\n Inputs: %f", inputs_h[i]);
	}

	targets_h = (float*) malloc(sizeof(float)*target_sz);
	float target_h[] = {0.28068f,0.28102f,0.21083f,0.16343f,0.1335f,0.14375f,0.15426f,0.17956f,0.181f,0.13546f,0.16423f,0.16872f,0.16751f,0.17407f,0.20102f,0.21674f,0.2353f,0.23643f,0.27007f,0.2636f,0.29228f,0.35502f,0.37991f,0.42459f,0.53498f,0.54122f,0.50136f,0.53606f,0.5222f,0.51959f,0.54858f,0.53393f,0.60038f,0.64978f,0.8001f,0.19499f,0.22385f,0.20716f,0.25567f,0.32676f};


    A_h = (float*) malloc( sizeof(float)*A_sz );
    for (unsigned int i=0; i < A_sz; i++) { A_h[i] = (rand()%100)/100.00; }

    B_h = (float*) malloc( sizeof(float)*B_sz );

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
    cudaMemcpy(OW_d, OW_h, sizeof(float)*OW_sz, cudaMemcpyHostToDevice);
    /*************************************************************************/
    
    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
	
   for (int j = 0; j < target_sz; j++){
	for (unsigned int i = 0; i < 4; i++) {
		 B_h[i] = inputs_h[i+j*B_sz]; 
	}
	targets_h[j] = target_h[j];
	cudaMemcpy(B_d, B_h, sizeof(float)*B_sz, cudaMemcpyHostToDevice);
	printf("\n Iter: %i", j);
    for(int i = 0; i < 5000; i++) {
	feed_forward(matArow, matBcol, matBrow, len, OUT_len, OW_d, C_d, Csig_d, OUT_d, A_d, B_d, cuda_ret);

	cudaMemcpy(C_h, C_d, sizeof(float)*C_sz, cudaMemcpyDeviceToHost);
    	cudaMemcpy(Csig_h, Csig_d, sizeof(float)*Csig_sz, cudaMemcpyDeviceToHost);
    	cudaMemcpy(OUT_h, OUT_d, sizeof(float)*OUT_sz, cudaMemcpyDeviceToHost);
  
	update_weight_h[0] = (OUT_h[0] - targets_h[j]) * 0.01f;
	cudaMemcpy(update_weight_d, update_weight_h, sizeof(float)*update_sz, cudaMemcpyHostToDevice);
	
 	back_prop_output(OW_d, C_d, OW_new_d, update_weight_d, OW_t_d, cuda_ret);
	back_prop_hidden(B_d, update_weight_d, A_d, OW_d, OW_t_d, B_new_d, B_temp_d, cuda_ret);
	
	cudaMemcpy(OW_h, OW_d, sizeof(float)*OW_sz, cudaMemcpyDeviceToHost);
	cudaMemcpy(OW_d, OW_h, sizeof(float)*OW_sz, cudaMemcpyHostToDevice);

	cudaMemcpy(A_h, A_d, sizeof(float)*A_sz, cudaMemcpyDeviceToHost);
	cudaMemcpy(A_d, A_h, sizeof(float)*A_sz, cudaMemcpyHostToDevice);
    }
	printf("\n OUTPUT: %f", OUT_h[0]);
  }
	cudaMemcpy(OW_h, OW_d, sizeof(float)*OW_sz, cudaMemcpyDeviceToHost);
        cudaMemcpy(OW_d, OW_h, sizeof(float)*OW_sz, cudaMemcpyHostToDevice);
	cudaMemcpy(A_h, A_d, sizeof(float)*A_sz, cudaMemcpyDeviceToHost);
        cudaMemcpy(A_d, A_h, sizeof(float)*A_sz, cudaMemcpyHostToDevice);

	for(int j = 0; j < 10; j++){
		for (unsigned int i = 0; i < 4; i++) {
        	         B_h[i] = tst_input_h[i+j*B_sz];
			printf("\n Test_inputs: %f", tst_input_h[i+j*B_sz]);
	        }
		cudaMemcpy(B_d, B_h, sizeof(float)*B_sz, cudaMemcpyHostToDevice);
	        printf("\n Test Iter: %i", j);
		feed_forward(matArow, matBcol, matBrow, len, OUT_len, OW_d, C_d, Csig_d, OUT_d, A_d, B_d, cuda_ret);
		cudaMemcpy(OUT_h, OUT_d, sizeof(float)*OUT_sz, cudaMemcpyDeviceToHost);
		printf("\n OUTPUT: %f", OUT_h[0]);
		if(OUT_h[0] > 0.5f){
			printf("\n Price Increase");
		}
		else{
			printf("\n Price Decrease");
		}		
	}
	cudaMemcpy(OW_h, OW_d, sizeof(float)*OW_sz, cudaMemcpyDeviceToHost);
	cudaMemcpy(A_h, A_d, sizeof(float)*A_sz, cudaMemcpyDeviceToHost);

    cudaMemcpy(B_new_h, B_new_d, sizeof(float)*B_new_sz, cudaMemcpyDeviceToHost);

    cudaMemcpy(B_h, B_d, sizeof(float)*B_sz, cudaMemcpyDeviceToHost);

    cudaMemcpy(C_h, C_d, sizeof(float)*C_sz, cudaMemcpyDeviceToHost);

    // Verify correctness -----------------------------------------------------
    printf("Verifying results..."); fflush(stdout);
	std::cout << C_sz;
    verify(A_h, B_h, C_h, OUT_h, B_new_h, matArow, matAcol, matBcol);
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
    free(inputs_h);
    free(targets_h);
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
