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
	//float test_inputs_h[] = {0.57853f,0.5962f,0.573655f,0.59057f,0.59921f,0.61089f,0.597f,0.61003f,0.602f,0.611646f,0.591f,0.59845f,0.60257f,0.6149f,0.598f,0.61321f,0.60684f,0.608941f,0.5912f,0.59624f,0.58916f,0.59493f,0.583f,0.59316f,0.60098f,0.60708f,0.593868f,0.59706f,0.59099f,0.60533f,0.573f,0.57423f,0.56301f,0.56955f,0.53558f,0.56568f,0.56491f,0.58027f,0.55024f,0.57996f,0.561817f,0.57499f,0.52865f,0.5323f,0.5501f,0.55418f,0.53444f,0.54858f,0.555f,0.557f,0.54213f,0.55367f,0.556f,0.55682f,0.53584f,0.53625f,0.53705f,0.53806f,0.51195f,0.51219f,0.51203f,0.519f,0.48335f,0.49481f,0.502f,0.502f,0.46717f,0.49846f,0.498f,0.49885f,0.46266f,0.46373f,0.48521f,0.50467f,0.48237f,0.50081f,0.51362f,0.5192f,0.4983f,0.49873f};
	float test_inputs_h[] = {0.10895f,0.10965f,0.09549f,0.1043f,0.10474f,0.147f,0.10231f,0.14435f,0.14499f,0.1685f,0.14211f,0.14456f,0.14505f,0.169927f,0.13858f,0.16251f,0.162125f,0.17456f,0.15291f,0.16944f,0.16995f,0.1912f,0.16271f,0.17877f,0.1808f,0.20789f,0.177f,0.20681f,0.20935f,0.21867f,0.19123f,0.20071f,0.19931f,0.2003f,0.18058f,0.1935f,0.1935f,0.24927f,0.1935f,0.2458f};

	for(unsigned int i = 0; i < tst_sz; i++){
                tst_input_h[i] = test_inputs_h[i];
        }


    inputs_h = (float*) malloc(sizeof(float)*inputs_sz);
	//float input_h[2];
	//float input_h[] = {0.523515f, 0.53555f, 0.5231f, 0.53235f, 0.53759f, 0.53856f, 0.52747f, 0.53442f};
	/*float input_h[] = {0.51869f,0.5208f,0.51292f,0.52053f,0.523515f,0.53555f,0.5231f,0.53235f,0.53759f,0.53856f,0.52747f,0.53442f,0.5369f,0.53749f,0.5285f,0.5297f,0.53467f,0.53518f,0.52714f,0.53365f,0.53516f,0.5362f,0.52282f,0.53088f,0.52299f,0.5348f,0.5187f,0.53329f,0.53277f,0.53349f,0.522155f,0.53113f,0.53024f,0.531f,0.5202f,0.52037f,0.52149f,0.5253f,0.51755f,0.51975f,0.5225f,0.5225f,0.510527f,0.516f,0.517f,0.5229f,0.51369f,0.51773f,0.519612f,0.52855f,0.51587f,0.52583f,0.52546f,0.526045f,0.5166f,0.5222f,0.52417f,0.5461f,0.5185f,0.52454f,0.52399f,0.53774f,0.52348f,0.53619f,0.5289f,0.5298f,0.50344f,0.50458f,0.5187f,0.53511f,0.51546f,0.53376f,0.5345f,0.53684f,0.52168f,0.53107f,0.53675f,0.55844f,0.5345f,0.54486f,0.5501f,0.55018f,0.5338f,0.53939f,0.539599f,0.545f,0.5354f,0.54127f,0.543f,0.54399f,0.52722f,0.52801f,0.52912f,0.53301f,0.51394f,0.51438f,0.52065f,0.523759f,0.51376f,0.52101f,0.52447f,0.53722f,0.52227f,0.53463f,0.536744f,0.559974f,0.53012f,0.5547f,0.549604f,0.553669f,0.54509f,0.5485f,0.55127f,0.55314f,0.529459f,0.54613f,0.543218f,0.54579f,0.5355f,0.53741f,0.5292f,0.52931f,0.51175f,0.51671f,0.51985f,0.53359f,0.51842f,0.52204f,0.523f,0.53339f,0.51646f,0.51959f,0.522133f,0.53083f,0.51611f,0.52948f,0.53595f,0.54288f,0.53142f,0.54227f,0.54544f,0.55727f,0.54066f,0.54122f,0.54124f,0.54694f,0.533515f,0.54657f,0.549f,0.54949f,0.54183f,0.54364f,0.549089f,0.57895f,0.549089f,0.57755f,0.573625f,0.58333f,0.5689f,0.57053f,0.57853f,0.5962f,0.573655f,0.59057f,0.59921f,0.61089f,0.597f,0.61003f,0.602f,0.611646f,0.591f,0.59845f,0.60257f,0.6149f,0.598f,0.61321f,0.60684f,0.608941f,0.5912f,0.59624f,0.58916f,0.59493f,0.583f,0.59316f,0.60098f,0.60708f,0.593868f,0.59706f,0.59099f,0.60533f,0.573f,0.57423f,0.56301f,0.56955f,0.53558f,0.56568f,0.56491f,0.58027f,0.55024f,0.57996f,0.561817f,0.57499f,0.52865f,0.5323f,0.5501f,0.55418f,0.53444f,0.54858f,0.555f,0.557f,0.54213f,0.55367f,0.556f,0.55682f,0.53584f,0.53625f,0.53705f,0.53806f,0.51195f,0.51219f,0.51203f,0.519f,0.48335f,0.49481f,0.502f,0.502f,0.46717f,0.49846f,0.498f,0.49885f,0.46266f,0.46373f,0.48521f,0.50467f,0.48237f,0.50081f,0.51362f,0.5192f,0.4983f,0.49873f,0.518f,0.52185f,0.50938f,0.51974f,0.50593f,0.51555f,0.5036f,0.51424f,0.51459f,0.52827f,0.510685f,0.52765f,0.53426f,0.5405f,0.52467f,0.53165f,0.52159f,0.53813f,0.51958f,0.53365f,0.52546f,0.52736f,0.508682f,0.5089f,0.51f,0.51686f,0.5045f,0.51383f,0.51651f,0.53578f,0.51627f,0.52745f,0.53017f,0.53378f,0.52085f,0.52283f,0.52605f,0.526365f,0.50547f,0.50572f,0.49998f,0.50859f,0.49088f,0.50141f,0.50218f,0.51425f,0.49494f,0.51357f,0.51281f,0.5225f,0.508f,0.51793f,0.513678f,0.51901f,0.5082f,0.51487f,0.52062f,0.53882f,0.5193f,0.53393f,0.54289f,0.5548f,0.54045f,0.55247f,0.5547f,0.560558f,0.54932f,0.5595f,0.559987f,0.562165f,0.550945f,0.55446f,0.55523f,0.56974f,0.54846f,0.56574f,0.57011f,0.57886f,0.569958f,0.57268f,0.56856f,0.57632f,0.567f,0.576f,0.57159f,0.6141f,0.56557f,0.60836f,0.60926f,0.628f,0.60503f,0.62718f,0.625f,0.628818f,0.6091f,0.61108f,0.6265f,0.648566f,0.62526f,0.64549f,0.642123f,0.64663f,0.63461f,0.6365f,0.621462f,0.6323f,0.60932f,0.61447f,0.61293f,0.61988f,0.59852f,0.60685f,0.60475f,0.6145f,0.60409f,0.61442f,0.615f,0.617543f,0.59134f,0.59401f,0.597375f,0.613395f,0.59601f,0.61061f,0.607402f,0.61916f,0.605f,0.61912f,0.62314f,0.62682f,0.61418f,0.61527f,0.61458f,0.62129f,0.608621f,0.61107f,0.616f,0.61746f,0.60286f,0.61299f,0.60698f,0.6146f,0.59988f,0.60038f,0.605f,0.60987f,0.5915f,0.59347f,0.58549f,0.5855f,0.56041f,0.57405f,0.58836f,0.59254f,0.5755f,0.57834f,0.57981f,0.58285f,0.568724f,0.58092f,0.59233f,0.59883f,0.586868f,0.59249f,0.59149f,0.59224f,0.57f,0.57063f,0.553f,0.57419f,0.55f,0.57225f,0.56038f,0.57041f,0.54837f,0.55034f,0.56135f,0.56323f,0.53836f,0.54661f,0.5556f,0.573124f,0.54994f,0.56972f,0.56628f,0.56677f,0.55414f,0.56662f,0.57068f,0.57617f,0.56016f,0.56063f,0.54266f,0.56318f,0.54173f,0.56263f,0.5721f,0.587351f,0.57082f,0.5845f,0.60655f,0.60888f,0.594776f,0.59967f,0.6085f,0.6298f,0.60704f,0.62448f,0.63064f,0.63274f,0.6193f,0.62591f,0.62937f,0.63175f,0.62345f,0.628f,0.62799f,0.63f,0.6184f,0.61952f,0.62004f,0.651099f,0.62004f,0.64978f,0.6508f,0.655525f,0.63613f,0.65058f,0.65009f,0.676755f,0.64942f,0.67113f,0.66803f,0.69036f,0.66332f,0.67879f,0.6845f,0.70644f,0.68382f,0.70313f,0.702846f,0.7125f,0.68771f,0.70476f,0.70109f,0.704864f,0.69004f,0.69828f,0.70063f,0.703f,0.69023f,0.69433f,0.694f,0.69968f,0.68704f,0.697f,0.69918f,0.717569f,0.69775f,0.71301f,0.71624f,0.72158f,0.70651f,0.72075f,0.7166f,0.72065f,0.709118f,0.71154f,0.71163f,0.718187f,0.703377f,0.71241f,0.710983f,0.7534f,0.71025f,0.74629f,0.75136f,0.775f,0.74336f,0.74555f,0.73741f,0.74146f,0.712906f,0.73709f,0.73886f,0.75855f,0.735435f,0.75547f,0.76031f,0.76611f,0.75631f,0.76229f,0.76911f,0.7768f,0.76353f};*/
	//float input_h[] = {0.51869f,0.5208f,0.51292f,0.52053f,0.523515f,0.53555f,0.5231f,0.53235f,0.53759f,0.53856f,0.52747f,0.53442f,0.5369f,0.53749f,0.5285f,0.5297f,0.53467f,0.53518f,0.52714f,0.53365f,0.53516f,0.5362f,0.52282f,0.53088f,0.52299f,0.5348f,0.5187f,0.53329f,0.53277f,0.53349f,0.522155f,0.53113f,0.53024f,0.531f,0.5202f,0.52037f,0.52149f,0.5253f,0.51755f,0.51975f,0.5225f,0.5225f,0.510527f,0.516f,0.517f,0.5229f,0.51369f,0.51773f,0.519612f,0.52855f,0.51587f,0.52583f,0.52546f,0.526045f,0.5166f,0.5222f,0.52417f,0.5461f,0.5185f,0.52454f,0.52399f,0.53774f,0.52348f,0.53619f,0.5289f,0.5298f,0.50344f,0.50458f,0.5187f,0.53511f,0.51546f,0.53376f,0.5345f,0.53684f,0.52168f,0.53107f,0.53675f,0.55844f,0.5345f,0.54486f,0.5501f,0.55018f,0.5338f,0.53939f,0.539599f,0.545f,0.5354f,0.54127f,0.543f,0.54399f,0.52722f,0.52801f,0.52912f,0.53301f,0.51394f,0.51438f,0.52065f,0.523759f,0.51376f,0.52101f,0.52447f,0.53722f,0.52227f,0.53463f,0.536744f,0.559974f,0.53012f,0.5547f,0.549604f,0.553669f,0.54509f,0.5485f,0.55127f,0.55314f,0.529459f,0.54613f,0.543218f,0.54579f,0.5355f,0.53741f,0.5292f,0.52931f,0.51175f,0.51671f,0.51985f,0.53359f,0.51842f,0.52204f,0.523f,0.53339f,0.51646f,0.51959f,0.522133f,0.53083f,0.51611f,0.52948f,0.53595f,0.54288f,0.53142f,0.54227f,0.54544f,0.55727f,0.54066f,0.54122f,0.54124f,0.54694f,0.533515f,0.54657f,0.549f,0.54949f,0.54183f,0.54364f,0.549089f,0.57895f,0.549089f,0.57755f,0.573625f,0.58333f,0.5689f,0.57053f,0.57853f,0.5962f,0.573655f,0.59057f,0.59921f,0.61089f,0.597f,0.61003f,0.602f,0.611646f,0.591f,0.59845f,0.60257f,0.6149f,0.598f,0.61321f,0.60684f,0.608941f,0.5912f,0.59624f,0.58916f,0.59493f,0.583f,0.59316f,0.60098f,0.60708f,0.593868f,0.59706f,0.59099f,0.60533f,0.573f,0.57423f,0.56301f,0.56955f,0.53558f,0.56568f,0.56491f,0.58027f,0.55024f,0.57996f};
//	float input_h[] = {0.51869f,0.5208f,0.51292f,0.52053f,0.523515f,0.53555f,0.5231f,0.53235f,0.53759f,0.53856f,0.52747f,0.53442f,0.5369f,0.53749f,0.5285f,0.5297f,0.53467f,0.53518f,0.52714f,0.53365f,0.53516f,0.5362f,0.52282f,0.53088f,0.52299f,0.5348f,0.5187f,0.53329f,0.53277f,0.53349f,0.522155f,0.53113f,0.53024f,0.531f,0.5202f,0.52037f,0.52149f,0.5253f,0.51755f,0.51975f,0.5225f,0.5225f,0.510527f,0.516f,0.517f,0.5229f,0.51369f,0.51773f,0.519612f,0.52855f,0.51587f,0.52583f,0.52546f,0.526045f,0.5166f,0.5222f,0.52417f,0.5461f,0.5185f,0.52454f,0.52399f,0.53774f,0.52348f,0.53619f,0.5289f,0.5298f,0.50344f,0.50458f,0.5187f,0.53511f,0.51546f,0.53376f,0.5345f,0.53684f,0.52168f,0.53107f,0.53675f,0.55844f,0.5345f,0.54486f,0.5501f,0.55018f,0.5338f,0.53939f,0.539599f,0.545f,0.5354f,0.54127f,0.543f,0.54399f,0.52722f,0.52801f,0.52912f,0.53301f,0.51394f,0.51438f,0.52065f,0.523759f,0.51376f,0.52101f,0.52447f,0.53722f,0.52227f,0.53463f,0.536744f,0.559974f,0.53012f,0.5547f,0.549604f,0.553669f,0.54509f,0.5485f,0.55127f,0.55314f,0.529459f,0.54613f,0.543218f,0.54579f,0.5355f,0.53741f,0.5292f,0.52931f,0.51175f,0.51671f,0.51985f,0.53359f,0.51842f,0.52204f,0.523f,0.53339f,0.51646f,0.51959f,0.522133f,0.53083f,0.51611f,0.52948f,0.53595f,0.54288f,0.53142f,0.54227f,0.54544f,0.55727f,0.54066f,0.54122f,0.54124f,0.54694f,0.533515f,0.54657f,0.549f,0.54949f,0.54183f,0.54364f,0.549089f,0.57895f,0.549089f,0.57755f,0.573625f,0.58333f,0.5689f,0.57053f};
	float input_h[] = {0.23409f,0.2566f,0.233265f,0.24486f,0.24613f,0.281717f,0.23872f,0.28068f,0.28068f,0.28522f,0.25868f,0.28102f,0.28416f,0.29276f,0.17601f,0.21083f,0.2123f,0.222f,0.13331f,0.16343f,0.1726f,0.17468f,0.12446f,0.1335f,0.1335f,0.16088f,0.127693f,0.14375f,0.1445f,0.16528f,0.14258f,0.15426f,0.15627f,0.185f,0.1448f,0.17956f,0.18326f,0.19347f,0.1733f,0.181f,0.18311f,0.18487f,0.13539f,0.13546f,0.13591f,0.16537f,0.1326f,0.16423f,0.17257f,0.17889f,0.155f,0.16872f,0.16914f,0.17345f,0.147394f,0.16751f,0.16751f,0.1884f,0.163252f,0.17407f,0.175f,0.2089f,0.17013f,0.20102f,0.1996f,0.22141f,0.198593f,0.21674f,0.21646f,0.24181f,0.20037f,0.2353f,0.2353f,0.2595f,0.23127f,0.23643f,0.2357f,0.31632f,0.23545f,0.27007f,0.2769f,0.28489f,0.180681f,0.2636f,0.25565f,0.3042f,0.23839f,0.29228f,0.28435f,0.36727f,0.280843f,0.35502f,0.35333f,0.3857f,0.34632f,0.37991f,0.38083f,0.43169f,0.37652f,0.42459f,0.4293f,0.543f,0.42861f,0.53498f,0.5392f,0.58907f,0.46817f,0.54122f,0.55032f,0.57394f,0.492f,0.50136f,0.50631f,0.58766f,0.4958f,0.53606f,0.53969f,0.54925f,0.510527f,0.5222f,0.5222f,0.559974f,0.50344f,0.51959f,0.522134f,0.6149f,0.51611f,0.54858f,0.555f,0.557f,0.46266f,0.53393f,0.54289f,0.648566f,0.54045f,0.60038f,0.605f,0.651099f,0.53836f,0.64978f,0.6508f,0.8065f,0.63613f,0.8001f,0.805f,0.82021f,0.178655f,0.19499f,0.197f,0.23043f,0.18762f,0.22385f,0.22485f,0.22986f,0.20467f,0.20716f,0.2075f,0.25709f,0.19555f,0.25567f};
	for(unsigned int i = 0; i < inputs_sz; i++){
		inputs_h[i] = input_h[i];
		//printf("\n Inputs: %f", inputs_h[i]);
	}

	targets_h = (float*) malloc(sizeof(float)*target_sz);
	//float target_h[] = {1.0f,1.0f,0.0f,1.0f,0.0f,1.0f,0.0f,0.0f,0.0f,0.0f,1.0f,1.0f,0.0f,1.0f,1.0f,0.0f,1.0f,0.0f,1.0f,0.0f,1.0f,0.0f,0.0f,1.0f,1.0f,1.0f,0.0f,0.0f,0.0f,0.0f,1.0f,0.0f,1.0f,1.0f,0.0f,1.0f,0.0f,1.0f,0.0f,1.0f,1.0f,0.0f,1.0f,0.0f,0.0f,1.0f,0.0f,0.0f,1.0f,0.0f,1.0f,1.0f,0.0f,0.0f,0.0f,1.0f,0.0f,1.0f,0.0f,1.0f,0.0f,1.0f,1.0f,1.0f,0.0f,1.0f,1.0f,0.0f,0.0f,0.0f,1.0f,1.0f,0.0f,1.0f,1.0f,1.0f,0.0f,1.0f,1.0f,1.0f,1.0f,1.0f,0.0f,1.0f,0.0f,0.0f,0.0f,1.0f,0.0f,1.0f,1.0f,0.0f,0.0f,1.0f,0.0f,0.0f,0.0f,1.0f,1.0f,1.0f,0.0f,1.0f,0.0f,0.0f,1.0f,0.0f,0.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,0.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,0.0f,0.0f,1.0f,1.0f,1.0f,0.0f,1.0f,1.0f,0.0f,0.0f,1.0f,1.0f,1.0f,0.0f,1.0f,1.0f,0.0f,1.0f,1.0f,0.0f,0.0f,0.0f,1.0f,1.0f,0.0f,0.0f,0.0f,0.0f,1.0f,0.0f,1.0f,1.0f,0.0f,0.0f,0.0f,1.0f,1.0f,0.0f,1.0f,1.0f,1.0f,1.0f,0.0f,0.0f,0.0f,0.0f,1.0f,1.0f,0.0f,0.0f,0.0f,1.0f,1.0f,1.0f,0.0f,1.0f,0.0f,1.0f,1.0f,0.0f,1.0f,0.0f,1.0f,0.0f,0.0f,0.0f,1.0f,0.0f,1.0f,1.0f,0.0f,0.0f,0.0f,1.0f,1.0f,1.0f,0.0f,0.0f,0.0f,0.0f,1.0f,1.0f,0.0f,1.0f,1.0f,1.0f,0.0f,0.0f,0.0f,1.0f,1.0f,1.0f,1.0f,1.0f,0.0f,1.0f,1.0f,1.0f,1.0f,0.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,0.0f,1.0f,0.0f,0.0f,1.0f,0.0f,0.0f,1.0f,0.0f,1.0f,1.0f,0.0f,0.0f,1.0f,0.0f,1.0f,0.0f,0.0f,1.0f,0.0f,0.0f,1.0f,0.0f};
	//float target_h[] = {1.0f,1.0f,0.0f,1.0f,0.0f,1.0f,0.0f,0.0f,0.0f,0.0f,1.0f,1.0f,0.0f,1.0f,1.0f,0.0f,1.0f,0.0f,1.0f,0.0f,1.0f,0.0f,0.0f,1.0f,1.0f,1.0f,0.0f,0.0f,0.0f,0.0f,1.0f,0.0f,1.0f,1.0f,0.0f,1.0f,0.0f,1.0f,0.0f,1.0f};
	//float target_h[] = {0.022708f,0.003888f,-0.00883f,0.007457f,-0.00519f,0.00454f,-0.00405f,-0.02026f,-0.00119f,-0.00722f,0.003353f,0.015645f,-0.0069f,0.004481f,0.02221f,-0.05895f,0.05783f,-0.00504f,0.025966f,-0.01004f,0.003485f,-0.0245f,-0.02581f,0.012889f,0.026142f,0.03754f,-0.01118f,-0.00432f,-0.01597f,-0.03852f,0.010315f,-0.00469f,0.019034f,0.024156f,-0.00194f,0.009885f,-0.00536f,0.062376f,-0.01215f,0.035125f};
//	float target_h[] = {0.536507f,0.500303f,0.43788f,0.444028f,0.454343f,0.519185f,0.51827f,0.540911f,0.502005f,0.43743f,0.552898f,0.506835f,0.498208f,0.509789f,0.538629f,0.51954f,0.521395f,0.5012f,0.535511f,0.49401f,0.527173f,0.553459f,0.51752f,0.529368f,0.564634f,0.502916f,0.481596f,0.517296f,0.493535f,0.49875f,0.513945f,0.493323f,0.531074f,0.520559f,0.557578f,0.319452f,0.536935f,0.481369f,0.558276f,0.569069f};
	float target_h[] = {0.28068f,0.28102f,0.21083f,0.16343f,0.1335f,0.14375f,0.15426f,0.17956f,0.181f,0.13546f,0.16423f,0.16872f,0.16751f,0.17407f,0.20102f,0.21674f,0.2353f,0.23643f,0.27007f,0.2636f,0.29228f,0.35502f,0.37991f,0.42459f,0.53498f,0.54122f,0.50136f,0.53606f,0.5222f,0.51959f,0.54858f,0.53393f,0.60038f,0.64978f,0.8001f,0.19499f,0.22385f,0.20716f,0.25567f,0.32676f};
	//for(unsigned int i = 0; i < target_sz; i++){
          //      target_h[i] = targets_h[i];
                //printf("\n Inputs: %f", inputs_h[i]);
       // }

    A_h = (float*) malloc( sizeof(float)*A_sz );
    for (unsigned int i=0; i < A_sz; i++) { A_h[i] = (rand()%100)/100.00; }

    B_h = (float*) malloc( sizeof(float)*B_sz );
    //for (unsigned int i=0; i < B_sz; i++) { B_h[i] = (rand()%100)/100.00; }

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
    //cudaMemcpy(B_d, B_h, sizeof(float)*B_sz, cudaMemcpyHostToDevice);
    cudaMemcpy(OW_d, OW_h, sizeof(float)*OW_sz, cudaMemcpyHostToDevice);
    /*************************************************************************/
    
    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Feed_forward function --------------------------------------------------
    //feed_forward(matArow, matBcol, matBrow, len, OUT_len, OW_d, C_d, Csig_d, OUT_d, A_d, B_d, cuda_ret);


    // Copy device variables from host ----------------------------------------
    //printf("Copying data from device to host..."); fflush(stdout);
    //startTime(&timer);

    /*************************************************************************/
    //Device To Host
    //cudaMemcpy(C_h, C_d, sizeof(float)*C_sz, cudaMemcpyDeviceToHost);
    //cudaMemcpy(Csig_h, Csig_d, sizeof(float)*Csig_sz, cudaMemcpyDeviceToHost);
    //cudaMemcpy(OUT_h, OUT_d, sizeof(float)*OUT_sz, cudaMemcpyDeviceToHost);
    /*************************************************************************/

    //cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
	
    //Back_prop_output---------------------------------------------------------
    //update_weight_h[0] = (OUT_h[0] - 1.0f) * 0.05f;
    //cudaMemcpy(update_weight_d, update_weight_h, sizeof(float)*update_sz, cudaMemcpyHostToDevice);
    
    //printf("\nupdate: %f", update_weight_h[0]);
       
    //back_prop_output(OW_d, C_d, OW_new_d, update_weight_d, OW_t_d, cuda_ret);
    
    //printf("\nOW_orig: %f", OW_h[1]);
	
    //back_prop_hidden(B_d, update_weight_d, A_d, OW_d, OW_t_d, B_new_d, B_temp_d, cuda_ret);
   for (int j = 0; j < target_sz; j++){
	for (unsigned int i = 0; i < 4; i++) {
		 B_h[i] = inputs_h[i+j*B_sz]; 
	}
	targets_h[j] = target_h[j];
	cudaMemcpy(B_d, B_h, sizeof(float)*B_sz, cudaMemcpyHostToDevice);
	printf("\n Iter: %i", j);
    for(int i = 0; i < 5000; i++) {
	feed_forward(matArow, matBcol, matBrow, len, OUT_len, OW_d, C_d, Csig_d, OUT_d, A_d, B_d, cuda_ret);
	/*for(int i = 0; i < A_sz; i++) {
                printf("\nA: %f ", A_h[i]);
            }	
	for(int i = 0; i < OW_sz; i++) {
                printf("\nOW: %f ", OW_h[i]);
            }
	for(int i = 0; i < B_sz; i++) {
                printf("\nB: %f ", B_h[i]);
            }*/
	cudaMemcpy(C_h, C_d, sizeof(float)*C_sz, cudaMemcpyDeviceToHost);
    	cudaMemcpy(Csig_h, Csig_d, sizeof(float)*Csig_sz, cudaMemcpyDeviceToHost);
    	cudaMemcpy(OUT_h, OUT_d, sizeof(float)*OUT_sz, cudaMemcpyDeviceToHost);
	/*for(int i = 0; i < C_sz; i++) {
                printf("\nC: %f ", C_h[i]);
            }
        for(int i = 0; i < C_sz; i++) {
                printf("\nC_sig: %f ", Csig_h[i]);
            }
	for(int i = 0; i < OUT_sz; i++){
                printf("\nOUT: %f ", OUT_h[i]);
        }*/    
	update_weight_h[0] = (OUT_h[0] - targets_h[j]) * 0.01f;
	//printf("\n Update_weight: %f", update_weight_h[0]);
	cudaMemcpy(update_weight_d, update_weight_h, sizeof(float)*update_sz, cudaMemcpyHostToDevice);
	
 	back_prop_output(OW_d, C_d, OW_new_d, update_weight_d, OW_t_d, cuda_ret);
	back_prop_hidden(B_d, update_weight_d, A_d, OW_d, OW_t_d, B_new_d, B_temp_d, cuda_ret);
	
	//cudaMemcpy(OW_new_h, OW_new_d, sizeof(float)*OW_new_sz, cudaMemcpyDeviceToHost);
	//cudaMemcpy(B_new_h, B_new_d, sizeof(float)*B_new_sz, cudaMemcpyDeviceToHost);
	//free(OW_h);
	//free(A_h);	
	//OW_d = OW_new_d;
	cudaMemcpy(OW_h, OW_d, sizeof(float)*OW_sz, cudaMemcpyDeviceToHost);
	cudaMemcpy(OW_d, OW_h, sizeof(float)*OW_sz, cudaMemcpyHostToDevice);
	//A_d = B_new_d;
	cudaMemcpy(A_h, A_d, sizeof(float)*A_sz, cudaMemcpyDeviceToHost);
	cudaMemcpy(A_d, A_h, sizeof(float)*A_sz, cudaMemcpyHostToDevice);
	//for(int i = 0; i < A_sz; i++) {
        //	printf("\nA: %f/ A_new: %f ", A_h[i], B_new_h[i]);
	//    }
	//if (i == 4){break;}	
    }
	printf("\n OUTPUT: %f", OUT_h[0]);
	/*for(int i = 0; i < A_sz; i++) {
             printf("\nA: %f ", A_h[i]);
         }
	for(int i = 0; i < OW_sz; i++) {
             printf("\nOW: %f ", OW_h[i]);
         }*/
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
		/*for(int i = 0; i < A_sz; i++) {
                	printf("\nA: %f ", A_h[i]);
            	}
	        for(int i = 0; i < OW_sz; i++) {
        	        printf("\nOW: %f ", OW_h[i]);
	            }
        	for(int i = 0; i < B_sz; i++) {
                	printf("\nB: %f ", B_h[i]);
	            }*/
        	//cudaMemcpy(B_d, B_h, sizeof(float)*B_sz, cudaMemcpyHostToDevice);
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
    //cudaMemcpy(OW_new_h, OW_new_d, sizeof(float)*OW_new_sz, cudaMemcpyDeviceToHost);
    //cudaMemcpy(OW_h, OW_d, sizeof(float)*OW_sz, cudaMemcpyDeviceToHost);
    //cudaMemcpy(OW_t_h, OW_t_d, sizeof(float)*OW_t_sz, cudaMemcpyDeviceToHost);
    cudaMemcpy(B_new_h, B_new_d, sizeof(float)*B_new_sz, cudaMemcpyDeviceToHost);
   // cudaMemcpy(update_weight_h, update_weight_d, sizeof(float)*update_sz, cudaMemcpyDeviceToHost);	
    cudaMemcpy(B_h, B_d, sizeof(float)*B_sz, cudaMemcpyDeviceToHost);
    //cudaMemcpy(B_temp_h, B_temp_d, sizeof(float)*B_temp_sz, cudaMemcpyDeviceToHost);
    cudaMemcpy(C_h, C_d, sizeof(float)*C_sz, cudaMemcpyDeviceToHost);
/*	
    printf("\nC: %f/%f/%f/%f/%f", C_h[0],C_h[1],C_h[2],C_h[3],C_h[4]);
    printf("\nOW_new: %f/%f/%f/%f/%f", OW_new_h[0],OW_new_h[1],OW_new_h[2],OW_new_h[3],OW_new_h[4]);
    printf("\nOW: %f/%f/%f/%f/%f ", OW_h[0],OW_h[1],OW_h[2],OW_h[3],OW_h[4]);
    for(int i = 0; i < 20; i++) {
	printf("\nA: %f/ B_new: %f ", A_h[i], B_new_h[i]);
    }	
    printf("\nupdate_weight_h: %f/  \n", update_weight_h[0]);
    printf("B: %f/%f/%f/%f/%f \n", B_h[0],B_h[1],B_h[2],B_h[3]);
    printf("B_temp: %f/%f/%f/%f/%f \n", B_temp_h[0],B_temp_h[1],B_temp_h[2],B_temp_h[3]);*/
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
