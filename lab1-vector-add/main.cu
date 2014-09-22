/******************************************************************************
 *cr
 *cr         (C) Copyright 2010-2013 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdio.h>
#include "support.h"
#include "kernel.cu"

int main(int argc, char**argv) {

    Timer timer;
    cudaError_t cuda_ret;
    int n;
	const unsigned int THREADS_PER_BLOCK = 512;
	unsigned int numBlocks;

    // Initialize host variables ----------------------------------------------
    printf("\nSetting up the problem..."); fflush(stdout);
    startTime(&timer);

    if(argc == 1) {
        n = 10000;
    } else if(argc == 2) {
        n = atoi(argv[1]);
    } else {
        printf("\n    Invalid input parameters!"
           "\n    Usage: ./vecadd               # Vector of size 10,000 is used"
           "\n    Usage: ./vecadd <m>           # Vector of size m is used"
           "\n");
        exit(0);
    }

    float* A_h = (float*) malloc( sizeof(float)*n );
    for (unsigned int i=0; i < n; i++) { A_h[i] = (rand()%100)/100.00; }

    float* B_h = (float*) malloc( sizeof(float)*n );
    for (unsigned int i=0; i < n; i++) { B_h[i] = (rand()%100)/100.00; }

    float* C_h = (float*) malloc( sizeof(float)*n );

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    printf("    Vector size = %u\n", n);

    // Allocate device variables ----------------------------------------------

    printf("Allocating device variables..."); fflush(stdout);
    startTime(&timer);
	float* A_d;
	float* B_d;
	float* C_d;
	//Allocating memory on the GPU
	cudaMalloc((void**)&A_d, n*sizeof(float));
	cudaMalloc((void**)&B_d, n*sizeof(float));
	cudaMalloc((void**)&C_d, n*sizeof(float));


    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy host variables to device ------------------------------------------

    printf("Copying data from host to device..."); fflush(stdout);
    startTime(&timer);

    //copying all three vectors from cpu to device
	cudaMemcpy(A_d, A_h, sizeof(float)*n, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, sizeof(float)*n, cudaMemcpyHostToDevice);

   //wait for all copying to finish before launching the kernel
    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
	
	//calculate numBlocks 
	numBlocks = (n - 1)/THREADS_PER_BLOCK + 1;
    // Launch kernel ----------------------------------------------------------

    printf("Launching kernel..."); fflush(stdout);
    startTime(&timer);
	
	//Setting up dimensions and launching the kernel
	dim3 gridDim(numBlocks, 1, 1);
	dim3 blockDim(THREADS_PER_BLOCK, 1, 1);
    vecAddKernel<<< gridDim, blockDim >>> (A_d, B_d, C_d, n);

	//wait for kernel stream to finish
    cuda_ret = cudaDeviceSynchronize();
	if(cuda_ret != cudaSuccess) FATAL("Unable to launch kernel");
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy device variables from host ----------------------------------------

    printf("Copying data from device to host..."); fflush(stdout);
    startTime(&timer);

    cudaMemcpy(C_h, C_d, sizeof(float)*n, cudaMemcpyDeviceToHost);

	//wait for all copying to finish
    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Verify correctness -----------------------------------------------------

    printf("Verifying results..."); fflush(stdout);

    verify(A_h, B_h, C_h, n);

    // Free memory ------------------------------------------------------------

    free(A_h);
    free(B_h);
    free(C_h);

    //Freeing memory on device
	cudaFree(A_d);
	cudaFree(B_d);
	cudaFree(C_d);

     //yay! successful!

    return 0;
}

