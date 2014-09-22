#include <wb.h>

#define wbCheck(stmt)                                                          \
  do {                                                                         \
    cudaError_t err = stmt;                                                    \
    if (err != cudaSuccess) {                                                  \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                              \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));           \
      return -1;                                                               \
    }                                                                          \
  } while (0)

// Compute C = A * B
__global__ void matrixMultiply(float *A, float *B, float *C, int numARows,
                               int numAColumns, int numBRows, int numBColumns,
                               int numCRows, int numCColumns) {
	
  //@@ Insert code to implement matrix multiplication here
int Row = blockIdx.y*blockDim.y + threadIdx.y;
int Column = blockIdx.x*blockDim.x + threadIdx.x;
//checking if matrices are multipliable
	if (numAColumns != numBRows) 
	return;
//Checking the boundry conditions
 if ((Row < numARows) && (Column < numBColumns)) {
	float P = 0.00;
	//comuting and adding elements of C
	for (int k = 0; k < numAColumns; ++k)
    P += A[Row*numAColumns+k] * B[k*numBRows+Column];
	 
   	C[Row*numCColumns+Column] = P;
	
  }


		
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set this)
  int numThreads; //num threads
  args = wbArg_read(argc, argv);
  //initializing it to 32
	numThreads = 32;
  wbTime_start(Generic, "Importing data and creating memory on host");
  hostA =( float * )wbImport(wbArg_getInputFile(args, 0), &numARows, &numAColumns);
  hostB =( float * )wbImport(wbArg_getInputFile(args, 1), &numBRows, &numBColumns);
  //@@ Set numCRows and numCColumns
  numCRows = numARows;
  numCColumns = numBColumns;
	
  //@@ Allocate the hostC matrix
	
	hostC = (float*) malloc( sizeof(float)*numCRows*numCColumns);
	
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

  wbTime_start(GPU, "Allocating GPU memory.");
	
  //@@ Allocate GPU memory here
	
	cudaMalloc((void**)&deviceA, numARows*numAColumns*sizeof(float));
	cudaMalloc((void**)&deviceB, numBRows*numBColumns*sizeof(float));
	cudaMalloc((void**)&deviceC, numCRows*numCColumns*sizeof(float));
	
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
	
  //@@ Copy memory to the GPU here
	
	cudaMemcpy(deviceA, hostA, numARows*numAColumns*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(deviceB, hostB, numBRows*numBColumns*sizeof(float),cudaMemcpyHostToDevice);
	
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
	dim3 DimGrid((numCColumns - 1) / numThreads + 1, (numCRows - 1) / numThreads + 1, 1);
    dim3 DimBlock(numThreads , numThreads, 1);
	
	wbTime_start(Compute, "Performing CUDA computation");
	
  //@@ Launch the GPU Kernel here
	
 matrixMultiply<<<DimGrid , DimBlock>>>(deviceA , deviceB , deviceC , numARows , numAColumns, numBRows ,numBColumns , numCRows , numCColumns);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
	cudaMemcpy(hostC, deviceC, numCRows*numCColumns*sizeof(float), cudaMemcpyDeviceToHost);

  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
	
  //@@ Free the GPU memory here
	cudaFree(deviceA);
	cudaFree(deviceB);
	cudaFree(deviceC);
	
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}
