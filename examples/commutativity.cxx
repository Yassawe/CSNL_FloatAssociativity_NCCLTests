//
// Example 1: Single Process, Single Thread, Multiple Devices
//

#include <stdio.h>
#include "cuda_runtime.h"
#include "nccl.h"
#include <stdlib.h>

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",       \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

int main(int argc, char *argv[]) {

    ncclComm_t comms[4];

    // managing 4 devices
    int nDev = 4;
    int size = 2621440;
    int devs[4] = {0, 1, 2, 3};

    // allocating and initializing device buffers
    float **sendbuff = (float **) malloc(nDev * sizeof(float *));
    float **recvbuff = (float **) malloc(nDev * sizeof(float *));
    cudaStream_t *s = (cudaStream_t *) malloc(sizeof(cudaStream_t) * nDev);
    

    float **data = (float **) malloc(nDev * sizeof(float *));

    for (int i = 0; i<nDev; ++i){
        data[i] = (float *)malloc(size*sizeof(float));
        
        srand(20214229*i);

        for(int j = 0; j<size; ++j){
          data[i][j] = (float)rand()/(RAND_MAX);
        }
    }


    for (int i = 0; i < nDev; ++i) {
        cudaSetDevice(i);
        cudaMalloc(sendbuff + i, size * sizeof(float));
        cudaMalloc(recvbuff + i, size * sizeof(float));
        cudaMemcpy(sendbuff[i], data[i], size*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(recvbuff[i], 0, size*sizeof(float));
        cudaStreamCreate(s + i);
    }

    // initializing NCCL
    ncclCommInitAll(comms, nDev, devs);

    // calling NCCL communication API. Group API is required when
    // using multiple devices per thread
    ncclGroupStart();


    //ORDER OF OPERATIONS IS DEFINED HERE

    int order[4] = {3, 1, 0, 2}; 

    for (int i = 0; i < nDev; ++i) {

        ncclAllReduce((const void *) sendbuff[order[i]],
                                (void *) recvbuff[order[i]], size, ncclFloat, ncclSum,
                                comms[order[i]], s[order[i]]);
    }


    ncclGroupEnd();

    // synchronizing on CUDA streams to wait for completion of NCCL operation
    for (int i = 0; i < nDev; ++i) {
        cudaSetDevice(i);
        cudaStreamSynchronize(s[i]);
    }

    // free device buffers
    for (int i = 0; i < nDev; ++i) {
        cudaSetDevice(i);
        cudaMemcpy(data[i], recvbuff[i], size*sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(sendbuff[i]);
        cudaFree(recvbuff[i]);
    }

    // finalizing NCCL
    for (int i = 0; i < nDev; ++i) {
        ncclCommDestroy(comms[i]);
    }

    printf("%.12f \n", data[0][12345]);
    printf("%.12f \n", data[1][12345]);
    printf("%.12f \n", data[2][12345]);
    printf("%.12f \n", data[3][12345]);




    return 0;
}