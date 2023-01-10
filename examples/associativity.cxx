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
    ncclUniqueId Ids[4];

    int nDev = 4;
    int size = 2621440;
    int devs[4] = {0, 1, 2, 3};


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


    int nGroup = 2;
    int order[2] = {0, 1};

    for(int i = 0; i<nGroup; ++i){
        ncclGetUniqueId(&Ids[order[i]]);
    }
    
    ncclGroupStart();
    for(int i = 0; i<nGroup; ++i){
        cudaSetDevice(order[i]);
        ncclCommInitRank(&comms[i], nGroup, Ids[i], i);
    }
    ncclGroupEnd();

    ncclGroupStart();
    for (int i = 0; i < nGroup; ++i) {
        ncclAllReduce((const void *) sendbuff[order[i]],
                                (void *) recvbuff[order[i]], size, ncclFloat, ncclSum,
                                comms[order[i]], s[order[i]]);
    }
    ncclGroupEnd();

    for (int i = 0; i < nGroup; ++i) {
        cudaSetDevice(order[i]);
        cudaStreamSynchronize(s[order[i]]);
    }

    for (int i = 0; i < nGroup; ++i) {
        cudaSetDevice(order[i]);
        cudaMemcpy(data[order[i]], recvbuff[order[i]], size*sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(sendbuff[order[i]]);
        cudaFree(recvbuff[order[i]]);
    }

    // finalizing NCCL
    for (int i = 0; i < nGroup; ++i) {
        ncclCommDestroy(comms[order[i]]);
    }

    // printf("%.12f \n", data[0][12345]);
    printf("%.12f \n", data[1][12345]);
    printf("%.12f \n", data[2][12345]);
    // printf("%.12f \n", data[3][12345]);

    return 0;
}