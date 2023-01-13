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

    // managing 4 devices
    int nDev = 4;
    int size = 2621440;
    int devs[4] = {0,1,2,3};

    ncclComm_t comms[nDev];

    // allocating and initializing device buffers
    float **sendbuff = (float **) malloc(nDev * sizeof(float *));
    float **recvbuff = (float **) malloc(nDev * sizeof(float *));
    cudaStream_t *s = (cudaStream_t *) malloc(sizeof(cudaStream_t) * nDev);
    

    float **data = (float **) malloc(nDev * sizeof(float *));

    for (int i = 0; i<nDev; ++i){
        data[i] = (float *)malloc(size*sizeof(float));
        
        srand(20214229*devs[i]);

        for(int j = 0; j<size; ++j){
          data[i][j] = (float)rand()/(RAND_MAX);
        }
    }


    for (int i = 0; i < nDev; ++i) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaMalloc(sendbuff + i, size * sizeof(float)));
        CUDACHECK(cudaMalloc(recvbuff + i, size * sizeof(float)));
        CUDACHECK(cudaMemcpy(sendbuff[i], data[i], size*sizeof(float), cudaMemcpyHostToDevice));
        CUDACHECK(cudaMemset(recvbuff[i], 0, size*sizeof(float)));
        CUDACHECK(cudaStreamCreate(s + i));
    }

    // initializing NCCL
    NCCLCHECK(ncclCommInitAll(comms, nDev, devs));

    // calling NCCL communication API. Group API is required when
    // using multiple devices per thread
    NCCLCHECK(ncclGroupStart());


    //ORDER OF OPERATIONS IS DEFINED HERE

    int order[4] = {3,0,1,2}; 


    for (int i = 0; i < nDev; ++i) {
        printf("\nallreduce on device %d\n", order[i]);
        NCCLCHECK(ncclAllReduce((const void *) sendbuff[order[i]],
                                (void *) recvbuff[order[i]], size, ncclFloat, ncclSum,
                                comms[order[i]], s[order[i]]));
    }


    NCCLCHECK(ncclGroupEnd());

    // synchronizing on CUDA streams to wait for completion of NCCL operation
    for (int i = 0; i < nDev; ++i) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaStreamSynchronize(s[i]));
    }

    for (int i = 0; i < nDev; ++i) {
        ncclCommDestroy(comms[i]);
    }

   

    // free device buffers
    for (int i = 0; i < nDev; ++i) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaMemcpy(data[i], recvbuff[i], size*sizeof(float), cudaMemcpyDeviceToHost));
        CUDACHECK(cudaFree(sendbuff[i]));
        CUDACHECK(cudaFree(recvbuff[i]));
    }

    // finalizing NCCL
   
    for (int i = 0; i < nDev; ++i) {
        printf("%.12f \n", data[i][12345]);
    }

    for (int i = 0; i < nDev; ++i) {
        ncclCommDestroy(comms[i]);
    }

    return 0;
}