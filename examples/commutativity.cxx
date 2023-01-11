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
        data[devs[i]] = (float *)malloc(size*sizeof(float));
        
        srand(20214229*devs[i]);

        for(int j = 0; j<size; ++j){
          data[devs[i]][j] = (float)rand()/(RAND_MAX);
        }
    }


    for (int i = 0; i < nDev; ++i) {
        CUDACHECK(cudaSetDevice(devs[i]));
        CUDACHECK(cudaMalloc(sendbuff + devs[i], size * sizeof(float)));
        CUDACHECK(cudaMalloc(recvbuff + devs[i], size * sizeof(float)));
        CUDACHECK(cudaMemcpy(sendbuff[devs[i]], data[devs[i]], size*sizeof(float), cudaMemcpyHostToDevice));
        CUDACHECK(cudaMemset(recvbuff[devs[i]], 0, size*sizeof(float)));
        CUDACHECK(cudaStreamCreate(s + devs[i]));
    }

    // initializing NCCL
    NCCLCHECK(ncclCommInitAll(comms, nDev, devs));

    // calling NCCL communication API. Group API is required when
    // using multiple devices per thread
    NCCLCHECK(ncclGroupStart());


    //ORDER OF OPERATIONS IS DEFINED HERE

    // int order[4] = {3, 1, 2, 0}; 


    for (int i = 0; i < nDev; ++i) {

        NCCLCHECK(ncclAllReduce((const void *) sendbuff[devs[i]],
                                (void *) recvbuff[devs[i]], size, ncclFloat, ncclSum,
                                comms[devs[i]], s[devs[i]]));
    }


    NCCLCHECK(ncclGroupEnd());

    // synchronizing on CUDA streams to wait for completion of NCCL operation
    for (int i = 0; i < nDev; ++i) {
        CUDACHECK(cudaSetDevice(devs[i]));
        CUDACHECK(cudaStreamSynchronize(s[devs[i]]));
    }

   

    // free device buffers
    for (int i = 0; i < nDev; ++i) {
        CUDACHECK(cudaSetDevice(devs[i]));
        CUDACHECK(cudaMemcpy(data[devs[i]], recvbuff[devs[i]], size*sizeof(float), cudaMemcpyDeviceToHost));
        CUDACHECK(cudaFree(sendbuff[devs[i]]));
        CUDACHECK(cudaFree(recvbuff[devs[i]]));
    }

    // finalizing NCCL
   
    for (int i = 0; i < nDev; ++i) {
        printf("%.12f \n", data[devs[i]][12345]);
    }

    for (int i = 0; i < nDev; ++i) {
        ncclCommDestroy(comms[devs[i]]);
    }

    return 0;
}