//
// Example 1: Single Process, Single Thread, Multiple Devices
//

#include <stdio.h>
#include "cuda_runtime.h"
#include "nccl.h"
#include <stdlib.h>
#include <unistd.h>


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


int performAllReduce(int gDevs, int* group, float **sendbuff, float **recvbuff, cudaStream_t* s, int size){

    ncclComm_t comms[gDevs];
    ncclCommInitAll(comms, gDevs, group);


    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < gDevs; ++i) {
      printf("salam allreduce started on device %d \n", group[i]);
      NCCLCHECK(ncclAllReduce((const void *) sendbuff[group[i]],
                                (void *) recvbuff[group[i]], size, ncclFloat, ncclSum,
                                comms[i], s[group[i]]));
    }
    NCCLCHECK(ncclGroupEnd());
    
    for (int i = 0; i < gDevs; ++i) {
        CUDACHECK(cudaSetDevice(group[i]));
        CUDACHECK(cudaStreamSynchronize(s[group[i]]));
    }

     for (int i = 0; i < gDevs; ++i) {
        NCCLCHECK(ncclCommDestroy(comms[i]));
    }

    return 0;
}


int main(int argc, char *argv[]) {

    int nDev = 4;
    int size = 2621440;


    float **sendbuff = (float **) malloc(nDev * sizeof(float *));
    float **recvbuff = (float **) malloc(nDev * sizeof(float *));
    float **data = (float **) malloc(nDev * sizeof(float *));
    cudaStream_t *s = (cudaStream_t *) malloc(sizeof(cudaStream_t) * nDev);
    

    for (int i = 0; i < nDev; ++i) {
        data[i] = (float *)malloc(size*sizeof(float));
        
        srand(20214229*i);

        for(int j = 0; j<size; ++j){
          data[i][j] = i;//(float)rand()/(RAND_MAX/100);
        }

        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaMalloc(sendbuff + i, size * sizeof(float)));
        CUDACHECK(cudaMalloc(recvbuff + i, size * sizeof(float)));
        CUDACHECK(cudaMemcpy(sendbuff[i], data[i], size*sizeof(float), cudaMemcpyHostToDevice));
        CUDACHECK(cudaMemset(recvbuff[i], 0, size*sizeof(float)));
        CUDACHECK(cudaStreamCreate(s + i));
    }

    //Communication starts here
    int group1[2] = {2,3};
    performAllReduce(2, group1, sendbuff, recvbuff, s, size);
    sleep(1);

    int group2[2] = {0,1};
    performAllReduce(2, group2, sendbuff, recvbuff, s, size);
    
    sleep(1);
    int group3[2] = {1,2};
    performAllReduce(2, group3, sendbuff, recvbuff, s, size);
    
    // int group1[4] = {0,1,2,3};
    // performAllReduce(4, group1, sendbuff, recvbuff, s, size);

    for (int i = 0; i < nDev; ++i) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaMemcpy(data[i], recvbuff[i], size*sizeof(float), cudaMemcpyDeviceToHost));
        CUDACHECK(cudaFree(sendbuff[i]));
        CUDACHECK(cudaFree(recvbuff[i]));
    }
    
    printf("%.12f \n", data[0][1434]);
    printf("%.12f \n", data[1][1434]);
    printf("%.12f \n", data[2][1434]);
    printf("%.12f \n", data[3][1434]);


   
    return 0;
}