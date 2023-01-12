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


int performAllReduce(int gDevs, int* group, float **buff, cudaStream_t* s, int size){

    ncclComm_t comms[gDevs];
    ncclCommInitAll(comms, gDevs, group);


    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < gDevs; ++i) {
      printf("salam allreduce started on device %d \n", group[i]);
      NCCLCHECK(ncclAllReduce((const void *) buff[group[i]],
                                (void *) buff[group[i]], size, ncclFloat, ncclSum,
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
    int pair[2];

    float **buff = (float **) malloc(nDev * sizeof(float *));
    float **data = (float **) malloc(nDev * sizeof(float *));
    cudaStream_t *s = (cudaStream_t *) malloc(sizeof(cudaStream_t) * nDev);
    

    for (int i = 0; i < nDev; ++i) {
        data[i] = (float *)malloc(size*sizeof(float));
        
        srand(20214229*i);

        for(int j = 0; j<size; ++j){
          data[i][j] = (float)rand()/(RAND_MAX/100);
        }

        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaMalloc(buff + i, size * sizeof(float)));
        CUDACHECK(cudaMemcpy(buff[i], data[i], size*sizeof(float), cudaMemcpyHostToDevice));
        CUDACHECK(cudaStreamCreate(s + i));
    }

    //Communication starts here
    pair[0] = 0;
    pair[1] = 1;
    performAllReduce(2, pair, buff, s, size);

    pair[0] = 2;
    pair[1] = 3;
    performAllReduce(2, pair, buff, s, size);
   
    pair[0] = 0;
    pair[1] = 2;
    performAllReduce(2, pair, buff, s, size);

    pair[0] = 1;
    pair[1] = 3;
    performAllReduce(2, pair, buff, s, size);
    
    // int allDevices[4] = {0,1,2,3};
    // performAllReduce(4, allDevices, buff, s, size);

    for (int i = 0; i < nDev; ++i) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaMemcpy(data[i], buff[i], size*sizeof(float), cudaMemcpyDeviceToHost));
        CUDACHECK(cudaFree(buff[i]));
    }
    
    printf("%.15f \n", data[0][1434]);
    printf("%.15f \n", data[1][1434]);
    printf("%.15f \n", data[2][1434]);
    printf("%.15f \n", data[3][1434]);


   
    return 0;
}