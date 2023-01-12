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

int setup(int nDev, float** buff, float** input, cudaStream_t* s, int size){
    for (int i = 0; i < nDev; ++i) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaMalloc(buff + i, size * sizeof(float)));
        CUDACHECK(cudaMemcpy(buff[i], input[i], size*sizeof(float), cudaMemcpyHostToDevice));
        CUDACHECK(cudaStreamCreate(s + i));
    }
}

int finishup(int nDev, float** buff, float** output, int size){
   for (int i = 0; i < nDev; ++i) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaMemcpy(output[i], buff[i], size*sizeof(float), cudaMemcpyDeviceToHost));
        CUDACHECK(cudaFree(buff[i]));
    }
}

int main(int argc, char *argv[]) {

    int nDev = 4;
    int size = 2621440;
    int pair[2];

    float **buff = (float **) malloc(nDev * sizeof(float *));
    float **input = (float **) malloc(nDev * sizeof(float *));
    float **outputTogether = (float **) malloc(nDev * sizeof(float *)); // for AllReduce between all 4 processors simultaneously
    float **outputPartial = (float **) malloc(nDev * sizeof(float *)); // for consecutive res of partial allreduce
    cudaStream_t *s = (cudaStream_t *) malloc(sizeof(cudaStream_t) * nDev);
    
    for(int i=0; i<nDev; ++i){
        input[i] = (float *)malloc(size*sizeof(float));
        outputTogether[i] = (float *)malloc(size*sizeof(float));
        outputPartial[i] = (float *)malloc(size*sizeof(float));

        srand(20214229*i);

        for(int j = 0; j<size; ++j){
          input[i][j] = (float)rand()/(RAND_MAX/100);
        }
    }


    // Partial Comm
    setup(nDev, buff, input, s, size);
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
    
    finishup(nDev, buff, outputPartial, size);

    //zusammen

    int devices[4] = {0,1,2,3};
    setup(nDev, buff, input, s, size);
    performAllReduce(4, devices, buff, s, size);
    finishup(nDev, buff, outputTogether, size);   
    
    //checking




    //cleaning up
    for(int i = 0; i<nDev; ++i){
      free(input[i]);
      free(outputTogether[i]);
      free(outputPartial[i]);
    }

    return 0;
}