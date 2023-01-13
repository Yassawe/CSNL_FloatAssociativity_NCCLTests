//
// Example 1: Single Process, Single Thread, Multiple Devices
//

#include <stdio.h>
#include "cuda_runtime.h"
#include "nccl.h"
#include <stdlib.h>
#include <unistd.h>
#include <math.h>

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


int performAllReduce(int gDevs, int* group, int* order, half **buff, cudaStream_t* s, int size){

    ncclComm_t comms[gDevs];
    ncclCommInitAll(comms, gDevs, group);

    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < gDevs; ++i) {
      NCCLCHECK(ncclAllReduce((const void *) buff[group[order[i]]],
                                (void *) buff[group[order[i]]], size, ncclFloat16, ncclSum,
                                comms[order[i]], s[group[order[i]]]));
    }
    NCCLCHECK(ncclGroupEnd());
    
    for (int i = 0; i < gDevs; ++i) {
        CUDACHECK(cudaSetDevice(group[order[i]]));
        CUDACHECK(cudaStreamSynchronize(s[group[order[i]]]));
    }

     for (int i = 0; i < gDevs; ++i) {
        NCCLCHECK(ncclCommDestroy(comms[i]));
    }

    return 0;
}

int setup(int nDev, half** buff, half** input, cudaStream_t* s, int size){
    for (int i = 0; i < nDev; ++i) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaMalloc(buff + i, size * sizeof(half)));
        CUDACHECK(cudaMemcpy(buff[i], input[i], size*sizeof(half), cudaMemcpyHostToDevice));
        CUDACHECK(cudaStreamCreate(s + i));
    }
    return 0;
}

int finishup(int nDev, half** buff, half** output, int size){
   for (int i = 0; i < nDev; ++i) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaMemcpy(output[i], buff[i], size*sizeof(half), cudaMemcpyDeviceToHost));
        CUDACHECK(cudaFree(buff[i]));
    }
    return 0;
}

void stats(float* difference, int size){
    float max = difference[0];
    float sum = 0;
    int nonzeros = 0;
    for (int i = 1; i < size; i++) {
        sum += difference[i];
        
        if (difference[i] > max) {
            max = difference[i];
        }

        if (difference[i]!=0) {
          nonzeros++;
        }

    }
    
    float meanWithZeros = sum / size;
    float meanWithoutZeros = sum/nonzeros;
    
    printf("Total elements in a message: %d\n", size);
    printf("Elements that are different: %d\n", nonzeros);
    printf("Maximum difference: %f\n", max);
    printf("Mean difference (including zeros): %f\n", meanWithZeros);
    printf("Mean difference (not including zeros): %f\n", meanWithoutZeros);
}


int main(int argc, char *argv[]) {

    int nDev = 4;
    int devices[4] = {0,1,2,3};
    int size = 1000000;
    int pair[2];

    half **buff = (half **) malloc(nDev * sizeof(half *));
    half **input = (half **) malloc(nDev * sizeof(half *));
    half **output1 = (half **) malloc(nDev * sizeof(half *)); // for AllReduce between all 4 processors simultaneously
    half **output2 = (half **) malloc(nDev * sizeof(half *)); // for consecutive res of partial allreduce
    cudaStream_t *s = (cudaStream_t *) malloc(sizeof(cudaStream_t) * nDev);
    
    for(int i=0; i<nDev; ++i){
        input[i] = (half *)malloc(size*sizeof(half));
        output1[i] = (half *)malloc(size*sizeof(half));
        output2[i] = (half *)malloc(size*sizeof(half));

        srand(20214229*i);

        for(int j = 0; j<size; ++j){
          input[i][j] = (half) (rand()/(RAND_MAX-1.0));
        }
    }


    int pairorder[2] = {0,1}; //order always expressed in indecies
    int order1[4] = {3,1,2,0};
    // int order2[4] = {0,1,2,3};

    // Partial Comm
    setup(nDev, buff, input, s, size);
    pair[0] = 0;
    pair[1] = 1;
    performAllReduce(2, pair, pairorder, buff, s, size);

    pair[0] = 2;
    pair[1] = 3;
    performAllReduce(2, pair, pairorder, buff, s, size);

    pair[0] = 0;
    pair[1] = 2;
    performAllReduce(2, pair, pairorder, buff, s, size);

    pair[0] = 1;
    pair[1] = 3;
    performAllReduce(2, pair, pairorder, buff, s, size);
    
    // performAllReduce(4, devices, order2, buff, s, size);
    finishup(nDev, buff, output1, size);

    //zusammen

    setup(nDev, buff, input, s, size);
    performAllReduce(4, devices, order1, buff, s, size);
    finishup(nDev, buff, output2, size);   
    
    //checking

    float* difference = (float *) malloc(size*sizeof(float));


    for(int device = 0; device<nDev; ++device){
        printf("\n\nDevice %d\n", device);
        
        for(int j=0; j<size; ++j){
            difference[j] = fabs(output1[device][j] - output2[device][j]);
            // if(output1[1][j]!=output2[1][j]){
            //   printf("Found difference, device %d index %d, %.12f vs %.12f \n", i, j, (float) output1[i][j], (float) output2[i][j]);
            // }  
        }

        stats(difference, size);
    }
    //cleaning up
    for(int i = 0; i<nDev; ++i){
      free(input[i]);
      free(output1[i]);
      free(output2[i]);
    }

    free(difference);

    return 0;
}