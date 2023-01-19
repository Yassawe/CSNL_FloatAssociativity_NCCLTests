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


typedef half datatype;
#define NCCLTYPE ncclFloat16

 
// typedef float datatype;
// #define NCCLTYPE ncclFloat


int performAllReduce(int gDevs, int* group, int* order, datatype **buff, cudaStream_t* s, int size){

    ncclComm_t comms[gDevs];
    ncclCommInitAll(comms, gDevs, group);

    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < gDevs; ++i) {
      NCCLCHECK(ncclAllReduce((const void *) buff[group[order[i]]],
                                (void *) buff[group[order[i]]], size, NCCLTYPE, ncclSum,
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

int setup(int nDev, datatype** buff, datatype** input, cudaStream_t* s, int size){
    for (int i = 0; i < nDev; ++i) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaMalloc(buff + i, size * sizeof(datatype)));
        CUDACHECK(cudaMemcpy(buff[i], input[i], size*sizeof(datatype), cudaMemcpyHostToDevice));
        CUDACHECK(cudaStreamCreate(s + i));
    }
    return 0;
}

int finishup(int nDev, datatype** buff, datatype** output, int size){
   for (int i = 0; i < nDev; ++i) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaMemcpy(output[i], buff[i], size*sizeof(datatype), cudaMemcpyDeviceToHost));
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
    printf("Maximum difference: %.10f\n", max);
    printf("Mean difference (including zeros): %.10f\n", meanWithZeros);
    printf("Mean difference (not including zeros): %.16f\n\n", meanWithoutZeros);
}


int main(int argc, char *argv[]) {
    int nDev = 4;
    int devices[4] = {0,1,2,3};
    int size = 1000000;
    int pair[2];

    datatype **buff = (datatype **) malloc(nDev * sizeof(datatype *));
    datatype **input = (datatype **) malloc(nDev * sizeof(datatype *));
    datatype **output1 = (datatype **) malloc(nDev * sizeof(datatype *)); // for AllReduce between all 4 processors simultaneously
    datatype **output2 = (datatype **) malloc(nDev * sizeof(datatype *)); // for consecutive res of partial allreduce
    cudaStream_t *s = (cudaStream_t *) malloc(nDev*sizeof(cudaStream_t));
    
    for(int i=0; i<nDev; ++i){
        input[i] = (datatype *)malloc(size*sizeof(datatype));
        output1[i] = (datatype *)malloc(size*sizeof(datatype));
        output2[i] = (datatype *)malloc(size*sizeof(datatype));

        srand(20214229*i);

        for(int j = 0; j<size; ++j){
          input[i][j] = (datatype) (rand()/(RAND_MAX-1.0));
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
            // if(output1[device][j]!=output2[device][j]){
            //   printf("Found difference, index %d, %.12f vs %.12f \n", j, (float) output1[device][j], (float) output2[device][j]);
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