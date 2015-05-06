#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#include <string.h>

#define DATA_LENGTH 100

#define CUDA_CALL(X) X; // {if(cudaError == X){printf("Error Calling %s at line %s\n", #X, __LINE__);}}
float * genInput(int l);

void verify(float *a, float *b, float *c, int l);

__global__ void vecAdd(float *in1, float *in2, float *out, int len) {
  int i=0;
  int tid = threadIdx.x;

  for(i=0; i<DATA_LENGTH; i++){
    // first half of first warp and second half of 2nd warp
    if(tid < 16 || tid > 47)
      out[i] = in1[i]+in2[i];
    else
      out[i] = in1[i]+in2[i];
  }

  for(i=0; i<DATA_LENGTH; i++){
    // only even threads not compactable
    if(threadIdx.x%2 == 0 )
      out[i] = in1[i]+in2[i];
    else
      out[i] = in1[i]+in2[i];
  }
  out[0] = in1[0] + in2[0];
}

int main(int argc, char **argv) {
  int inputLength;
  float *hostInput1;
  float *hostInput2;
  float *hostOutput;
  float *deviceInput1;
  float *deviceInput2;
  float *deviceOutput;
  struct timeval t;
  gettimeofday(&t, NULL);
  srand(t.tv_sec);

  inputLength = DATA_LENGTH;

  hostInput1 = genInput(inputLength);
  hostInput2 = genInput(inputLength);
  hostOutput = ( float * )malloc(inputLength * sizeof(float));

  //@@ Allocate GPU memory here
  CUDA_CALL(cudaMalloc((void**)&deviceInput1, inputLength*sizeof(float)));
  CUDA_CALL(cudaMalloc((void**)&deviceInput2, inputLength*sizeof(float)));
  CUDA_CALL(cudaMalloc((void**)&deviceOutput, inputLength*sizeof(float)));

  //@@ Copy memory to the GPU here
  CUDA_CALL(cudaMemcpy(deviceInput1, hostInput1, sizeof(float)*inputLength, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(deviceInput2, hostInput2, sizeof(float)*inputLength, cudaMemcpyHostToDevice));

  //@@ Initialize the grid and block dimensions here
  dim3 numBlocks(1,1,1);
  //dim3 numThreads(ThreadsPerBlock,1,1);
  dim3 numThreads(64,1,1);

  //@@ Launch the GPU Kernel here
  vecAdd<<<numBlocks, numThreads>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);

  cudaDeviceSynchronize();

  //@@ Copy the GPU memory back to the CPU here
  CUDA_CALL(cudaMemcpy(hostOutput, deviceOutput, inputLength*sizeof(float), cudaMemcpyDeviceToHost));

  //@@ Free the GPU memory here
  CUDA_CALL(cudaFree(deviceInput1));
  CUDA_CALL(cudaFree(deviceInput2));
  CUDA_CALL(cudaFree(deviceOutput));

  verify(hostInput1, hostInput2, hostOutput, inputLength);

  free(hostInput1);
  free(hostInput2);
  free(hostOutput);

  return 0;
}

float * genInput(int l)
{
  int i;
  float * arr = (float*)malloc(l*sizeof(float));
  for(i=0; i<l; i++){
    arr[i] = rand();
    arr[i] = arr[i]/rand();
  }
  return arr;
}

void verify(float *a, float *b, float *c, int l)
{
  char buff1[50] = {0};
  char buff2[50] = {0};
  int i;
  for(i=0; i<l; i++){
    float d = a[i]+b[i];
    sprintf(buff1, "%1.8f", d); 
    sprintf(buff2, "%1.8f", c[i]);
    if(strcmp(buff1, buff2) != 0){
      printf("ERROR at index %d, Exp %1.8f Got %1.8f\n",i,d,c[i]);
      break;
    }
  }
}


