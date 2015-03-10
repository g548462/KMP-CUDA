#include<stdlib.h>
#include <stdio.h>
#include <string.h>
#include<omp.h>
#include<time.h>
#define Size 100000
#define patternSize 20
#define patternNum 100000
#define ThreadNum 500
#define BlockNum 200
#define GPUNUMS 4

void cpu_preKmp(char *x, int m, int kmpNext[])
{
	int i, j;
	i = 0;
	j = kmpNext[0] = -1;
	while(i < m)
	{
		while(j>-1 && x[i]!=x[j])
			j = kmpNext[j];
		i++;
		j++;
		if(x[i]==x[j])
			kmpNext[i] = kmpNext[j];
		else
			kmpNext[i] = j;

	}
}

/*******************************************

This variable   m:pattern.length    x:pattern
				n:array.length      y:array

*******************************************/
void cpu_KMP(char *x, int m, char *y, int n,int *answer)
{
	int i, j, kmpNext[patternSize], id=0;

	cpu_preKmp(x,m,kmpNext);
	i = j = 0;
	while(j < n)
	{
		while(i>-1 && x[i]!=y[j])
		{
		  	i = kmpNext[i];
		}
		i++;
		j++;
		if(i >= m)
		{
			i = kmpNext[i];
			answer[id++]=j-2;	
		}

	}
}
__device__ void preKmp(char *x, int m, int kmpNext[])
{
	int i, j;
	i = 0;
	j = kmpNext[0] = -1;
	while(i < m)
	{
		while(j>-1 && x[i]!=x[j])
			j = kmpNext[j];
		i++;
		j++;
		if(x[i]==x[j])
			kmpNext[i] = kmpNext[j];
		else
			kmpNext[i] = j;

	}
}

/*******************************************

This variable   m:pattern.length    x:pattern
				n:array.length      y:array

*******************************************/
__device__ void KMP(char *x, int m, char *y, int n,int *answer,int id)
{
	int i, j, kmpNext[patternSize];

	preKmp(x,m,kmpNext);
	i = j = 0;
	while(j < n)
	{
		while(i>-1 && x[i]!=y[j])
		{
		  	i = kmpNext[i];
		}
		i++;
		j++;
		if(i >= m)
		{
			i = kmpNext[i];
			answer[id]=j-2;	
		}

	}
}

__global__ void kmp_kernel(char *array,char *pattern,int *answer)
{
  int id=blockIdx.x*blockDim.x+threadIdx.x;
  char *p;
  p=&pattern[id*(patternSize+1)];
  KMP(p,patternSize,array,Size,answer,id);
   
}

int main(int argc,char *argv[])
{
  int i=0,j=0,tmp,*answer;
  
  char *array,*b,*pattern;
  
  srand(time(0));
  array=(char*)malloc(sizeof(char)*Size);
  b=(char*)malloc(sizeof(char)*26);
  pattern=(char*)malloc(sizeof(char)*(patternSize+1)*patternNum);
  answer=(int*)malloc(sizeof(int)*patternNum);
  /************************************
  *   cudaMalloc
  ************************************/


  b="abcdefghijklmnopqrstuvwxyz";
  for(i=0;i<Size;i++)
	array[i]=b[rand()%26];

  for(i=0;i<patternNum;i++)
  {
	tmp=rand()%(Size-patternSize);
	for(j=0;j<patternSize+1;j++)
	{
	  if(j!=patternSize)
	  {
		pattern[i*(patternSize+1)+j]=array[tmp++];
		//printf("%d   %c\n",i,array[tmp-1]);
	  }
	  else
	  {
		//printf("===================== %d   \n",j);
		pattern[i*(patternSize+1)+j]='\0';
		//printf("%c\n",pattern[i*patternSize+j]);
	  }
	}
  }
  for(i=0;i<patternNum;i++)
  {
	answer[i]=0;
  }


//CUDA KMP===============================================	
  int num_gpus = 0; 
  cudaGetDeviceCount(&num_gpus);
  if(num_gpus < 1){
		printf("no CUDA capable devices were detected\n");
   	    exit(1);
  }
  omp_set_num_threads(num_gpus);  // create as many CPU threads as there are CUDA devices
	printf("num gpus:%d\n", num_gpus);

  float elapsedTime;

  int hostThread = GPUNUMS;

  #pragma omp parallel num_threads(hostThread)
  {
	  cudaError_t r;
	  unsigned int cpu_thread_id = omp_get_thread_num();
	  unsigned int num_cpu_threads = omp_get_num_threads();
	  int gpu_id = -1,*d_answer;
	  cudaSetDevice(cpu_thread_id % num_gpus);        // "% num_gpus" allows more CPU threads than GPU devices
    cudaGetDevice(&gpu_id);
	  char *d_array,*d_pattern;


	  r=cudaMalloc((void**)&d_array,sizeof(char)*Size);
	  printf("cudaMalloc d_array : %s\n",cudaGetErrorString(r));
	  r=cudaMalloc((void**)&d_pattern,sizeof(char)*(patternSize+1)*patternNum/num_gpus);
	  printf("cudaMalloc d_pattern : %s\n",cudaGetErrorString(r));
	  r=cudaMalloc((void**)&d_answer,sizeof(int)*patternNum/num_gpus);
	  printf("cudaMalloc d_answer : %s\n",cudaGetErrorString(r));


	  r=cudaMemcpy(d_array,array,sizeof(char)*Size,cudaMemcpyHostToDevice);
	  printf("Memcpy H->D d_array : %s\n",cudaGetErrorString(r));
	  
	  r=cudaMemcpy(d_pattern,pattern+(patternSize+1)*patternNum/num_gpus*cpu_thread_id,sizeof(char)*(patternSize+1)*patternNum/num_gpus,cudaMemcpyHostToDevice);
	  printf("Memcpy H->D d_pattern : %s\n",cudaGetErrorString(r));
	  
	  r=cudaMemcpy(d_answer,answer+patternNum/num_gpus*cpu_thread_id,sizeof(int)*patternNum/num_gpus,cudaMemcpyHostToDevice);
	  printf("Memcpy H->D d_answer : %s\n",cudaGetErrorString(r));
	  
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);

	  kmp_kernel<<<BlockNum, ThreadNum>>>(d_array, d_pattern, d_answer);

		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsedTime, start, stop);


	  r=cudaMemcpy(answer+patternNum/num_gpus*cpu_thread_id,d_answer,sizeof(int)*patternNum/num_gpus,cudaMemcpyDeviceToHost);
	  printf("Memcpy D->H answer : %s\n",cudaGetErrorString(r));
  }

 /* printf("Array:\n");
  printf("%s\n", array);
  for(i=0;i<(patternSize+1)*patternNum;i++)
	  printf("%c", pattern[i]);
  printf("\n\n");
  for(i=0;i<patternNum;i++)
	printf("%d, %d\n", i, answer[i]);
 */
//CPU KMP=================================================
	for(i=0;i<patternNum;i++)
		answer[i] = 0;
	
	float cpu_start, cpu_stop;
	float CPU_exe_time;

	cpu_start = (float)clock();
	for(int i=0;i<patternNum;i++)
		cpu_KMP(pattern+21*i, strlen(pattern), array, strlen(array), answer);
	cpu_stop = (float)clock();
	CPU_exe_time = (cpu_stop - cpu_start)/(float)CLOCKS_PER_SEC;

	//printf("pattern size:%d\n", strlen(pattern));
	printf("CUDA execution time: %f\n", elapsedTime/1000);
	printf("CPU execution time: %f\n", CPU_exe_time);
	printf("speedup: %f\n", CPU_exe_time/(elapsedTime/1000));
  return 0;
}
