#include<stdlib.h>
#include <stdio.h>
#include <string.h>
#include<time.h>
#define Size 10
#define patternSize 3
#define patternNum 20
#define ThreadNum 20 
#define BlockNum 1

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
	int i, j, kmpNext[Size];

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
			answer[id]=j-1;	
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
  int i=0,j=0,tmp,*answer,*d_answer;
  cudaError_t r;
  char *array,*b,*pattern;
  char *d_array,*d_pattern;



  srand(time(0));
  array=(char*)malloc(sizeof(char)*Size);
  b=(char*)malloc(sizeof(char)*26);
  pattern=(char*)malloc(sizeof(char)*(patternSize+1)*patternNum);
  answer=(int*)malloc(sizeof(int)*patternNum);
  /************************************
  *   cudaMalloc
  ************************************/

  r=cudaMalloc((void**)&d_array,sizeof(char)*Size);
  printf("cudaMalloc d_array : %s\n",cudaGetErrorString(r));
  r=cudaMalloc((void**)&d_pattern,sizeof(char)*(patternSize+1)*patternNum);
  printf("cudaMalloc d_pattern : %s\n",cudaGetErrorString(r));
  r=cudaMalloc((void**)&d_answer,sizeof(int)*patternNum);
  printf("cudaMalloc d_answer : %s\n",cudaGetErrorString(r));


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
		printf("%d   %c\n",i,array[tmp-1]);
	  }
	  else
	  {
		printf("===================== %d   \n",j);
		pattern[i*(patternSize+1)+j]='\0';
		printf("%c\n",pattern[i*patternSize+j]);
	  }
	}
  }
  for(i=0;i<patternNum;i++)
  {
	answer[i]=0;
  }


  r=cudaMemcpy(d_array,array,sizeof(char)*Size,cudaMemcpyHostToDevice);
  printf("Memcpy H->D d_array : %s\n",cudaGetErrorString(r));
  
  r=cudaMemcpy(d_pattern,pattern,sizeof(char)*(patternSize+1)*patternNum,cudaMemcpyHostToDevice);
  printf("Memcpy H->D d_pattern : %s\n",cudaGetErrorString(r));
  
  r=cudaMemcpy(d_answer,answer,sizeof(int)*patternNum,cudaMemcpyHostToDevice);
  printf("Memcpy H->D d_answer : %s\n",cudaGetErrorString(r));
  
  kmp_kernel<<<BlockNum, ThreadNum>>>(d_array, d_pattern, d_answer);

  r=cudaMemcpy(answer, d_answer, sizeof(int)*patternNum, cudaMemcpyDeviceToHost);
  printf("Memcpy D->H answer : %s\n",cudaGetErrorString(r));


  printf("Array:\n");
  printf("%s\n", array);
  for(i=0;i<(patternSize+1)*patternNum;i++)
	  printf("%c", pattern[i]);
  printf("\n\n");
  for(i=0;i<patternNum;i++)
	printf("%d, %d\n", i, answer[i]);
  //printf("array : %c\n",array[3]);
  //for(i=0;i<patternSize*patternNum;i++)
  	//printf("%s\n",pattern);
  
  //for(i=0;i<patternNum;i++)
	//printf("%d  %s\n",i,pattern[i]);

  //KMP<<<blockNum,threadNum>>>(b, strlen(b), array, strlen(array));

  
  return 0;
}
