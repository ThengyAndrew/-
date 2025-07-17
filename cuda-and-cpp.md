# CUDA & cpp

### 参考资料

* [https://zhuanlan.zhihu.com/p/462822421](https://zhuanlan.zhihu.com/p/462822421)



#### 首个代码

{% code lineNumbers="true" %}
```cpp
// freshman.h头文件
#ifndef FRESHMAN_H
#define FRESHMAN_H
// 检查返回结果cudaError_t是否成功
#define CHECK(call)\   
{\
  const cudaError_t error=call;\
  if(error!=cudaSuccess)\
  {\
      printf("ERROR: %s:%d,",__FILE__,__LINE__);\
      printf("code:%d,reason:%s\n",error,cudaGetErrorString(error));\
      exit(1);\
  }\
}
#endif

// SumArrays.cu源代码文件
#include <cuda_runtime.h> // cuda相关头文件
#include <stdio.h>
#include "freshman.h"

void initialData(float* ip,int size)
{
  time_t t;
  srand((unsigned )time(&t));
  for(int i=0;i<size;i++)
  {
    ip[i]=(float)(rand()&0xffff)tf("\n");
  }
}

__global__ void sumArraysCuda(float*a,float*b,float*res)
{
  int i=threadIdx.x;
  res[i]=a[i]+b[i];
}
int main(int argc,char **argv)
{
  int dev = 0;
  cudaSetDevice(dev); // 当有多个GPU时，选定cuda设备，默认是0即第一个主GPU，多GPU时0,1,2以此类推

  int nElem=32;
  printf("Vector size:%d\n",nElem);
  int nByte=sizeof(float)*nElem;
  float *a_h=(float*)malloc(nByte);
  float *b_h=(float*)malloc(nByte);
  float *res_h=(float*)malloc(nByte);
  memset(res_h,0,nByte);


// 在 GPU 中给三个数组申请内存空间
  float *a_d,*b_d,*res_d;
  CHECK(cudaMalloc((float**)&a_d,nByte));
  CHECK(cudaMalloc((float**)&b_d,nByte));
  CHECK(cudaMalloc((float**)&res_d,nByte));

  initialData(a_h,nElem); 
  initialData(b_h,nElem);

  CHECK(cudaMemcpy(a_d,a_h,nByte,cudaMemcpyHostToDevice)); // 从主机内存拷贝到设备内存
  CHECK(cudaMemcpy(b_d,b_h,nByte,cudaMemcpyHostToDevice)); 

// 核心
  dim3 block(nElem); // 1 个 block 32 个 Thread
  dim3 grid(nElem/block.x); // 元素总数/单个 bl 的 Th 数 = 一个 grid 的 block 数
  sumArraysCuda<<<grid,block>>>(a_d,b_d,res_d);
  printf("Execution configuration<<<%d,%d>>>\n",block.x,grid.x);

  CHECK(cudaMemcpy(res_h,res_d,nByte,cudaMemcpyDeviceToHost)); // 将结果从设备内存拷贝到主机内存！
  cudaFree(a_d);
  cudaFree(b_d);
  cudaFree(res_d);

  free(a_h);
  free(b_h);
  free(res_h);

  return 0;
}\
```
{% endcode %}

{% hint style="info" %}
这里为什么自动进行了 32 个元素的运算?

1 block starts 32 Threads, and 1 id for 1 Thread.
{% endhint %}

#### grid-block-thread 的分划很像存储系统

| BlockIdx.x | ThreadIdx.x | Global idx |
| ---------- | ----------- | ---------- |
| 0          | 0 \~ 31     | 0 \~ 31    |
| 1          | 0 \~ 31     | 32 \~ 63   |
| 2          | 0 \~ 31     | 64 \~ 95   |
| 3          | 0 \~ 31     | 96 \~ 127  |



### GPU 的硬件（基于 Fermi 架构）

```
+-------------------------------------------------------------+
|                        GPU Device                           |
|                                                             |
|  +--------+   +--------+   ...   +--------+                 |
|  |   SM0  |   |   SM1  |         |   SM15 |  ← 多个 SM（流式多处理器）|
|  +--------+   +--------+         +--------+                 |
|     ↑            ↑                   ↑                      |
|   每个 SM 有多个 CUDA Core、寄存器、共享内存                 |
|                                                             |
|                      GigaThread Engine                      |
|                （调度线程块给 SM 的大管家）                 |
|                                                             |
|                  全局内存（global DRAM）                    |
+-------------------------------------------------------------+
```

#### <mark style="background-color:purple;">SM（Streaming Multiprocessor）流式多处理器</mark>

每个 SM 类似一个“微型并行处理器阵列”，拥有：

* 多个 <mark style="background-color:purple;">**CUDA Core（整数/浮点处理单元）**</mark>
  * 每个 SM 内部有多个 CUDA Core（例如 128 个）
  * 一个 CUDA Core 相当于处理一个 thread 的最小单位
  * 但是 CUDA 是以 32 个线程为单位调度（warp）
* 共享内存（block 内线程共享）
  * 每个 SM 自带的小型高速缓存（几 KB 到 100 KB）
  * block 内的线程可以共享访问
  * 访问速度比全局内存快得多
  * 适合线程间通信或缓存中间结果
* 寄存器组（线程私有变量）
* warp 调度器（调度 32 个线程一起执行）
* 一个线程块（block）被分配到一个 SM 上后，SM 内的硬件会管理其中所有线程的执行。

一个 GPU 通常有 16、48、80 甚至上百个 SM（如 A100 有 108 个 SM）



#### GigaThread Engine（超线程引擎）

这是一个 **全局调度器**，在 Fermi 架构中首次出现。

它的功能：

* 在 kernel 启动时，将线程块分配到合适的 SM 上
* 当某个 SM 空闲时，继续从待分配队列中拉取下一个 block
* 不管你有 1000 个 block，哪几个能同时执行，是由 GigaThread 决定的

👉 **GigaThread 不调度单个线程，只调度线程块（block）！**



#### DRAM（Global Memory，全局内存）

* 是 GPU 的显存（如 RTX 3060 有 12GB）
* 所有线程都可以访问，但**访问慢**
* 适合存放大数据，比如输入矩阵、输出结果

### 特性与技巧

#### 访存合并

pytorch 的模式:  <mark style="background-color:blue;">**row-major**</mark>

#### 向量化访存 vectorized

| 类型           | 对应位宽    |
| ------------ | ------- |
| `float2`     | 64-bit  |
| `float4`     | 128-bit |
| `int2`       | 64-bit  |
| `int4`       | 128-bit |
| `half2`      | 32-bit  |
| `__half2`    | 32-bit  |
| `bfloat162`  | 32-bit  |
| `bfloat168*` | 128-bit |
| `char4`      | 32-bit  |
| `uchar4`     | 32-bit  |
| `ushort2`    | 32-bit  |
| 自定义结构体       | 任意      |

示例：
{% code lineNumbers="true" %}
```cpp
__global__ void silu_coalesced_float4_kernel(const float* input, float* output, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col4 = blockIdx.x * blockDim.x + threadIdx.x; // col4 是逻辑上的列编号，每个单位对应一个 float4（4 个元素）
    int col = col4 << 2;

    if (row < rows && col + 3 < cols) {
        const float4* input_v4 = reinterpret_cast<const float4*>(input); // 通过 reinterpret_cast 把 float* 指针强转为 float4* 指针（每次访问 16 字节）
        float4 val = input_v4[row * (cols >> 2) + col4]; // 第 row 行、第 col4 个 float4 的偏移量是 row * (cols >> 2) + col4

        val.x = silu(val.x);
        val.y = silu(val.y);
        val.z = silu(val.z);
        val.w = silu(val.w);

        float4* output_v4 = reinterpret_cast<float4*>(output);
        output_v4[row * (cols >> 2) + col4] = val;
    }
}
```
{% endcode %}

#### SFU 单元加速

| 函数                 | 描述             |
| ------------------ | -------------- |
| `__sinf(x)`        | 快速计算 `sin(x)`  |
| `__cosf(x)`        | 快速计算 `cos(x)`  |
| `__expf(x)`        | 快速计算 `exp(x)`  |
| `__logf(x)`        | 快速计算 `log(x)`  |
| `__powf(x, y)`     | 快速计算 `x^y`     |
| `__sqrtf(x)`       | 快速计算 `sqrt(x)` |
| `__fdividef(a, b)` | 快速除法           |

