#ifndef CAFFE_SYNCEDMEM_HPP_
#define CAFFE_SYNCEDMEM_HPP_

#include <cstdlib>

#ifdef USE_MKL
#include "mkl.h"
#endif

#include "caffe/common.hpp"

namespace caffe
{

  // If CUDA is available and in GPU mode, host memory will be allocated pinned,
  // using cudaMallocHost. It avoids dynamic pinning for transfers (DMA).
  // The improvement in performance seems negligible in the single GPU case,
  // but might be more significant for parallel training. Most importantly,
  // it improved stability for large models on many GPUs.
  // 如果CUDA可用且处于GPU模式，则将使用cudaMallocHost固定分配主机内存。
  // 它避免了动态固定传输（DMA）。在单个GPU的情况下，性能的提高似乎可以忽略不计，
  // 但对于并行训练来说可能更重要。最重要的是，它提高了许多GPU上大型模型的稳定性。
  /**
   * @brief 在主机内存上分配空间，并根据情况设置use_cuda的值
   * @param ptr        指向指针的指针
   * @param size       内存大小
   * @param use_cuda   是否使用GPU
   */
  inline void CaffeMallocHost(void **ptr, size_t size, bool *use_cuda)
  {
#ifndef CPU_ONLY
    if (Caffe::mode() == Caffe::GPU)
    {
      /* 使用cudaMallocHost来分配页面锁定（pinned）内存,
        分配的内存是可供 GPU 直接访问的，提高CPU和GPU之间的内存传输速度 */
      CUDA_CHECK(cudaMallocHost(ptr, size));
      *use_cuda = true;
      return;
    }
#endif
#ifdef USE_MKL
    /* 使用MKL（Intel Math Kernel Library）的mkl_malloc来分配内存 */
    *ptr = mkl_malloc(size ? size : 1, 64);
#else
    *ptr = malloc(size);
#endif
    *use_cuda = false;
    /* 使用CHECK宏来检查分配的指针是否为空，如果为空，则输出错误信息并终止程序。 */
    CHECK(*ptr) << "host allocation of size " << size << " failed";
  }

  /**
   * @brief 释放主机上的内存
   * @param ptr        指向指针的指针
   * @param use_cuda   是否使用GPU
   */
  inline void CaffeFreeHost(void *ptr, bool use_cuda)
  {
#ifndef CPU_ONLY
    if (use_cuda)
    {
      CUDA_CHECK(cudaFreeHost(ptr));
      return;
    }
#endif
#ifdef USE_MKL
    mkl_free(ptr);
#else
    free(ptr);
#endif
  }

  /**
   * @brief Manages memory allocation and synchronization between the host (CPU)
   *        and device (GPU).
   *        SyncedMemory类是Caffe框架中的一个核心类，用于管理在CPU和GPU之间同步的数据。
   *        这个类的目的是为了高效地在CPU和GPU之间传输数据，同时避免不必要的复制操作。
   *        提供了对数据的同步访问，确保在读取数据时，数据是在正确的设备上初始化的
   *
   * TODO(dox): more thorough description.
   */
  class SyncedMemory
  {
  public:
    SyncedMemory();                     // 默认构造函数
    explicit SyncedMemory(size_t size); // 构造函数，用于指定大小的分配
    ~SyncedMemory();                    // 析构函数，用于释放内存
    const void *cpu_data();             // 返回 CPU 上数据的只读指针
    void set_cpu_data(void *data);      // 设置 CPU 上的数据指针
    const void *gpu_data();             // 返回 GPU 上数据的只读指针
    void set_gpu_data(void *data);      // 设置 GPU 上的数据指针
    void *mutable_cpu_data();           // 返回 CPU 上数据的可写指针
    void *mutable_gpu_data();           // 返回 GPU 上数据的可写指针
    enum SyncedHead                     // 表示数据的同步状态
    {
      UNINITIALIZED, // 未初始化
      HEAD_AT_CPU,   // 在主机CPU上的数据是最新的
      HEAD_AT_GPU,   // 在设备GPU上的数据是最新的
      SYNCED         // CPU和GPU上已经同步
    };
    SyncedHead head() const { return head_; } // 返回当前数据的同步状态
    size_t size() const { return size_; }     // 返回分配的内存大小

#ifndef CPU_ONLY
    /**
     * @brief 异步地将数据从 CPU 推送到 GPU。
     *        这个函数是异步的，意味着它会在指定的 CUDA 流上启动数据传输，
     *        但不会等待传输完成就返回，从而允许重叠的数据传输和计算。
    */
    void async_gpu_push(const cudaStream_t &stream); // 异步将数据从 CPU 推送到 GPU（非 CPU_ONLY 模式下可用）
#endif

  private:
    void check_device(); // 检查当前设备是否与数据所在的设备匹配

    /**
     * @brief 将数据从 GPU 复制到 CPU。这个函数确保了数据在 CPU 上是最新的，
     *        并且根据当前的同步状态（head_）来决定是否需要执行复制操作。
     */
    void to_cpu(); // 将数据从 GPU 复制到 CPU

    /**
     * @brief 将数据从 CPU 复制到 GPU。这个函数确保了数据在 GPU 上是最新的，
     *        并且根据当前的同步状态（head_）来决定是否需要执行复制操作
     */
    void to_gpu(); // 将数据从 CPU 复制到 GPU

    void *cpu_ptr_;            // 指向 CPU 上数据的指针
    void *gpu_ptr_;            // 指向 GPU 上数据的指针
    size_t size_;              // 分配的内存大小
    SyncedHead head_;          // 当前数据的同步状态
    bool own_cpu_data_;        // 指示是否拥有 CPU 数据的所有权，
                               // 当一个syncedmemory类实例共享其他实例的资源时，其own_cpu_data_为false，在析构时不需要释放资源
    bool cpu_malloc_use_cuda_; // 指示是否使用 CUDA 分配 CPU 内存
    bool own_gpu_data_;        // 指示是否拥有 GPU 数据的所有权，同上
    int device_;               // 当前设备 ID，仅在DEBUG模式下有用

    DISABLE_COPY_AND_ASSIGN(SyncedMemory);
    // private:
    //      SyncedMemory(const SyncedMemory&);
    //      SyncedMemory& operator=(const SyncedMemory&)

  }; // class SyncedMemory

} // namespace caffe

#endif // CAFFE_SYNCEDMEM_HPP_
