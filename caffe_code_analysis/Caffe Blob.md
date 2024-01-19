# Caffe Blob

## 1.   概述

* **Blob**：是基础的数据结构，用来保存学习到的参数以及网络传输过程中产生的数据。
* **Layer**：是网络的基本单元，由此派生出了各种层。
* **Net**：是网络的搭建，将 Layer 所派生出层类组合成网络。
* **Solver**：是 Net 的求解。



## 2.   Blob

### 2.1     源码文件

* include/blob.hpp
* src/blob.cpp



### 2.2     整体概览

```c++
template <typename Dtype>
class Blob {
public:
  Blob()
       : data_(), diff_(), count_(0), capacity_(0) {}

  /// @brief Deprecated; use <code>Blob(const vector<int>& shape)</code>.
  explicit Blob(const int num, const int channels, const int height,
      const int width);
  explicit Blob(const vector<int>& shape);

  /// @brief Deprecated; use <code>Reshape(const vector<int>& shape)</code>.
  void Reshape(const int num, const int channels, const int height,
      const int width);
  
  ......
    
protected:
  shared_ptr<SyncedMemory> data_;
  shared_ptr<SyncedMemory> diff_;
  shared_ptr<SyncedMemory> shape_data_;
  vector<int> shape_;
  int count_;
  int capacity_;

  DISABLE_COPY_AND_ASSIGN(Blob);
  /* private: Blob(const Blob&); Blob& operator=(const Blob&) */
};  // class Blob
```



### 2.3     成员变量

```c++
shared_ptr<SyncedMemory> data_;
shared_ptr<SyncedMemory> diff_;
shared_ptr<SyncedMemory> shape_data_;
vector<int> shape_;
int count_;
int capacity_;
```

* Blob只是个基本的数据结构，因此内部的变量相对较少
* data_ 指针：指针类型是 shared_ptr，属于boost库的一个智能指针，这一部分主要用来**申请内存存储data，data主要是正向传播的时候用的**。

* diff_指针：主要用来存储偏差，update data
* shape_data _ ，shape _ ：都是存储Blob的形状
* count_：表示Blob中的元素个数，也就是  个数 * 通道数 * 高度 * 宽度  batch_size * channels * hight * width
* capacity_:表示当前的元素个数，因为Blob可能会 reshape





### 2.4     成员函数

```
Blob():data_(), diff_(), count_(0), capacity_(0) {}
/// @brief 反对使用 <code>Blob(const vector<int>& shape)</code>.
explicit Blob(const int num, const int channels, const int height,const int width);
explicit Blob(const vector<int> &shape);
```

* **Blob**：作为一个最基础的类，其中构造函数开辟一个内存空间来存储数据



```c++
void Reshape(const int num, const int channels, const int height, const int width);
/* Reshape具体实现 */
void Reshape(const vector<int> &shape);
void Reshape(const BlobShape &shape);
void ReshapeLike(const Blob &other);


template <typename Dtype>
void Blob<Dtype>::Reshape(const vector<int> &shape)
{
    /* 检查shape的大小是否小于等于最大轴数kMaxBlobAxes */
    CHECK_LE(shape.size(), kMaxBlobAxes);
    /* 初始化count_为1，用于计算Blob的总元素数量 */
    count_ = 1;
    /* 调整shape_的大小，确保它能够存储新的shape信息 */
    shape_.resize(shape.size());
    /* 如果shape_data_不存在或者其大小不足以存储新的shape信息，则创建一个新的SyncedMemory对象 */
    if (!shape_data_ || shape_data_->size() < shape.size() * sizeof(int))
    {
        shape_data_.reset(new SyncedMemory(shape.size() * sizeof(int)));
    }
    /* 获取shape_data_的cpu_data指针，用于写入新的shape信息 */
    int *shape_data = static_cast<int *>(shape_data_->mutable_cpu_data());
    /* 遍历shape的每个维度 */
    for (int i = 0; i < shape.size(); ++i)
    {
        /* 检查每个维度的大小是否大于等于0 */
        CHECK_GE(shape[i], 0);
        /* 如果count_不为0，检查当前维度的大小是否不会导致整数溢出 即最终count_的大小是否超出范围 */
        if (count_ != 0)
        {
            CHECK_LE(shape[i], INT_MAX / count_) << "blob size exceeds INT_MAX";
        }
        /* 更新count_，乘以当前维度的大小 */
        count_ *= shape[i];
        /* 将新的维度大小赋值给shape_ */
        shape_[i] = shape[i];
        /* 将维度大小写入shape_data_ */
        shape_data[i] = shape[i];
    }

    /* 如果count_超过了当前的容量capacity_，则重新分配数据内存 */
    if (count_ > capacity_)
    {
        capacity_ = count_;
        /* 创建新的SyncedMemory对象用于存储data和diff（梯度） */
        data_.reset(new SyncedMemory(capacity_ * sizeof(Dtype)));
        diff_.reset(new SyncedMemory(capacity_ * sizeof(Dtype)));
    }
}
```

* **Reshape**：此函数既可以用于创建内存的初始分配，也可以用于在Layer::Reshape或Layer::Forward期间调整顶部blob的尺寸。当更改blob的大小时，只有在没有足够内存的情况下才会重新分配内存，并且永远不会释放多余的内存。



```
inline int count() const { return count_; }
inline int count(int start_axis, int end_axis) const
inline int count(int start_axis) const
```

* **count**：Blob类里面有重载很多个count()函数，主要还是为了统计Blob的容量（volume），或者是某一片（slice），从某个startAxis到具体某个endAxis的shape乘积。     [ startAxis ,  endAxis )



```c++
inline int CanonicalAxisIndex(int axis_index)

/* 返回索引第index个轴的维度（如果索引为负数，
   则返回从末尾算起的负索引第index个轴线的维度）。 */
inline int shape(int index) const
{
	return shape_[CanonicalAxisIndex(index)];
}
```

* **CanonicalAxisIndex**：返回（通常）用户指定轴的“规范”版本，允许负索引（例如，最后一个轴为-1），如果超出索引将崩溃。
  * 对于Blob中的4个基本变量num,channel,height,width
  * 可以直接通过shape( 0 ),shape(1 ),shape(2 ),shape(3 )来访问。
  * ​	            或者shape(-4),shape(-3),shape(-2),shape(-1)



```
inline int offset(const int n, const int c = 0, const int h = 0, const int w = 0) const
inline int offset(const vector<int> &indices) const
```

* **offset**：offset计算的方式也支持两种方式，一种直接指定n,c,h,w或者放到一个vector中进行计算，偏差是根据对应的n,c,h,w，返回的offset是 `((n * channels() + c) * height() + h) * width() + w`



```
/* Copy from a source Blob. */
void CopyFrom(const Blob<Dtype> &source, bool copy_diff = false, bool reshape = false);
```

* **CopyFrom**：从一个blob中copy数据，通过开关控制是否 copy_diff，如果是False则copy data，True则copy diff。reshape控制是否需要reshape



```c++
inline Dtype data_at(const int n, const int c, const int h, const int w) const
{
	return cpu_data()[offset(n, c, h, w)];
}

inline Dtype diff_at(const int n, const int c, const int h, const int w) const
    
inline Dtype data_at(const vector<int> &index) const
{
	return cpu_data()[offset(index)];
}

inline Dtype diff_at(const vector<int> &index) const
    
inline const shared_ptr<SyncedMemory> &data() const
inline const shared_ptr<SyncedMemory> &diff() const
```

* **at**：这一部分函数主要通过给定的位置访问数据，根据位置计算与数据起始的偏差offset，再通过cpu_data* 指针获得地址。



```C++
const Dtype *cpu_data() const;
void set_cpu_data(Dtype *data);
const int *gpu_shape() const;
const Dtype *gpu_data() const;
void set_gpu_data(Dtype *data);
const Dtype *cpu_diff() const;
const Dtype *gpu_diff() const;
Dtype *mutable_cpu_data();
Dtype *mutable_gpu_data();
Dtype *mutable_cpu_diff();
Dtype *mutable_gpu_diff();
```

* **data**：设置和获取data
  * **data**：存储前向传播的数据
  * **diff**：存储反向传播的梯度



```c++
template <typename Dtype>
void Blob<Dtype>::Update()
{
    // We will perform update based on where the data is located.
    /* 根据数据所在的位置进行更新 */
    switch (data_->head())
    {
    case SyncedMemory::HEAD_AT_CPU:
      // perform computation on CPU
      /**
       * 调用caffe_axpy函数，它在CPU上执行向量减法。
       * 这里将差异数据（diff_）乘以-1，然后与原始数据（data_）相加，实现参数更新。
      */
      caffe_axpy<Dtype>(count_, Dtype(-1),
                        static_cast<const Dtype *>(diff_->cpu_data()),
                        static_cast<Dtype *>(data_->mutable_cpu_data()));
      break;
    case SyncedMemory::HEAD_AT_GPU:
    case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
      // perform computation on GPU
      caffe_gpu_axpy<Dtype>(count_, Dtype(-1),
                            static_cast<const Dtype *>(diff_->gpu_data()),
                            static_cast<Dtype *>(data_->mutable_gpu_data()));
#else
      NO_GPU;
#endif
      break;
    default:
      LOG(FATAL) << "Syncedmem not initialized.";
    }
}
```

* **update**：这个函数在 caffe 的 util 下面的match-functions.cpp里面，主要是负责了线性代数库的调用，实现的功能是`Y = alpha * X + beta * Y`。即 blob 里面的data部分减去diff部分



```
void FromProto(const BlobProto &proto, bool reshape = true);
void ToProto(BlobProto *proto, bool write_diff = false) const;
```

* **Proto**：这两个函数主要是将数据序列化，存储到BlobProto，这里说到Proto是谷歌的一个数据序列化的存储格式，可以实现语言、平台无关、可扩展的序列化结构数据结构。



```c++
/// @brief Compute the sum of absolute values (L1 norm) of the data.
Dtype asum_data() const;
/// @brief Compute the sum of absolute values (L1 norm) of the diff.
Dtype asum_diff() const;
/// @brief Compute the sum of squares (L2 norm squared) of the data.
Dtype sumsq_data() const;
/// @brief Compute the sum of squares (L2 norm squared) of the diff.
Dtype sumsq_diff() const;

/// @brief Scale the blob data by a constant factor.
void scale_data(Dtype scale_factor);
/// @brief Scale the blob diff by a constant factor.
void scale_diff(Dtype scale_factor);
```

* **asum**：表示L1范数，**sumsq**：表示求L2范数，**scala**：表示乘法，乘以一个因子
  * **L0范数**：向量中非0的元素个数
  * **L1范数**：向量中各个元素的绝对值之和
  * **L2范数**：向量中各个元素的平方和然后求平方根，**L2范数可以防止过拟合，提升模型的泛化能力**

* 扩展

  * 对于p-范数，如果 $x = [x_{1}, x_{2}, \dots, x_{n}]^T$，

    * 那么向量x的p-范数就是 $||X||_{p} = (|x_{1}|^{p} + |x_{2}|^{p} + \dots + |x_{n}|^{p})^{\frac{1}{p}}$
    * L1范数 $||X||_{1} = (|x_{1}| + |x_{2}| + \dots + |x_{n}|)$
    * L2范数 $||X||_{2} = (|x_{1}|^{2} + |x_{2}|^{2} + \dots + |x_{n}|^{2})^{\frac{1}{2}}$
    * 特别的，L0范数：指向量中非零元素的个数。无穷范数：指向量中所有元素的最大绝对值
    
    

```c++
/**
* @brief Set the data_ shared_ptr to point to the SyncedMemory holding the
*        data_ of Blob other -- useful in Layer%s which simply perform a copy
*        in their Forward pass.
*        将data_的shared_ptr设置为指向SyncedMemory，
*        该SyncedMemory保存Blob other的data_——在层%s中很有用，后者只是在其Forward过程中执行复制
*
* This deallocates the SyncedMemory holding this Blob's data_, as
* shared_ptr calls its destructor when reset with the "=" operator.
* 这将释放保存此Blob数据的SyncedMemory，
* 因为shared_ptr在使用“=”运算符重置时调用其析构函数。
*/
void ShareData(const Blob &other);
/**
* @brief Set the diff_ shared_ptr to point to the SyncedMemory holding the
*        diff_ of Blob other -- useful in Layer%s which simply perform a copy
*        in their Forward pass.
*
* This deallocates the SyncedMemory holding this Blob's diff_, as
* shared_ptr calls its destructor when reset with the "=" operator.
*/
void ShareDiff(const Blob &other);
```

* **Share**：一个是共享data，一个是共享diff，具体就是将别的blob的data和diff指针给这个Blob，实现数据的共享。同时需要注意的是这个操作会引起这个Blob里面的SyncedMemory被释放，因为shared_ptr指针被用 **=** 重置的时候会调用原本数据的析构器。



```
/**
* @brief 比较两个Blob形状是否相同
*/
bool ShapeEquals(const BlobProto &other);
```

* **ShapeEquals**：比较两个Blob形状是否相同















