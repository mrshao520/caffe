#ifndef CAFFE_BLOB_HPP_
#define CAFFE_BLOB_HPP_

#include <algorithm>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/syncedmem.hpp"

const int kMaxBlobAxes = 32;

namespace caffe
{

  /**
   * @brief A wrapper around SyncedMemory holders serving as the basic
   *        computational unit through which Layer%s, Net%s, and Solver%s
   *        interact.
   *        Blob 是 Caffe 中的基础数据结构，用于存储神经网络的权重、偏置、激活值等
   *
   * TODO(dox): more thorough description.
   */
  template <typename Dtype>
  class Blob
  {
  public:
    Blob()
        : data_(), diff_(), count_(0), capacity_(0) {}

    /// @brief Deprecated; use <code>Blob(const vector<int>& shape)</code>.
    explicit Blob(const int num, const int channels, const int height,
                  const int width);
    explicit Blob(const vector<int> &shape);

    /// @brief Deprecated; use <code>Reshape(const vector<int>& shape)</code>.
    void Reshape(const int num, const int channels, const int height,
                 const int width);
    /**
     * @brief Change the dimensions of the blob, allocating new memory if
     *        necessary.
     *
     * This function can be called both to create an initial allocation
     * of memory, and to adjust the dimensions of a top blob during Layer::Reshape
     * or Layer::Forward. When changing the size of blob, memory will only be
     * reallocated if sufficient memory does not already exist, and excess memory
     * will never be freed.
     *
     * 此函数既可以用于创建内存的初始分配，也可以用于在Layer::Reshape或Layer::Forward期间调整顶部blob的尺寸。
     * 当更改blob的大小时，只有在没有足够内存的情况下才会重新分配内存，并且永远不会释放多余的内存。
     *
     * Note that reshaping an input blob and immediately calling Net::Backward is
     * an error; either Net::Forward or Net::Reshape need to be called to
     * propagate the new input shape to higher layers.
     */
    void Reshape(const vector<int> &shape);
    void Reshape(const BlobShape &shape);
    void ReshapeLike(const Blob &other);

    /**
     * @brief 返回维度和count_组成的字符串
     */
    inline string shape_string() const
    {
      ostringstream stream;
      for (int i = 0; i < shape_.size(); ++i)
      {
        stream << shape_[i] << " ";
      }
      stream << "(" << count_ << ")";
      return stream.str();
    }
    inline const vector<int> &shape() const { return shape_; }

    /**
     * @brief Returns the dimension of the index-th axis (or the negative index-th
     *        axis from the end, if index is negative).
     *        返回索引第index个轴的维度（如果索引为负数，则返回从末尾算起的负索引第index个轴线的维度）。
     *
     * @param index the axis index, which may be negative as it will be
     *        "canonicalized" using CanonicalAxisIndex.
     *        Dies on out of range index.
     */
    inline int shape(int index) const
    {
      return shape_[CanonicalAxisIndex(index)];
    }

    /**
     * @brief 返回存储Blob的形状shape_的长度
     */
    inline int num_axes() const { return shape_.size(); }

    /**
     * @brief 返回Blob中的元素个数count_，batch_size * channels * hight * width
     */
    inline int count() const { return count_; }

    /**
     * @brief Compute the volume of a slice; i.e., the product of dimensions
     *        among a range of axes.
     *
     * @param start_axis The first axis to include in the slice.
     *
     * @param end_axis The first axis to exclude from the slice.
     *
     * range in [start_axis, end_axis)
     */
    inline int count(int start_axis, int end_axis) const
    {
      CHECK_LE(start_axis, end_axis);
      CHECK_GE(start_axis, 0);
      CHECK_GE(end_axis, 0);
      CHECK_LE(start_axis, num_axes());
      CHECK_LE(end_axis, num_axes());
      int count = 1;
      for (int i = start_axis; i < end_axis; ++i)
      {
        count *= shape(i);
      }
      return count;
    }
    /**
     * @brief Compute the volume of a slice spanning from a particular first
     *        axis to the final axis.
     *
     * @param start_axis The first axis to include in the slice.
     */
    inline int count(int start_axis) const
    {
      return count(start_axis, num_axes());
    }

    /**
     * @brief Returns the 'canonical' version of a (usually) user-specified axis,
     *        allowing for negative indexing (e.g., -1 for the last axis).
     *        返回（通常）用户指定轴的“规范”版本，允许负索引（例如，最后一个轴为-1）。
     *
     * @param axis_index the axis index.
     *        If 0 <= index < num_axes(), return index.
     *        If -num_axes <= index <= -1, return (num_axes() - (-index)),
     *        e.g., the last axis index (num_axes() - 1) if index == -1,
     *        the second to last if index == -2, etc.
     *        Dies on out of range index. 超出索引将崩溃
     */
    inline int CanonicalAxisIndex(int axis_index) const
    {
      /* 用于检查axis_index是否大于或等于-num_axes()。
         如果条件不成立，将抛出一个错误，错误信息指出
         axis_index超出了范围，并提供了当前Blob的形状信息。 */
      CHECK_GE(axis_index, -num_axes())
          << "axis " << axis_index << " out of range for " << num_axes()
          << "-D Blob with shape " << shape_string();
      CHECK_LT(axis_index, num_axes())
          << "axis " << axis_index << " out of range for " << num_axes()
          << "-D Blob with shape " << shape_string();
      if (axis_index < 0)
      {
        return axis_index + num_axes();
      }
      return axis_index;
    }

    /// @brief Deprecated legacy shape accessor num: use shape(0) instead.
    inline int num() const { return LegacyShape(0); }
    /// @brief Deprecated legacy shape accessor channels: use shape(1) instead.
    inline int channels() const { return LegacyShape(1); }
    /// @brief Deprecated legacy shape accessor height: use shape(2) instead.
    inline int height() const { return LegacyShape(2); }
    /// @brief Deprecated legacy shape accessor width: use shape(3) instead.
    inline int width() const { return LegacyShape(3); }

    inline int LegacyShape(int index) const
    {
      CHECK_LE(num_axes(), 4)
          << "Cannot use legacy accessors on Blobs with > 4 axes.";
      CHECK_LT(index, 4);
      CHECK_GE(index, -4);
      if (index >= num_axes() || index < -num_axes())
      {
        // Axis is out of range, but still in [0, 3] (or [-4, -1] for reverse
        // indexing) -- this special case simulates the one-padding used to fill
        // extraneous axes of legacy blobs.
        return 1;
      }
      return shape(index);
    }

    /**
     * @brief 根据基本变量num, channels, height, width 计算偏移量
     */
    inline int offset(const int n, const int c = 0, const int h = 0,
                      const int w = 0) const
    {
      CHECK_GE(n, 0);
      CHECK_LE(n, num());
      CHECK_GE(channels(), 0);
      CHECK_LE(c, channels());
      CHECK_GE(height(), 0);
      CHECK_LE(h, height());
      CHECK_GE(width(), 0);
      CHECK_LE(w, width());
      return ((n * channels() + c) * height() + h) * width() + w;
    }

    inline int offset(const vector<int> &indices) const
    {
      CHECK_LE(indices.size(), num_axes());
      int offset = 0;
      for (int i = 0; i < num_axes(); ++i)
      {
        offset *= shape(i);
        if (indices.size() > i)
        {
          CHECK_GE(indices[i], 0);
          CHECK_LT(indices[i], shape(i));
          offset += indices[i];
        }
      }
      return offset;
    }
    /**
     * @brief Copy from a source Blob.
     *
     * @param source the Blob to copy from
     * @param copy_diff if false, copy the data; if true, copy the diff
     * @param reshape if false, require this Blob to be pre-shaped to the shape
     *        of other (and die otherwise); if true, Reshape this Blob to other's
     *        shape if necessary
     */
    void CopyFrom(const Blob<Dtype> &source, bool copy_diff = false,
                  bool reshape = false);

    inline Dtype data_at(const int n, const int c, const int h,
                         const int w) const
    {
      return cpu_data()[offset(n, c, h, w)];
    }

    inline Dtype diff_at(const int n, const int c, const int h,
                         const int w) const
    {
      return cpu_diff()[offset(n, c, h, w)];
    }

    inline Dtype data_at(const vector<int> &index) const
    {
      return cpu_data()[offset(index)];
    }

    inline Dtype diff_at(const vector<int> &index) const
    {
      return cpu_diff()[offset(index)];
    }

    inline const shared_ptr<SyncedMemory> &data() const
    {
      CHECK(data_);
      return data_;
    }

    inline const shared_ptr<SyncedMemory> &diff() const
    {
      CHECK(diff_);
      return diff_;
    }

    /**
     * @brief 设置和获取data
     */
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

    /**
     * @brief 更新blob对象中的数据
     *        Y = alpha * X + beta * Y blob里面的data部分减去diff部分
     */
    void Update();

    void FromProto(const BlobProto &proto, bool reshape = true);
    void ToProto(BlobProto *proto, bool write_diff = false) const;

    /// @brief Compute the sum of absolute values (L1 norm) of the data.
    ///        计算L1范数：向量中各个元素的绝对值之和
    Dtype asum_data() const;
    /// @brief Compute the sum of absolute values (L1 norm) of the diff.
    Dtype asum_diff() const;
    /// @brief Compute the sum of squares (L2 norm squared) of the data.
    ///        计算L2范数：向量中各个元素的平方和然后求平方根，L2范数可以防止过拟合，提升模型的泛化能力
    Dtype sumsq_data() const;
    /// @brief Compute the sum of squares (L2 norm squared) of the diff.
    Dtype sumsq_diff() const;

    /**
     * @brief 将Blob中的data和diff乘以一个常数因子scale_factor
     *        这在神经网络训练过程中非常有用，例如在更新权重时按学习率缩放梯度，
     *        或者在实现某些算法时对数据进行标准化。
     * @param scale_factor 常数因子
     */
    /// @brief Scale the blob data by a constant factor.
    void scale_data(Dtype scale_factor);
    /// @brief Scale the blob diff by a constant factor.
    void scale_diff(Dtype scale_factor);

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

    /**
     * @brief 比较两个Blob形状是否相同
     */
    bool ShapeEquals(const BlobProto &other);

  protected:
    shared_ptr<SyncedMemory> data_;       /* 主要用来正向传播的时候用 */
    shared_ptr<SyncedMemory> diff_;       /* 存储偏差 */
    shared_ptr<SyncedMemory> shape_data_; /* 存储Blob的形状 */
    vector<int> shape_;                   /* 存储Blob的形状 */
    int count_;                           /* Blob中的元素个数，batch_size * channels * hight * width */
    int capacity_;                        /* 当前元素的个数,或者data_.size()  capacity_会默认初始化为0，且在其他操作之前*/

    /* 禁止拷贝和赋值构造 */
    DISABLE_COPY_AND_ASSIGN(Blob);
    /* private: Blob(const Blob&); Blob& operator=(const Blob&) */

  }; // class Blob

} // namespace caffe

#endif // CAFFE_BLOB_HPP_
