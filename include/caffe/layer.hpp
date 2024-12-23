#ifndef CAFFE_LAYER_H_
#define CAFFE_LAYER_H_

#include <algorithm>
#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/math_functions.hpp"

/**
 Forward declare boost::thread instead of including boost/thread.hpp
 to avoid a boost/NVCC issues (#1009, #1010) on OSX.
 */
namespace boost
{
  class mutex;
}

namespace caffe
{

  /**
   * @brief An interface for the units of computation which can be composed into a
   *        Net.计算单元的接口，可以组成一个网络。
   *
   * Layer%s must implement a Forward function, in which they take their input
   * (bottom) Blob%s (if any) and compute their output Blob%s (if any).
   * They may also implement a Backward function, in which they compute the error
   * gradients with respect to their input Blob%s, given the error gradients with
   * their output Blob%s.
   * 层%s必须实现Forward函数，在该函数中，它们获取输入（bottom）Blob%s（如果有）并计算输出（top）Blob%s（如有）。
   * 它们还可以实现Backward函数，在该函数中，它们计算相对于其输入Blob%s的误差梯度，给定其输出Blob%s的误差梯度
   */
  template <typename Dtype>
  class Layer
  {
  public:
    /**
     * You should not implement your own constructor. Any set up code should go
     * to SetUp(), where the dimensions of the bottom blobs are provided to the
     * layer.
     */
    explicit Layer(const LayerParameter &param)
        : layer_param_(param)
    {
      // Set phase and copy blobs (if there are any).
      phase_ = param.phase();
      if (layer_param_.blobs_size() > 0)
      {
        blobs_.resize(layer_param_.blobs_size());
        for (int i = 0; i < layer_param_.blobs_size(); ++i)
        {
          blobs_[i].reset(new Blob<Dtype>());          ///< 创建一个新的Blob对象
          blobs_[i]->FromProto(layer_param_.blobs(i)); ///< 从protobuf消息中读取数据并初始化Blob
        }
      }
    }
    virtual ~Layer() {}

    /**
     * @brief Implements common layer setup functionality.
     *
     * @param bottom the preshaped input blobs 输入数据,输入形状固定
     * @param top
     *     the allocated but unshaped output blobs, to be shaped by Reshape 输出数据，形状未固定
     *
     * Checks that the number of bottom and top blobs is correct.
     * Calls LayerSetUp to do special layer setup for individual layer types,
     * followed by Reshape to set up sizes of top blobs and internal buffers.
     * Sets up the loss weight multiplier blobs for any non-zero loss weights.
     * This method may not be overridden.
     * 首先check 这个bottom和top的blob是否正确，再调用Layersetup对每一具体的层做进一步设置，
     * 之后再做reshape来设置top blobs和internal buffer。
     * 最后再设置loss weight multiplier 的blob对每一个非零的loss和weight。
     * 一般这个方法被继承之后是不会被重写的。
     */
    void SetUp(const vector<Blob<Dtype> *> &bottom,
               const vector<Blob<Dtype> *> &top)
    {
      /// 这个函数用于检查输入输出blob的数量是否与层的期望匹配。这有助于确保层的输入输出配置是正确的
      CheckBlobCounts(bottom, top);
      /// 这个虚函数是层特定的设置，它在层的子类中实现。它用于执行层的初始化操作，例如设置层的参数、初始化权重和偏置等。
      LayerSetUp(bottom, top);
      /// 这个函数用于根据输入blob的形状来调整输出blob的形状。
      /// 在神经网络中，不同层的输出形状可能不同，因此在层的设置过程中需要根据输入来调整输出形状。
      Reshape(bottom, top);
      /// 这个函数用于设置输出blob在损失函数中的权重。
      SetLossWeights(top);
    }

    /**
     * @brief Does layer-specific setup: your layer should implement this function
     *        as well as Reshape.
     *        进行特定层的设置：您的层应该实现此功能。
     *
     * @param bottom
     *     the preshaped input blobs, whose data fields store the input data for
     *     this layer
     * @param top
     *     the allocated but unshaped output blobs
     *
     * This method should do one-time layer specific setup. This includes reading
     * and processing relevent parameters from the <code>layer_param_</code>.
     * Setting up the shapes of top blobs and internal buffers should be done in
     * <code>Reshape</code>, which will be called before the forward pass to
     * adjust the top blob sizes.
     * 此方法应进行一次性特定于层的设置。这包括从<code>layer_param_</code>读取和处理相关参数。
     * 设置top blob和内部缓冲区的形状应在<code>Reshape</code>中完成，
     * 该函数将在正向传递之前调用以调整top blob的大小。
     */
    virtual void LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                            const vector<Blob<Dtype> *> &top) {}

    /**
     * @brief Adjust the shapes of top blobs and internal buffers to accommodate
     *        the shapes of the bottom blobs.
     *        调整top blobs和内部缓冲区的形状，以适应bottom blobs的形状。
     *
     * @param bottom the input blobs, with the requested input shapes
     * @param top the top blobs, which should be reshaped as needed
     *
     * This method should reshape top blobs as needed according to the shapes
     * of the bottom (input) blobs, as well as reshaping any internal buffers
     * and making any other necessary adjustments so that the layer can
     * accommodate the bottom blobs.
     */
    virtual void Reshape(const vector<Blob<Dtype> *> &bottom,
                         const vector<Blob<Dtype> *> &top) = 0;

    /**
     * @brief Given the bottom blobs, compute the top blobs and the loss.
     *        前向传播
     *
     * @param bottom
     *     the input blobs, whose data fields store the input data for this layer
     * @param top
     *     the preshaped output blobs, whose data fields will store this layers'
     *     outputs
     * \return The total loss from the layer.
     *
     * The Forward wrapper calls the relevant device wrapper function
     * (Forward_cpu or Forward_gpu) to compute the top blob values given the
     * bottom blobs.  If the layer has any non-zero loss_weights, the wrapper
     * then computes and returns the loss.
     * Forward包装器调用相关的设备包装器函数（Forward_cpu或Forward_gpu）
     * 来计算给定bottom blob的top blob值。
     * 如果该层有任何非零的loss_weights，则包装器会计算并返回损失。
     *
     * Your layer should implement Forward_cpu and (optionally) Forward_gpu.
     */
    inline Dtype Forward(const vector<Blob<Dtype> *> &bottom,
                         const vector<Blob<Dtype> *> &top);

    /**
     * @brief Given the top blob error gradients, compute the bottom blob error
     *        gradients.
     *        反向传播
     *
     * @param top
     *     the output blobs, whose diff fields store the gradient of the error
     *     with respect to themselves
     * @param propagate_down
     *     a vector with equal length to bottom, with each index indicating
     *     whether to propagate the error gradients down to the bottom blob at
     *     the corresponding index
     *     与bottom blob相等的向量，每个索引指示是否将误差梯度向下传播到相应索引处的bottom blob
     * @param bottom
     *     the input blobs, whose diff fields will store the gradient of the error
     *     with respect to themselves after Backward is run
     *
     * The Backward wrapper calls the relevant device wrapper function
     * (Backward_cpu or Backward_gpu) to compute the bottom blob diffs given the
     * top blob diffs.
     * Backward包装器调用相关的设备包装器函数（Backward_cpu或Backward_gpu）
     * 来计算给定top blob diffs 的bottom blob diffs。
     *
     * Your layer should implement Backward_cpu and (optionally) Backward_gpu.
     */
    inline void Backward(const vector<Blob<Dtype> *> &top,
                         const vector<bool> &propagate_down,
                         const vector<Blob<Dtype> *> &bottom);

    /**
     * @brief Returns the vector of learnable parameter blobs.
     *        返回可学习参数的blobs
     */
    vector<shared_ptr<Blob<Dtype> > > &blobs()
    {
      return blobs_;
    }

    /**
     * @brief Returns the layer parameter.
     *        返回层的参数
     */
    const LayerParameter &layer_param() const { return layer_param_; }

    /**
     * @brief Writes the layer parameter to a protocol buffer
     *        将层的参数写入protocol buffer里
     */
    virtual void ToProto(LayerParameter *param, bool write_diff = false);

    /**
     * @brief Returns the scalar loss associated with a top blob at a given index.
     *        返回与给定索引处 top blob 相关的损失。
     */
    inline Dtype loss(const int top_index) const
    {
      return (loss_.size() > top_index) ? loss_[top_index] : Dtype(0);
    }

    /**
     * @brief Sets the loss associated with a top blob at a given index.
     *        设置与给定索引处的 top blob 相关的损失。
     */
    inline void set_loss(const int top_index, const Dtype value)
    {
      if (loss_.size() <= top_index)
      {
        loss_.resize(top_index + 1, Dtype(0));
      }
      loss_[top_index] = value;
    }

    /**
     * @brief Returns the layer type.
     */
    virtual inline const char *type() const { return ""; }

    /**
     * @brief Returns the exact number of bottom blobs required by the layer,
     *        or -1 if no exact number is required.
     *        返回层所需的输入blob的确切数量，如果不需要确切数量，则返回-1。
     *
     * This method should be overridden to return a non-negative value if your
     * layer expects some exact number of bottom blobs.
     */
    virtual inline int ExactNumBottomBlobs() const { return -1; }
    /**
     * @brief Returns the minimum number of bottom blobs required by the layer,
     *        or -1 if no minimum number is required.
     *        返回图层所需的最小输入 blobs 数量，如果不需要最小数量，则返回-1。
     *
     * This method should be overridden to return a non-negative value if your
     * layer expects some minimum number of bottom blobs.
     */
    virtual inline int MinBottomBlobs() const { return -1; }
    /**
     * @brief Returns the maximum number of bottom blobs required by the layer,
     *        or -1 if no maximum number is required.
     *        返回图层所需的最大输入 blobs 数量，如果不需要最大数量，则返回-1。
     *
     * This method should be overridden to return a non-negative value if your
     * layer expects some maximum number of bottom blobs.
     */
    virtual inline int MaxBottomBlobs() const { return -1; }
    /**
     * @brief Returns the exact number of top blobs required by the layer,
     *        or -1 if no exact number is required.
     *        返回层所需的输出blob的确切数量，如果不需要确切数量，则返回-1。
     *
     * This method should be overridden to return a non-negative value if your
     * layer expects some exact number of top blobs.
     */
    virtual inline int ExactNumTopBlobs() const { return -1; }
    /**
     * @brief Returns the minimum number of top blobs required by the layer,
     *        or -1 if no minimum number is required.
     *        返回图层所需的最小输出 blobs 数量，如果不需要最小数量，则返回-1。
     *
     * This method should be overridden to return a non-negative value if your
     * layer expects some minimum number of top blobs.
     */
    virtual inline int MinTopBlobs() const { return -1; }
    /**
     * @brief Returns the maximum number of top blobs required by the layer,
     *        or -1 if no maximum number is required.
     *        返回图层所需的最大输出 blobs 数量，如果不需要最小数量，则返回-1。
     *
     * This method should be overridden to return a non-negative value if your
     * layer expects some maximum number of top blobs.
     */
    virtual inline int MaxTopBlobs() const { return -1; }
    /**
     * @brief Returns true if the layer requires an equal number of bottom and
     *        top blobs.
     *        如果该层有相等数量的输入和输出，则返回true
     *
     * This method should be overridden to return true if your layer expects an
     * equal number of bottom and top blobs.
     */
    virtual inline bool EqualNumBottomTopBlobs() const { return false; }

    /**
     * @brief Return whether "anonymous" top blobs are created automatically
     *        by the layer.
     *
     * If this method returns true, Net::Init will create enough "anonymous" top
     * blobs to fulfill the requirement specified by ExactNumTopBlobs() or
     * MinTopBlobs().
     */
    virtual inline bool AutoTopBlobs() const { return false; }

    /**
     * @brief Return whether to allow force_backward for a given bottom blob
     *        index.
     *
     * If AllowForceBackward(i) == false, we will ignore the force_backward
     * setting and backpropagate to blob i only if it needs gradient information
     * (as is done when force_backward == false).
     */
    virtual inline bool AllowForceBackward(const int bottom_index) const
    {
      return true;
    }

    /**
     * @brief Specifies whether the layer should compute gradients w.r.t. a
     *        parameter at a particular index given by param_id.
     *        指定层是否应计算参数在param_id给定的特定索引处的梯度
     *
     * You can safely ignore false values and always compute gradients
     * for all parameters, but possibly with wasteful computation.
     */
    inline bool param_propagate_down(const int param_id)
    {
      return (param_propagate_down_.size() > param_id) ? param_propagate_down_[param_id] : false;
    }
    /**
     * @brief Sets whether the layer should compute gradients w.r.t. a
     *        parameter at a particular index given by param_id.
     *        设置层是否应计算参数在param_id给定的特定索引处的梯度
     */
    inline void set_param_propagate_down(const int param_id, const bool value)
    {
      if (param_propagate_down_.size() <= param_id)
      {
        param_propagate_down_.resize(param_id + 1, true);
      }
      param_propagate_down_[param_id] = value;
    }

  protected:
    /** The protobuf that stores the layer parameters
     * 这是一个用于存储层参数的对象。在Protobuf（Google的一种数据交换格式）
     * 中定义了层的各种参数，如卷积层的卷积核大小、步长等。
     */
    LayerParameter layer_param_;
    /** The phase: TRAIN or TEST
     * 这个成员变量用于指示当前是训练阶段（TRAIN）还是测试阶段（TEST）。
     * 在训练和测试阶段，某些层的操作可能会有所不同，
     * 例如，dropout层在训练时会随机丢弃一些神经元，而在测试时则不会。
     */
    Phase phase_;
    /** The vector that stores the learnable parameters as a set of blobs.
     * 这是一个存储可学习参数的向量，每个参数都是一个Blob智能指针。
     * Blob是Caffe中用于存储数据的多维数组， Dtype通常是指数据类型，如float或double。
     */
    vector<shared_ptr<Blob<Dtype> > > blobs_;
    /** Vector indicating whether to compute the diff of each param blob.
     * 这个向量指示是否需要计算每个参数blob的梯度（diff）。
     * 在反向传播过程中，这个向量用于决定哪些参数需要更新。
     */
    vector<bool> param_propagate_down_;

    /** The vector that indicates whether each top blob has a non-zero weight in
     *  the objective function.
     * 这个向量表示每个顶部blob（top blob）在目标函数中是否有非零权重。
     * 在训练过程中，每个blob的损失贡献可能不同，这个向量用于记录这些贡献。
     */
    vector<Dtype> loss_;

    /** @brief Using the CPU device, compute the layer output. */
    virtual void Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                             const vector<Blob<Dtype> *> &top) = 0;
    /**
     * @brief Using the GPU device, compute the layer output.
     *        Fall back to Forward_cpu() if unavailable.
     */
    virtual void Forward_gpu(const vector<Blob<Dtype> *> &bottom,
                             const vector<Blob<Dtype> *> &top)
    {
      // LOG(WARNING) << "Using CPU code as backup.";
      return Forward_cpu(bottom, top);
    }

    /**
     * @brief Using the CPU device, compute the gradients for any parameters and
     *        for the bottom blobs if propagate_down is true.
     */
    virtual void Backward_cpu(const vector<Blob<Dtype> *> &top,
                              const vector<bool> &propagate_down,
                              const vector<Blob<Dtype> *> &bottom) = 0;
    /**
     * @brief Using the GPU device, compute the gradients for any parameters and
     *        for the bottom blobs if propagate_down is true.
     *        Fall back to Backward_cpu() if unavailable.
     */
    virtual void Backward_gpu(const vector<Blob<Dtype> *> &top,
                              const vector<bool> &propagate_down,
                              const vector<Blob<Dtype> *> &bottom)
    {
      // LOG(WARNING) << "Using CPU code as backup.";
      Backward_cpu(top, propagate_down, bottom);
    }

    /**
     * Called by the parent Layer's SetUp to check that the number of bottom
     * and top Blobs provided as input match the expected numbers specified by
     * the {ExactNum,Min,Max}{Bottom,Top}Blobs() functions.
     */
    virtual void CheckBlobCounts(const vector<Blob<Dtype> *> &bottom,
                                 const vector<Blob<Dtype> *> &top)
    {
      if (ExactNumBottomBlobs() >= 0)
      {
        CHECK_EQ(ExactNumBottomBlobs(), bottom.size())
            << type() << " Layer takes " << ExactNumBottomBlobs()
            << " bottom blob(s) as input.";
      }
      if (MinBottomBlobs() >= 0)
      {
        CHECK_LE(MinBottomBlobs(), bottom.size())
            << type() << " Layer takes at least " << MinBottomBlobs()
            << " bottom blob(s) as input.";
      }
      if (MaxBottomBlobs() >= 0)
      {
        CHECK_GE(MaxBottomBlobs(), bottom.size())
            << type() << " Layer takes at most " << MaxBottomBlobs()
            << " bottom blob(s) as input.";
      }
      if (ExactNumTopBlobs() >= 0)
      {
        CHECK_EQ(ExactNumTopBlobs(), top.size())
            << type() << " Layer produces " << ExactNumTopBlobs()
            << " top blob(s) as output.";
      }
      if (MinTopBlobs() >= 0)
      {
        CHECK_LE(MinTopBlobs(), top.size())
            << type() << " Layer produces at least " << MinTopBlobs()
            << " top blob(s) as output.";
      }
      if (MaxTopBlobs() >= 0)
      {
        CHECK_GE(MaxTopBlobs(), top.size())
            << type() << " Layer produces at most " << MaxTopBlobs()
            << " top blob(s) as output.";
      }
      if (EqualNumBottomTopBlobs())
      {
        CHECK_EQ(bottom.size(), top.size())
            << type() << " Layer produces one top blob as output for each "
            << "bottom blob input.";
      }
    }

    /**
     * Called by SetUp to initialize the weights associated with any top blobs in
     * the loss function. Store non-zero loss weights in the diff blob.
     * 由SetUp调用，以初始化与损失函数中任何输出blob关联的权重。将非零损失权重存储在diff blob中。
     */
    inline void SetLossWeights(const vector<Blob<Dtype> *> &top)
    {
      const int num_loss_weights = layer_param_.loss_weight_size();
      if (num_loss_weights)
      {
        CHECK_EQ(top.size(), num_loss_weights) << "loss_weight must be "
                                                  "unspecified or specified once per top blob.";
        for (int top_id = 0; top_id < top.size(); ++top_id)
        {
          const Dtype loss_weight = layer_param_.loss_weight(top_id);
          if (loss_weight == Dtype(0))
          {
            continue;
          }
          this->set_loss(top_id, loss_weight);
          const int count = top[top_id]->count();
          Dtype *loss_multiplier = top[top_id]->mutable_cpu_diff();
          caffe_set(count, loss_weight, loss_multiplier);
        }
      }
    }

  private:
    DISABLE_COPY_AND_ASSIGN(Layer);
    // private: Layer(const Layer&); Layer& operator=(const Layer&)

  }; // class Layer

  // Forward and backward wrappers. You should implement the cpu and
  // gpu specific implementations instead, and should not change these
  // functions.
  template <typename Dtype>
  inline Dtype Layer<Dtype>::Forward(const vector<Blob<Dtype> *> &bottom,
                                     const vector<Blob<Dtype> *> &top)
  {
    Dtype loss = 0;  /// 初始化损失值为 0
    Reshape(bottom, top); /// 根据输入和输出挑战层的大小
    switch (Caffe::mode())
    {
    case Caffe::CPU:
      // 调用CPU特定的前向传播函数 
      Forward_cpu(bottom, top);
      // 计算损失值
      for (int top_id = 0; top_id < top.size(); ++top_id)
      {
        // 如果当前输出不需要计算损失
        if (!this->loss(top_id))
        {
          continue;
        }
        const int count = top[top_id]->count(); // 获取 Blob 中元素的数量
        const Dtype *data = top[top_id]->cpu_data(); // 获取 Blob 的数据指针
        const Dtype *loss_weights = top[top_id]->cpu_diff(); // 获取 Blob 的损失权重指针
        loss += caffe_cpu_dot(count, data, loss_weights); // 计算并累加损失
      }
      break;
    case Caffe::GPU:
      Forward_gpu(bottom, top);
#ifndef CPU_ONLY
      for (int top_id = 0; top_id < top.size(); ++top_id)
      {
        if (!this->loss(top_id))
        {
          continue;
        }
        const int count = top[top_id]->count();// 获取 Blob 中元素的数量
        const Dtype *data = top[top_id]->gpu_data();// 获取 Blob 的数据指针
        const Dtype *loss_weights = top[top_id]->gpu_diff();// 获取 Blob 的损失权重指针
        Dtype blob_loss = 0; // 初始化 Blob 的损失值为 0
        caffe_gpu_dot(count, data, loss_weights, &blob_loss); // 在 GPU 上计算点积
        loss += blob_loss; // 累加到总损失中
      }
#endif
      break;
    default:
      LOG(FATAL) << "Unknown caffe mode.";
    }
    return loss; // 返回计算得到的损失值
  }

  template <typename Dtype>
  inline void Layer<Dtype>::Backward(const vector<Blob<Dtype> *> &top,
                                     const vector<bool> &propagate_down,
                                     const vector<Blob<Dtype> *> &bottom)
  {
    switch (Caffe::mode())
    {
    case Caffe::CPU:
      Backward_cpu(top, propagate_down, bottom);
      break;
    case Caffe::GPU:
      Backward_gpu(top, propagate_down, bottom);
      break;
    default:
      LOG(FATAL) << "Unknown caffe mode.";
    }
  }

  // Serialize LayerParameter to protocol buffer 将参数信息序列化到protocol buffer
  template <typename Dtype>
  void Layer<Dtype>::ToProto(LayerParameter *param, bool write_diff)
  {
    param->Clear();
    param->CopyFrom(layer_param_);
    param->clear_blobs();
    for (int i = 0; i < blobs_.size(); ++i)
    {
      blobs_[i]->ToProto(param->add_blobs(), write_diff);
    }
  }

} // namespace caffe

#endif // CAFFE_LAYER_H_
