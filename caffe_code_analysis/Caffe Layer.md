# Caffe Layer

## 1.   概述

* **Blob**：是基础的数据结构，用来保存学习到的参数以及网络传输过程中产生的数据，神经网络的权重、偏置、激活值等
* **Layer**：是网络的基本单元，由此派生出了各种层。
* **Net**：是网络的搭建，将 Layer 所派生出层类组合成网络。
* **Solver**：是 Net 的求解。



## 2.   Layer

### 2.1     源码文件

* include/caffe/layer.hpp
* include/caffe/layers/...
* src/caffe/layer.cpp
* src/caffe/layer/...



### 2.2     整体概览









### 2.3     成员变量







### 2.4     成员函数

```c++
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
            blobs_[i].reset(new Blob<Dtype>()); ///< 创建一个新的Blob对象
            blobs_[i]->FromProto(layer_param_.blobs(i)); ///< 从protobuf消息中读取数据并初始化Blob
        }
    }
}
virtual ~Layer() {}
```

* 构造函数。首先对phase_赋值，在参数的blobs大小大于零情况下，blobs_（一个类型为指向blob类的shared_ptr指针的vector）申请空间，然后将传入的layer_param中的blob拷贝过来。

























