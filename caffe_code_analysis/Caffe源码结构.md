# Caffe源码结构

## 1.   目录结构

![image-20240117165331025](assets/Caffe%E6%BA%90%E7%A0%81%E7%BB%93%E6%9E%84/image-20240117165331025.png)![image-20240117165352433](assets/Caffe%E6%BA%90%E7%A0%81%E7%BB%93%E6%9E%84/image-20240117165352433.png)

### 1.1     核心代码文件夹

* **tools**：保存的源码是用于生成二进制处理程序的，caffe在训练时实际是直接调用这些二进制文件

* **include**：Caffe的实现代码的头文件
* **src**：实现Caffe的源文件



### 1.2     主要文件夹

* **data**：用于存放下载的训练数据集
* **docs**：帮助文档
* **example**：一些代码样例
* **matlab**：MATLAB接口文件
* **python**：Python接口文件
* **model**：一些配置好的模型参数
* **scripts**：一些文档和数据用到的脚本



## 2.   源码结构

![image-20240117165943619](assets/Caffe%E6%BA%90%E7%A0%81%E7%BB%93%E6%9E%84/image-20240117165943619.png)

* **src**
  * **gtest**：google test 测试
  * **caffe**：caffe源代码
    * **blob.cpp**：Blob 存储数据 的结构的实现
    * **layer.cpp**：Layer 网络中的层 的锁操作，详细实现在layer文件夹中
    * **net.cpp**：Net 网络，包含多个层 的实现
    * **solver.cpp**：Solver 训练 的锁操作，详细实现在soler文件夹
    * **test**：用gtest测试caffe的代码
    * **util**：数据转换时用的一些代码。caffe速度快，很大程度上得益于内存设计上的优化（blob数据结构采用proto）和对卷积的优化（部分与im2col相关）
    * **proto**：Google Protocol Buffer 是一种数据存储格式，帮助caffe提速
    * **layers**：Layer（网络中的层）的实现
    * **solvers**：Solver （训练）的实现



## 3.   源码架构

Caffe架构如下图所示

![caffe_code_overview.png](assets/Caffe%E6%BA%90%E7%A0%81%E7%BB%93%E6%9E%84/caffe_code_overview-17054827824922.png)

Caffe框架主要有四个组件，Blob，Solver，Net，Layer。

- Blob是Caffe实际存储数据的结构，是一个不定维的矩阵，在Caffe中一般用来表示一个拉直的四维矩阵，四个维度分别对应**Batch Size（N），Feature Map的通道数（C）,Feature Map高度(H)和宽度(W)。**
- Layer是Net的基本组成单元，例如一个卷积层或一個池化层。每个Layer的输入和输出Feature map表示为Input Blob和Output Blob。
- 每个Net则由若干个Layer构成。
- Solver负责深度网络的训练，每个Solver中包含一个训练网络对象和一个测试网络对象。



## 4.   工厂模式

工厂模式示意图如下图所示

![factory_pattern.png](assets/Caffe%E6%BA%90%E7%A0%81%E7%BB%93%E6%9E%84/factory_pattern-17054835212084.png)

Solver和Layer使用了工厂模式。

拿Layer举例，layer.hpp中的Layer类是总的产品标准，使用virtual修饰函数，layer文件夹中的PoolingLayer、ConcatLayer等类继承Layer类，是Layer类的不同实现。然后layer_factory.hpp中的LayerRegistry类实现了Layer的注册，根据层的名称然后new对应的类返回Layer类型。至此，工厂模式流程完成。



## 5.   依赖库及其作用

### 5.1   必须依赖库

* BLAS库
  * 作用：调用基础线性代数函数
  * 可以选择ATLAS，MKL，OpenBLAS
  * BLAS（Basic Linear Algebra Subprograms，基础线性代数程序集）是一个应用程序接口（API）标准，用以规范发布基础线性代数操作的数值库（如矢量或矩阵乘法）。

* Boost库

  * 作用：C++主程序编写
  * 它是一个可移植、跨平台，提供源代码的C++库，作为标准库的后备。很多性能被C++11，C++14支持，不过有差别。
  * shared_ptr.hpp：智能指针

  - date_time/posix_time/posix_time.hpp：时间操作函数；

  - make_shared.hpp：make_shared工厂函数代替new操作符；

  - thread.hpp：线程操作；

  - math/special_functions/next.hpp：数学函数；

  - python.hpp：C++/Python互操作；

  - python/raw_function.hpp：C++/Python互操作；

  - python/suite/indexing/vector_indexing_suite.hpp：C++/Python互操作；

* ProtoBuf库：Google Protocol Buffer
  * 作用：用于文本解析，即解析prototxt文件
  * 它是一种轻便高效的结构化数据存储格式，可以用于结构化数据串行化，或者说序列化
  * 要使用ProtoBuf库，首先需要自己编写一个.proto文件，定义我们程序中需要处理的结构化数据，在protobuf中，结构化数据被称为Message。在一个 .proto 文件中可以定义多个消息类型。用 Protobuf 编译器 （protoc.exe）将 .proto 文件编译成目标语言，会生成对应的 .h 文件和 .cc 文件，.proto 文件中的每一个消息有一个对应的类。
* GLog库
  * 作用：日志输出
  * 它是一个应用程序的日志库，提供基于C++风格的流的日志API，以及各种辅助的宏。它的使用方式与C++的stream操作类似。

* GFlags库
  * 作用：处理命令行参数
  * 它是google的一个开源的处理命令行参数的库，使用C++开发，可以替代getopt函数。GFlags与getopt函数不同，在GFlags中，标记的定义分散在源代码中，不需要列举在一个地方。

* HDF5库

  * 作用：支持的数据库之一
  * HDF（HierarchicalData File）是美国国家高级计算应用中心（NCSA）为了满足各种领域研究需求而研制的一种能高效存储和分发科学数据的新型数据格式。它可以存储不同类型的图像和数码数据的文件格式，并且可以在不同类型的机器上传输，同时还有统一处理这种文件格式的函数库。

  * HDF5是分层式数据管理结构。HDF5不但能处理更多的对象，存储更大的文件，支持并行I/O，线程和具备现代操作系统与应用程序所要求的其它特性，而且数据模型变得更简单，概括性更强。



### 5.2     可选依赖库

* OpenCV库
  * 作用：图像操作
  * OpenCV，Open Source Computer Vision Library，是一个跨平台计算机视觉库。它轻量级而且高效——由一系列 C 函数和少量 C++ 类构成，同时提供了Python、Ruby、MATLAB等语言的接口，实现了图像处理和计算机视觉方面的很多通用算法。

* LevelDB库

  * 作用：支持的数据库之一
  * 它是一个超级快、超级小的Key-Value数据存储服务，是由OpenLDAP项目的Symas开发的。使用内存映射文件，因此读取的性能跟内存数据库一样，其大小受限于虚拟地址空间的大小。

  * 依赖snappy库

    * 作用：压缩与解压缩

    * 是一个C++库，用来压缩和解压缩的开发包。它旨在提供高速压缩速度和合理的压缩率。Snappy比zlib更快，但文件相对要大20%到100%

      

* CUDA库

  * 作用：编写GPU程序，进行GPU加速，深度学习的引擎
  * CUDA（Compute Unified Device Architecture，统一计算架构[1]）是由NVIDIA所推出的一种集成技术，可以利用GPU作为C-编译器的开发环境。

* cuDNN

  * 作用：用于Caffe的GPU加速
  * cuDNN是用于深度神经网络的GPU加速库。它强调性能、易用性和低内存开销。





## 参考

* [Caffe源码结构 — 李华清的博客 1.0-dev 文档 (hqli.github.io)](https://hqli.github.io/doc/experience/caffe_code/Caffe源码结构.html)















