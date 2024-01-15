# 安装编译 Caffe



## 安装 Caffe 框架的第三方库

```bash
apt-get update
apt-get install libprotobuf-dev libleveldb-dev  \
		libsnappy-dev libhdf5-serial-dev  \
		protobuf-compiler libboost-all-dev \
		libgflags-dev libgoogle-glog-dev liblmdb-dev

# apt-get install libopencv-dev # 下载的是OpenCV4，而Caffe需要的是OpenCV3

# 通过或 MKL 安装 ATLAS 或安装 OpenBLAS 以获得更好的 CPU 性能
apt-get install libatlas-base-dev libopenblas-dev
```



## 安装 OpenCV3

```bash
apt-get install libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
apt-get install python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev

# 下载源码
wget https://github.com/opencv/opencv/archive/3.4.14.zip

# 解压缩
unzip 3.4.14.zip

# CUDA 12.0放弃了对遗留纹理引用的支持。因此，任何使用遗留纹理引用的代码都不能再使用 CUDA 12.0或更高版本进行正确编译。
# 正如评论中指出的，通过恢复到 CUDA 11.x，其中仍然支持遗留纹理引用(尽管不推荐) ，您不会遇到这个问题。
# 无法使用CUDA 12.0编译 v3.x 版本
mkdir build && cd build
cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local ..
make -j12 
make install

```



## 安装 Caffe 框架 makefile版本

* 修改配置文件

  ```bash
  # 重命令 example 后缀的配置文件
  cp Makefile.config.example Makefile.config
  ```
* OpenCV版本使用3，而不是默认的2

  ```bash
  # Uncomment if you're using OpenCV 3
  OPENCV_VERSION := 3 
  ```

* CUDA 架构部分，只保留 3.5 以上的算力。CUDA 11.4 版本不再支持低于 3.5 的算力。保留低算力会导致编译时报出找不到低算力 API 的错误。

```bash
# 原始配置：
# CUDA_ARCH := -gencode arch=compute_20,code=sm_20 \
# 		-gencode arch=compute_20,code=sm_21 \
# 		-gencode arch=compute_30,code=sm_30 \
# 		-gencode arch=compute_35,code=sm_35 \
# 		-gencode arch=compute_50,code=sm_50 \
# 		-gencode arch=compute_52,code=sm_52 \
# 		-gencode arch=compute_60,code=sm_60 \
# 		-gencode arch=compute_61,code=sm_61 \
# 		-gencode arch=compute_61,code=compute_61
#
# 去掉前三行 20、21、30，变为下面的样子
CUDA_ARCH := -gencode arch=compute_50,code=sm_50 \
		-gencode arch=compute_52,code=sm_52 \
		-gencode arch=compute_60,code=sm_60 \
		-gencode arch=compute_61,code=sm_61 \
		-gencode arch=compute_61,code=compute_61
```

* 线性代数库部分，默认使用的是是 ATLAS，我们需要将其改为 OpenBLAS。据很多资料说，OpenBLAS 比 ATLAS 运行得快，原因在于 ATLAS 对多线程 CPU 的支持不够。

  ```bash
  # BLAS := atlas
  # 找到上面这一行，将 atlas 改为 open
  BLAS := open
  ```

* 头文件目录和库目录要增加一些目录。主要是针对 hdf5 这项依赖。Ubuntu 20.04 里，用 apt-get 安装的 hdf5 依赖，头文件并不是放在 /usr/include 里，而是放在 /usr/include/hdf5/serial 里。库文件也挪了位置，不在 /usr/lib 里，而在 /usr/lib/x86_64-linux-gnu/hdf5/serial 里。这些子目录并不会被默认包含，我们需要明确指定。

```
# 找到下面三行
# # Whatever else you find you need goes here.
# INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include
# LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib
#
# 做如下修改：
# INCLUDE_DIRS 在末尾添加 /usr/include/hdf5/serial 目录
# LIBRARY_DIRS 在末尾添加 /usr/lib/x86_64-linux-gnu/hdf5/serial 目录
INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include /usr/include/hdf5/serial
LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib \
                /usr/lib/x86_64-linux-gnu/hdf5/serial
```



* 编译、安装

  ```
  make all -j8 && make test -j8 && make runtest -j8
  ```

  



## 安装 Caffe 框架 CMake版本



> Caffe 部分API不支持 CUDNN8 ，先使用docker，后续再研究



* 使用 python 3 版本

  ```cmake
  set(python_version "3" CACHE STRING "Specify which Python version to use")
  ```

  





