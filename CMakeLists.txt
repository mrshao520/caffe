# CMake最低版本
cmake_minimum_required(VERSION 2.8.7)

# cmake_policy(VERSION <min>[...<max>])  //通过版本来设置策略
# 使用 NEW 来声明后续代码依赖于此policy，相当于强制指定使用某个策略
# 使用 OLD 来向使用此policy的配置过程发出警告信息，相当于强制指定某个策略过期
if(POLICY CMP0046)
  cmake_policy(SET CMP0046 NEW)
endif()
if(POLICY CMP0054)
  cmake_policy(SET CMP0054 NEW)
endif()

# CMP0022 此策略是在 CMake 版本 2.8.12 中引入的。
# 当未设置策略并使用 OLD 行为时， CMake 版本 3.26.0-rc3 会发出警告。
# 使用 cmake_policy() 命令将其明确设置为 OLD 或 NEW 。
if(POLICY CMP0022)
  cmake_policy(SET CMP0022 OLD)
endif()


# ---[ Caffe project
# 设置项目名称，以及使用的编程语言
project(Caffe LANGUAGES C CXX)

# ---[ Caffe version
set(CAFFE_TARGET_VERSION "1.0.0" CACHE STRING "Caffe logical version")
set(CAFFE_TARGET_SOVERSION "1.0.0" CACHE STRING "Caffe soname version")

# 向编译器添加-D定义，作用于代码中的 #define
add_definitions(-DCAFFE_VERSION=${CAFFE_TARGET_VERSION})

# ---[ Using cmake scripts and modules
message(STATUS "PROJECT_SOURCE_DIR : ${PROJECT_SOURCE_DIR}")
# 这个变量用来定义自己的cmake模块所在的路径。
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

# 引入第三方项目
include(ExternalProject)

# 使用 GNUInstallDirs.cmake 定义目标安装的标准位置。
# CMAKE_INSTALL_BINDIR：这将用于定义用户可执行文件所在的子目录，即所选安装目录下的 bin 目录。
# CMAKE_INSTALL_LIBDIR：这将扩展到目标代码库(即静态库和动态库)所在的子目录。
#                       在64位系统上，它是 lib64 ，而在32位系统上，它只是 lib 。
# CMAKE_INSTALL_INCLUDEDIR：使用这个变量为头文件获取正确的子目录，该变量为 include。
include(GNUInstallDirs)
message(STATUS "CMAKE_INSTALL_BINDIR : ${CMAKE_INSTALL_BINDIR}")
message(STATUS "CMAKE_INSTALL_LIBDIR : ${CMAKE_INSTALL_LIBDIR}")
message(STATUS "CMAKE_INSTALL_INCLUDEDIR : ${CMAKE_INSTALL_INCLUDEDIR}")

include(cmake/Utils.cmake)
include(cmake/Targets.cmake)
include(cmake/Misc.cmake)
include(cmake/Summary.cmake)
include(cmake/ConfigGen.cmake)

##########################################
# test_option
# include(cmake_learning/Utils.cmake)
# test_option(TEST_CUDA "Build project with CUDA" ON if NOT ON)
# test_option(TEST_CUDA "Build project with CUDA" OFF)


# ---[ Options
caffe_option(CPU_ONLY  "Build Caffe without CUDA support" OFF) # TODO: rename to USE_CUDA
caffe_option(USE_CUDNN "Build Caffe with cuDNN library support" ON IF NOT CPU_ONLY)
caffe_option(USE_NCCL "Build Caffe with NCCL library support" OFF)
caffe_option(BUILD_SHARED_LIBS "Build shared libraries" ON)
caffe_option(BUILD_python "Build Python wrapper" ON)
set(python_version "2" CACHE STRING "Specify which Python version to use")
caffe_option(BUILD_matlab "Build Matlab wrapper" OFF IF UNIX OR APPLE)
caffe_option(BUILD_docs   "Build documentation" ON IF UNIX OR APPLE)
caffe_option(BUILD_python_layer "Build the Caffe Python layer" ON)
caffe_option(USE_OPENCV "Build with OpenCV support" ON)
caffe_option(USE_LEVELDB "Build with levelDB" ON)
caffe_option(USE_LMDB "Build with lmdb" ON)
caffe_option(ALLOW_LMDB_NOLOCK "Allow MDB_NOLOCK when reading LMDB files (only if necessary)" OFF)
caffe_option(USE_OPENMP "Link with OpenMP (when your BLAS wants OpenMP and you get linker errors)" OFF)

# This code is taken from https://github.com/sh1r0/caffe-android-lib
caffe_option(USE_HDF5 "Build with hdf5" ON)

# ---[ Dependencies  引入第三方库的CMAKE
include(cmake/Dependencies.cmake)

# ---[ Flags
if(UNIX OR APPLE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall")
endif()
message(STATUS "CMAKE_CXX_FLAGS : ${CMAKE_CXX_FLAGS}")
message(STATUS "operation system is ${CMAKE_SYSTEM} and name is ${CMAKE_SYSTEM_NAME}")

caffe_set_caffe_link()
message(STATUS "CMAKE_CXX_COMPILER_ID : ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "caffe_link : ${Caffe_LINK}")

message(STATUS "USE_libstdcpp : ${USE_libstdcpp}")
if(USE_libstdcpp)
  # g++默认libstdc++链接库
  # clang++默认libc++链接库
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++")
  message("-- Warning: forcing libstdc++ (controlled by USE_libstdcpp option in cmake)")
endif()

# ---[ Warnings
# 将 warning 放入 CMAKE_CXX_FLAGS
# 取消警告的编译选项         -Wno-sign-compare
# 取消对未使用参数的编译警告  -Wno-uninitialized
caffe_warnings_disable(CMAKE_CXX_FLAGS -Wno-sign-compare -Wno-uninitialized)

# ---[ Config generation
# 使用config_file()将给定的输入文件复制到输出文件，并替换输入文件内容中的一些变量值。
configure_file(cmake/Templates/caffe_config.h.in "${PROJECT_BINARY_DIR}/caffe_config.h")
message(STATUS "PROJECT_BINARY_DIR : ${PROJECT_BINARY_DIR}")

# ---[ Includes
set(Caffe_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
set(Caffe_SRC_DIR ${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_BINARY_DIR})

# ---[ Includes & defines for CUDA

# cuda_compile() does not have per-call dependencies or include pathes
# (cuda_compile() has per-call flags, but we set them here too for clarity)
#
# list(REMOVE_ITEM ...) invocations remove PRIVATE and PUBLIC keywords from collected definitions and include pathes
if(HAVE_CUDA)
  # pass include pathes to cuda_include_directories()
  set(Caffe_ALL_INCLUDE_DIRS ${Caffe_INCLUDE_DIRS})
  message(STATUS "Caffe_ALL_INCLUDE_DIRS : ${Caffe_ALL_INCLUDE_DIRS}")
  list(REMOVE_ITEM Caffe_ALL_INCLUDE_DIRS PRIVATE PUBLIC)

  # 设置应传递给nvcc的目录。这些路径通常包含其他文件
  cuda_include_directories(${Caffe_INCLUDE_DIR} ${Caffe_SRC_DIR} ${Caffe_ALL_INCLUDE_DIRS})

  # add definitions to nvcc flags directly
  set(Caffe_ALL_DEFINITIONS ${Caffe_DEFINITIONS})
  list(REMOVE_ITEM Caffe_ALL_DEFINITIONS PRIVATE PUBLIC)
  list(APPEND CUDA_NVCC_FLAGS ${Caffe_ALL_DEFINITIONS})
endif()

# ---[ Subdirectories
add_subdirectory(src/gtest)
add_subdirectory(src/caffe)
add_subdirectory(tools)
add_subdirectory(examples)
add_subdirectory(python)
add_subdirectory(matlab)
add_subdirectory(docs)

# ---[ Linter target
# 当你运行 make lint 或者在 IDE 中配置 lint 目标时，
# CMake 将执行位于 ${PROJECT_SOURCE_DIR}/cmake/lint.cmake 的脚本。
# 这个脚本应该包含了一系列的指令来执行代码检查、风格分析或其他任何你想要的 lint 任务。
add_custom_target(lint COMMAND ${CMAKE_COMMAND} -P ${PROJECT_SOURCE_DIR}/cmake/lint.cmake)

# ---[ pytest target
if(BUILD_python)
  add_custom_target(pytest COMMAND python${python_version} -m unittest discover -s caffe/test WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/python )
  add_dependencies(pytest pycaffe)
endif()

# ---[ uninstall target
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Uninstall.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/cmake/Uninstall.cmake
    IMMEDIATE @ONLY)

# uninstall 目标
# 设置之后运行 Uninstall.cmake，卸载caffe
add_custom_target(uninstall
    COMMAND ${CMAKE_COMMAND} -P
    ${CMAKE_CURRENT_BINARY_DIR}/cmake/Uninstall.cmake)

# ---[ Configuration summary  配置汇总
caffe_print_configuration_summary()

# ---[ Export configs generation
caffe_generate_export_configs()

