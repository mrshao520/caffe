if(CPU_ONLY)
  return()
endif()

# 设置GPU算力
# Known NVIDIA GPU achitectures Caffe can be compiled for.
# This list will be used for CUDA_ARCH_NAME = All option
set(Caffe_known_gpu_archs "20 21(20) 30 35 50 60 61 70 75 86 89")

################################################################################################
# A function for automatic detection of GPUs installed  (if autodetection is enabled)
# Usage:
#   caffe_detect_installed_gpus(out_variable)
function(caffe_detect_installed_gpus out_variable)
  message(STATUS "CUDA_gpu_detect_output : ${CUDA_gpu_detect_output}")
  unset(CUDA_gpu_detect_output CACHE)
  if(NOT CUDA_gpu_detect_output)
    set(__cufile ${PROJECT_BINARY_DIR}/detect_cuda_archs.cu)

    file(WRITE ${__cufile} ""
      "#include <cstdio>\n"
      "int main()\n"
      "{\n"
      "  int count = 0;\n"
      "  if (cudaSuccess != cudaGetDeviceCount(&count)) return -1;\n"
      "  if (count == 0) return -1;\n"
      "  for (int device = 0; device < count; ++device)\n"
      "  {\n"
      "    cudaDeviceProp prop;\n"
      "    if (cudaSuccess == cudaGetDeviceProperties(&prop, device))\n"
      "      std::printf(\"%d.%d \", prop.major, prop.minor);\n"
      "  }\n"
      "  return 0;\n"
      "}\n")

    # 运行给定序列的一个或多个命令。
    execute_process(COMMAND "${CUDA_NVCC_EXECUTABLE}" "--run" "${__cufile}" # CMake 直接使用操作系统API执行子进程：
                    WORKING_DIRECTORY "${PROJECT_BINARY_DIR}/CMakeFiles/"   # 指定的目录将被设置成子进程的当前工作目录
                    # 该变量将被设置成包含最后一个子进程的结果，
                    # 这将是最后一个子级的整数返回代码或描述错误情况的字符串。
                    RESULT_VARIABLE __nvcc_res    # return 返回值   

                    # OUTPUT_VARIABLE ERROR_VARIABLE: 命名的变量将分别设置为标准输出和标准错误管道的内容。
                    # 如果为两个管道命名相同的变量，则它们的输出将按生成的顺序合并
                    OUTPUT_VARIABLE __nvcc_out    # printf 管道输出

                    # OUTPUT_QUIET, ERROR_QUIET: OUTPUT_VARIABLE 上的标准输出或 
                    # ERROR_VARIABLE 上的标准错误未连接（无可变内容）。
                    # *_FILE 和 ECHO_*_VARIABLE 选件不受影响。
                    ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)

    message(STATUS "detect_cuda_archs RESULT_VARIABLE : ${__nvcc_res}")
    message(STATUS "detect_cuda_archs OUTPUT_VARIABLE : ${__nvcc_out}")

    if(__nvcc_res EQUAL 0)
      # 在变量__nvcc_out中查找子串“2.1”，并将所有找到的“2.1”替换为“2.1(2.0)”。
      string(REPLACE "2.1" "2.1(2.0)" __nvcc_out "${__nvcc_out}")
      set(CUDA_gpu_detect_output ${__nvcc_out} CACHE INTERNAL "Returned GPU architetures from caffe_detect_gpus tool" FORCE)
    endif()
  endif()

  if(NOT CUDA_gpu_detect_output)
    message(STATUS "Automatic GPU detection failed. Building for all known architectures.")
    set(${out_variable} ${Caffe_known_gpu_archs} PARENT_SCOPE)
  else()
    set(${out_variable} ${CUDA_gpu_detect_output} PARENT_SCOPE)
  endif()
endfunction()


################################################################################################
# Function for selecting GPU arch flags for nvcc based on CUDA_ARCH_NAME
# Usage:
#   caffe_select_nvcc_arch_flags(out_variable)
function(caffe_select_nvcc_arch_flags out_variable)
  # List of arch names
  set(__archs_names "Fermi" "Kepler" "Maxwell" "Pascal" "All" "Manual")
  set(__archs_name_default "All")
  if(NOT CMAKE_CROSSCOMPILING)  # 交叉编译
    list(APPEND __archs_names "Auto")
    set(__archs_name_default "Auto")
  endif()

  # 将在CMake GUI上面显示 CUDA_ARCH_NAME, 并设置可选项
  # set CUDA_ARCH_NAME strings (so it will be seen as dropbox in CMake-Gui)
  set(CUDA_ARCH_NAME ${__archs_name_default} CACHE STRING "Select target NVIDIA GPU achitecture.")
  # 为某个范围的零个或多个对象设置一个属性。
  # 所需的 PROPERTY 选项后紧跟要设置的属性名称。其余参数用于以分号分隔列表的形式组成属性值。
  set_property( CACHE CUDA_ARCH_NAME PROPERTY STRINGS "" ${__archs_names} )
  mark_as_advanced(CUDA_ARCH_NAME)


  # verify CUDA_ARCH_NAME value
  if(NOT ";${__archs_names};" MATCHES ";${CUDA_ARCH_NAME};")
    string(REPLACE ";" ", " __archs_names "${__archs_names}")
    message(FATAL_ERROR "Only ${__archs_names} architeture names are supported.")
  endif()

  if(${CUDA_ARCH_NAME} STREQUAL "Manual")
    set(CUDA_ARCH_BIN ${Caffe_known_gpu_archs} CACHE STRING "Specify 'real' GPU architectures to build binaries for, BIN(PTX) format is supported")
    set(CUDA_ARCH_PTX "50"                     CACHE STRING "Specify 'virtual' PTX architectures to build PTX intermediate code for")
    mark_as_advanced(CUDA_ARCH_BIN CUDA_ARCH_PTX)
  else()
    unset(CUDA_ARCH_BIN CACHE)
    unset(CUDA_ARCH_PTX CACHE)
  endif()

  message(STATUS "CUDA_ARCH_NAME : ${CUDA_ARCH_NAME}")
  if(${CUDA_ARCH_NAME} STREQUAL "Fermi")
    set(__cuda_arch_bin "20 21(20)")
  elseif(${CUDA_ARCH_NAME} STREQUAL "Kepler")
    set(__cuda_arch_bin "30 35")
  elseif(${CUDA_ARCH_NAME} STREQUAL "Maxwell")
    set(__cuda_arch_bin "50")
  elseif(${CUDA_ARCH_NAME} STREQUAL "Pascal")
    set(__cuda_arch_bin "60 61")
  elseif(${CUDA_ARCH_NAME} STREQUAL "All")
    set(__cuda_arch_bin ${Caffe_known_gpu_archs})
  elseif(${CUDA_ARCH_NAME} STREQUAL "Auto")
    set(__cuda_arch_bin "60")
    # caffe_detect_installed_gpus(__cuda_arch_bin)
  else()  # (${CUDA_ARCH_NAME} STREQUAL "Manual")
    set(__cuda_arch_bin ${CUDA_ARCH_BIN})
  endif()

  # remove dots and convert to lists
  # 使用正则表达式来替换变量__cuda_arch_bin中的所有点（.）字符
  string(REGEX REPLACE "\\." "" __cuda_arch_bin "${__cuda_arch_bin}")
  string(REGEX REPLACE "\\." "" __cuda_arch_ptx "${CUDA_ARCH_PTX}")
  string(REGEX MATCHALL "[0-9()]+" __cuda_arch_bin "${__cuda_arch_bin}")
  string(REGEX MATCHALL "[0-9]+"   __cuda_arch_ptx "${__cuda_arch_ptx}")

  message(STATUS "__cuda_arch_bin : ${__cuda_arch_bin}")
  message(STATUS "__cuda_arch_ptx : ${__cuda_arch_ptx}")

  caffe_list_unique(__cuda_arch_bin __cuda_arch_ptx)

  set(__nvcc_flags "")
  set(__nvcc_archs_readable "")

  string(COMPARE LESS "${CUDA_VERSION}" "9.0" iscudaolderthan90)
  if(NOT iscudaolderthan90)
    string(REPLACE "21(20)" "" __cuda_arch_bin "${__cuda_arch_bin}")
    string(REPLACE "20" "" __cuda_arch_bin "${__cuda_arch_bin}")
  endif()

  # Tell NVCC to add binaries for the specified GPUs
  foreach(__arch ${__cuda_arch_bin})
    if(__arch MATCHES "([0-9]+)\\(([0-9]+)\\)")
      # User explicitly specified PTX for the concrete BIN
      list(APPEND __nvcc_flags -gencode arch=compute_${CMAKE_MATCH_2},code=sm_${CMAKE_MATCH_1})
      list(APPEND __nvcc_archs_readable sm_${CMAKE_MATCH_1})
    else()
      # User didn't explicitly specify PTX for the concrete BIN, we assume PTX=BIN
      list(APPEND __nvcc_flags -gencode arch=compute_${__arch},code=sm_${__arch})
      list(APPEND __nvcc_archs_readable sm_${__arch})
    endif()
  endforeach()

  # Tell NVCC to add PTX intermediate code for the specified architectures
  foreach(__arch ${__cuda_arch_ptx})
    list(APPEND __nvcc_flags -gencode arch=compute_${__arch},code=compute_${__arch})
    list(APPEND __nvcc_archs_readable compute_${__arch})
  endforeach()

  # string(REPLACE ";" " " __nvcc_flags "${__nvcc_flags}")
  string(REPLACE ";" " " __nvcc_archs_readable "${__nvcc_archs_readable}")
  set(${out_variable}          ${__nvcc_flags}          PARENT_SCOPE)
  set(${out_variable}_readable ${__nvcc_archs_readable} PARENT_SCOPE)
endfunction()

################################################################################################
# Short command for cuda compilation
# Usage:
#   caffe_cuda_compile(<objlist_variable> <cuda_files>)
macro(caffe_cuda_compile objlist_variable)
  foreach(var CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS_DEBUG)
    set(${var}_backup_in_cuda_compile_ "${${var}}")

    # we remove /EHa as it generates warnings under windows
    string(REPLACE "/EHa" "" ${var} "${${var}}")

  endforeach()

  if(UNIX OR APPLE)
    list(APPEND CUDA_NVCC_FLAGS -Xcompiler -fPIC)
  endif()

  if(APPLE)
    list(APPEND CUDA_NVCC_FLAGS -Xcompiler -Wno-unused-function)
  endif()

  cuda_compile(cuda_objcs ${ARGN})

  foreach(var CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS_DEBUG)
    set(${var} "${${var}_backup_in_cuda_compile_}")
    unset(${var}_backup_in_cuda_compile_)
  endforeach()

  set(${objlist_variable} ${cuda_objcs})
endmacro()

################################################################################################
# Short command for cuDNN detection. Believe it soon will be a part of CUDA toolkit distribution.
# That's why not FindcuDNN.cmake file, but just the macro
# Usage:
#   detect_cuDNN()
function(detect_cuDNN)
  set(CUDNN_ROOT "" CACHE PATH "CUDNN root folder")

  find_path(CUDNN_INCLUDE cudnn.h
            PATHS ${CUDNN_ROOT} $ENV{CUDNN_ROOT} ${CUDA_TOOLKIT_INCLUDE}
            DOC "Path to cuDNN include directory." )

  # dynamic libs have different suffix in mac and linux
  if(APPLE)
    set(CUDNN_LIB_NAME "libcudnn.dylib")
  else()
    set(CUDNN_LIB_NAME "libcudnn.so")
  endif()

  # 获取完整文件名的特定组成部分。
  # get_filename_component(<var> <FileName> <mode> [CACHE])
  get_filename_component(__libpath_hist ${CUDA_CUDART_LIBRARY} PATH)
  message(STATUS "__libpath_hist : ${__libpath_hist}")
  message(STATUS "CUDA_CUDART_LIBRARY : ${CUDA_CUDART_LIBRARY}")


  find_library(CUDNN_LIBRARY NAMES ${CUDNN_LIB_NAME}
   PATHS ${CUDNN_ROOT} $ENV{CUDNN_ROOT} ${CUDNN_INCLUDE} ${__libpath_hist} ${__libpath_hist}/../lib
   DOC "Path to cuDNN library.")
  
  if(CUDNN_INCLUDE AND CUDNN_LIBRARY)
    set(HAVE_CUDNN  TRUE PARENT_SCOPE)
    set(CUDNN_FOUND TRUE PARENT_SCOPE)

    file(READ ${CUDNN_INCLUDE}/cudnn.h CUDNN_VERSION_FILE_CONTENTS)
    # file(READ ${CUDNN_INCLUDE}/cudnn_version.h CUDNN_VERSION_FILE_CONTENTS)

    # cuDNN v3 and beyond
    string(REGEX MATCH "define CUDNN_MAJOR * +([0-9]+)"
           CUDNN_VERSION_MAJOR "${CUDNN_VERSION_FILE_CONTENTS}")
    string(REGEX REPLACE "define CUDNN_MAJOR * +([0-9]+)" "\\1"
           CUDNN_VERSION_MAJOR "${CUDNN_VERSION_MAJOR}")
    string(REGEX MATCH "define CUDNN_MINOR * +([0-9]+)"
           CUDNN_VERSION_MINOR "${CUDNN_VERSION_FILE_CONTENTS}")
    string(REGEX REPLACE "define CUDNN_MINOR * +([0-9]+)" "\\1"
           CUDNN_VERSION_MINOR "${CUDNN_VERSION_MINOR}")
    string(REGEX MATCH "define CUDNN_PATCHLEVEL * +([0-9]+)"
           CUDNN_VERSION_PATCH "${CUDNN_VERSION_FILE_CONTENTS}")
    string(REGEX REPLACE "define CUDNN_PATCHLEVEL * +([0-9]+)" "\\1"
           CUDNN_VERSION_PATCH "${CUDNN_VERSION_PATCH}")

    if(NOT CUDNN_VERSION_MAJOR)
      message(STATUS "Not find CUDNN_VERSION, maybe CUDNN_VERSION is not in ${CUDNN_INCLUDE}/cudnn.h")
      set(CUDNN_VERSION "???")
    else()
      set(CUDNN_VERSION "${CUDNN_VERSION_MAJOR}.${CUDNN_VERSION_MINOR}.${CUDNN_VERSION_PATCH}")
    endif()

    message(STATUS "Found cuDNN: ver. ${CUDNN_VERSION} found (include: ${CUDNN_INCLUDE}, library: ${CUDNN_LIBRARY})")

    # string(COMPARE <op> <string1> <string2> <out-var>)
    # 比较字符串并将 true 或 false 存储在 <output_variable> 中。
    string(COMPARE LESS "${CUDNN_VERSION_MAJOR}" 3 cuDNNVersionIncompatible)

    if(cuDNNVersionIncompatible)
      message(FATAL_ERROR "cuDNN version >3 is required.")
    endif()

    set(CUDNN_VERSION "${CUDNN_VERSION}" PARENT_SCOPE)
    mark_as_advanced(CUDNN_INCLUDE CUDNN_LIBRARY CUDNN_ROOT)

  endif()
endfunction()

################################################################################################
###  Non macro section
################################################################################################

find_package(CUDA 5.5 QUIET)
find_cuda_helper_libs(curand)  # cmake 2.8.7 compatibility which doesn't search for curand

if(NOT CUDA_FOUND)
  message(STATUS "CUDA not found!")
  return()
endif()

set(HAVE_CUDA TRUE)
message(STATUS "CUDA detected: " ${CUDA_VERSION})
list(APPEND Caffe_INCLUDE_DIRS PUBLIC ${CUDA_INCLUDE_DIRS})
list(APPEND Caffe_LINKER_LIBS PUBLIC ${CUDA_CUDART_LIBRARY}
                                     ${CUDA_curand_LIBRARY} ${CUDA_CUBLAS_LIBRARIES})

# cudnn detection
if(USE_CUDNN)
  detect_cuDNN()
  if(HAVE_CUDNN)
    list(APPEND Caffe_DEFINITIONS PUBLIC -DUSE_CUDNN)
    list(APPEND Caffe_INCLUDE_DIRS PUBLIC ${CUDNN_INCLUDE})
    list(APPEND Caffe_LINKER_LIBS PUBLIC ${CUDNN_LIBRARY})
  endif()
endif()

# setting nvcc arch flags
caffe_select_nvcc_arch_flags(NVCC_FLAGS_EXTRA)
list(APPEND CUDA_NVCC_FLAGS ${NVCC_FLAGS_EXTRA})
message(STATUS "flags Added CUDA NVCC flags for: ${NVCC_FLAGS_EXTRA}")
message(STATUS "readable Added CUDA NVCC flags for: ${NVCC_FLAGS_EXTRA_readable}")

# Boost 1.55 workaround, see https://svn.boost.org/trac/boost/ticket/9392 or
# https://github.com/ComputationalRadiationPhysics/picongpu/blob/master/src/picongpu/CMakeLists.txt
if(Boost_VERSION EQUAL 105500)
  message(STATUS "Cuda + Boost 1.55: Applying noinline work around")
  # avoid warning for CMake >= 2.8.12
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} \"-DBOOST_NOINLINE=__attribute__((noinline))\" ")
endif()

# disable some nvcc diagnostic that apears in boost, glog, glags, opencv, etc.
# 使用 foreach 循环来处理一系列的诊断抑制选项，并将它们添加到 CUDA_NVCC_FLAGS 变量中
foreach(diag cc_clobber_ignored integer_sign_change useless_using_declaration set_but_not_used)
  # 将一个诊断抑制选项添加到CUDA_NVCC_FLAGS变量中。
  # 这个选项是通过-Xcudafe指定的，后面跟着--diag_suppress=${diag}，其中${diag}是在循环变量diag中当前的值
  list(APPEND CUDA_NVCC_FLAGS -Xcudafe --diag_suppress=${diag})
endforeach()

# setting default testing device
if(NOT CUDA_TEST_DEVICE)
  set(CUDA_TEST_DEVICE -1)
endif()

mark_as_advanced(CUDA_BUILD_CUBIN CUDA_BUILD_EMULATION CUDA_VERBOSE_BUILD)
mark_as_advanced(CUDA_SDK_ROOT_DIR CUDA_SEPARABLE_COMPILATION)

# Handle clang/libc++ issue
if(APPLE)
  caffe_detect_darwin_version(OSX_VERSION)

  # OSX 10.9 and higher uses clang/libc++ by default which is incompatible with old CUDA toolkits
  if(OSX_VERSION VERSION_GREATER 10.8)
    # enabled by default if and only if CUDA version is less than 7.0
    caffe_option(USE_libstdcpp "Use libstdc++ instead of libc++" (CUDA_VERSION VERSION_LESS 7.0))
  endif()
endif()
