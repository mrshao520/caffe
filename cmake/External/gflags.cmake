if (NOT __GFLAGS_INCLUDED) # guard against multiple includes 防止多次引用
  set(__GFLAGS_INCLUDED TRUE)

  # use the system-wide gflags if present
  find_package(GFlags)
  if (GFLAGS_FOUND) 
    # gflags 已安装，并通过find_package找到GFLAGS
    set(GFLAGS_EXTERNAL FALSE)
  else()
    # gflags 未安装，通过ExternalProject_Add安装

    # gflags will use pthreads if it's available in the system, so we must link with it
    find_package(Threads)

    # build directory
    set(gflags_PREFIX ${CMAKE_BINARY_DIR}/external/gflags-prefix)
    # install directory
    set(gflags_INSTALL ${CMAKE_BINARY_DIR}/external/gflags-install)
    # ----------------------------------------------------------------
    # 如果未指定上述任何选项，则按如下方式计算它们的默认值。
    # 如果给定了该选项或设置了目录属性，则将在指定的前缀下生成和安装外部项目
    # TMP_DIR      = <prefix>/tmp
    # STAMP_DIR    = <prefix>/src/<name>-stamp
    # DOWNLOAD_DIR = <prefix>/src
    # SOURCE_DIR   = <prefix>/src/<name>
    # BINARY_DIR   = <prefix>/src/<name>-build
    # INSTALL_DIR  = <prefix>
    # LOG_DIR      = <STAMP_DIR>
    # ----------------------------------------------------------------


    # we build gflags statically, but want to link it into the caffe shared library
    # this requires position-independent code
    # 动态链接库
    if (UNIX)
        set(GFLAGS_EXTRA_COMPILER_FLAGS "-fPIC")
    endif()

    set(GFLAGS_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${GFLAGS_EXTRA_COMPILER_FLAGS})
    set(GFLAGS_C_FLAGS ${CMAKE_C_FLAGS} ${GFLAGS_EXTRA_COMPILER_FLAGS})

    # ExternalProject_Add()函数创建一个外部工程可以驱动下载、更新/补丁、配置、构建、安装和测试流程的自定义目标；
    ExternalProject_Add(gflags
      PREFIX ${gflags_PREFIX} # 外部项目的根目录
      GIT_REPOSITORY "https://github.com/gflags/gflags.git"  # git存储库的url
      GIT_TAG "v2.1.2"  # git分支名称、标记
      UPDATE_COMMAND ""
      INSTALL_DIR ${gflags_INSTALL}
      CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                 -DCMAKE_INSTALL_PREFIX=${gflags_INSTALL}
                 -DBUILD_SHARED_LIBS=OFF
                 -DBUILD_STATIC_LIBS=ON
                 -DBUILD_PACKAGING=OFF
                 -DBUILD_TESTING=OFF
                 -DBUILD_NC_TESTS=OFF
                 -BUILD_CONFIG_TESTS=OFF
                 -DINSTALL_HEADERS=ON
                 -DCMAKE_C_FLAGS=${GFLAGS_C_FLAGS}
                 -DCMAKE_CXX_FLAGS=${GFLAGS_CXX_FLAGS}
      LOG_DOWNLOAD 1
      LOG_INSTALL 1
      )

    set(GFLAGS_FOUND TRUE)
    set(GFLAGS_INCLUDE_DIRS ${gflags_INSTALL}/include)
    set(GFLAGS_LIBRARIES ${gflags_INSTALL}/lib/libgflags.a ${CMAKE_THREAD_LIBS_INIT})
    set(GFLAGS_LIBRARY_DIRS ${gflags_INSTALL}/lib)
    set(GFLAGS_EXTERNAL TRUE)

    list(APPEND external_project_dependencies gflags)
  endif()

endif()
