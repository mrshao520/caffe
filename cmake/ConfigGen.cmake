
################################################################################################
# Helper function to get all list items that begin with given prefix
# Usage:
#   caffe_get_items_with_prefix(<prefix> <list_variable> <output_variable>)
function(caffe_get_items_with_prefix prefix list_variable output_variable)
  set(__result "")
  foreach(__e ${${list_variable}})
    if(__e MATCHES "^${prefix}.*")
      list(APPEND __result ${__e})
    endif()
  endforeach()
  set(${output_variable} ${__result} PARENT_SCOPE)
endfunction()

################################################################################################
# Function for generation Caffe build- and install- tree export config files
# 用于生成Caffe构建和安装树导出配置文件的函数
# Usage:
#  caffe_generate_export_configs()
function(caffe_generate_export_configs)
  set(install_cmake_suffix "share/Caffe")

  if(NOT HAVE_CUDA)
    set(HAVE_CUDA FALSE)
  endif()

  set(HDF5_IMPORTED OFF)
  foreach(_lib ${HDF5_LIBRARIES} ${HDF5_HL_LIBRARIES})
    if(TARGET ${_lib})
      set(HDF5_IMPORTED ON)
    endif()
  endforeach()

  # This code is taken from https://github.com/sh1r0/caffe-android-lib
  if(USE_HDF5)
    list(APPEND Caffe_DEFINITIONS -DUSE_HDF5)
  endif()

  if(NOT HAVE_CUDNN)
    set(HAVE_CUDNN FALSE)
  endif()

  # ---[ Configure build-tree CaffeConfig.cmake file ]---
  configure_file("cmake/Templates/CaffeConfig.cmake.in" "${PROJECT_BINARY_DIR}/CaffeConfig.cmake" @ONLY)

  # Add targets to the build-tree export set
  # 导出外部项目的目标或包，以便直接从当前项目的构建树中使用它们，而无需安装。

  # export(TARGETS <target>... [...])
  # 创建一个可能由外部项目包含的文件 <filename> ，以从当前项目的构建树导入由 <target>... 命名的目标。
  export(TARGETS caffe caffeproto FILE "${PROJECT_BINARY_DIR}/CaffeTargets.cmake")

  # 将当前构建目录存储在包 <PackageName> 的 CMake 用户包注册表中。 
  # find_package() 命令在搜索包 <PackageName> 时可能会考虑该目录。
  # 这有助于依赖项目从当前项目的构建树中找到并使用包，而无需用户的帮助。
  # 请注意，此命令在包注册表中创建的条目只能与与构建树一起使用的包配置文件 ( <PackageName>Config.cmake ) 结合使用。
  export(PACKAGE Caffe)

  # ---[ Configure install-tree CaffeConfig.cmake file ]---
  configure_file("cmake/Templates/CaffeConfig.cmake.in" "${PROJECT_BINARY_DIR}/cmake/CaffeConfig.cmake" @ONLY)

  # Install the CaffeConfig.cmake and export set to use with install-tree
  install(FILES "${PROJECT_BINARY_DIR}/cmake/CaffeConfig.cmake" DESTINATION ${install_cmake_suffix})
  install(EXPORT CaffeTargets DESTINATION ${install_cmake_suffix})

  # ---[ Configure and install version file ]---

  # TODO: Lines below are commented because Caffe doesn't declare its version in headers.
  # When the declarations are added, modify `caffe_extract_caffe_version()` macro and uncomment

  # configure_file(cmake/Templates/CaffeConfigVersion.cmake.in "${PROJECT_BINARY_DIR}/CaffeConfigVersion.cmake" @ONLY)
  # install(FILES "${PROJECT_BINARY_DIR}/CaffeConfigVersion.cmake" DESTINATION ${install_cmake_suffix})
endfunction()


