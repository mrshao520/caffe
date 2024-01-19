# - Try to find GFLAGS
#
# The following variables are optionally searched for defaults
#  GFLAGS_ROOT_DIR:            Base directory where all GFLAGS components are found
#
# The following are set after configuration is done:
#  GFLAGS_FOUND
#  GFLAGS_INCLUDE_DIRS
#  GFLAGS_LIBRARIES
#  GFLAGS_LIBRARYRARY_DIRS


########################################################
# - find_path (<VAR> name1 [path1 path2 ...])
# 该命令用于查找包含指定文件的目录。创建一个高速缓存条目，或者如果指定了 NO_CACHE ，则创建一个普通变量，
# 以 <VAR> 命名，以存储该命令的结果。如果找到目录中的文件，结果将存储在变量中，除非清除变量，否则不会重复搜索。
# 如果未找到任何内容，结果将为 <VAR>-NOTFOUND 。
#
# - find_library (<VAR> name1 [path1 path2 ...])
# 与find_path类似

include(FindPackageHandleStandardArgs)

set(GFLAGS_ROOT_DIR "" CACHE PATH "Folder contains Gflags")

# We are testing only a couple of files in the include directories
if(WIN32)
    find_path(GFLAGS_INCLUDE_DIR gflags/gflags.h
        PATHS ${GFLAGS_ROOT_DIR}/src/windows)
else()
    find_path(GFLAGS_INCLUDE_DIR gflags/gflags.h
        PATHS ${GFLAGS_ROOT_DIR})
endif()

message(STATUS "GFLAGS_INCLUDE_DIR : ${GFLAGS_INCLUDE_DIR}")


if(MSVC)
    find_library(GFLAGS_LIBRARY_RELEASE
        NAMES libgflags
        PATHS ${GFLAGS_ROOT_DIR}
        PATH_SUFFIXES Release)

    find_library(GFLAGS_LIBRARY_DEBUG
        NAMES libgflags-debug
        PATHS ${GFLAGS_ROOT_DIR}
        PATH_SUFFIXES Debug)

    set(GFLAGS_LIBRARY optimized ${GFLAGS_LIBRARY_RELEASE} debug ${GFLAGS_LIBRARY_DEBUG})
else()
    find_library(GFLAGS_LIBRARY gflags)
endif()

message(STATUS "GFLAGS_LIBRARY : ${GFLAGS_LIBRARY}")


#########################################################
# 包含了标准FindPackageHandleStandardArgs.cmake，并调用相应的CMake命令。
# 如果找到所有需要的变量，并且版本匹配，则将GFLAGS_FOUND变量设置为TRUE:
find_package_handle_standard_args(GFlags DEFAULT_MSG GFLAGS_INCLUDE_DIR GFLAGS_LIBRARY)


if(GFLAGS_FOUND)
    set(GFLAGS_INCLUDE_DIRS ${GFLAGS_INCLUDE_DIR})
    set(GFLAGS_LIBRARIES ${GFLAGS_LIBRARY})
    message(STATUS "Found gflags  (include: ${GFLAGS_INCLUDE_DIR}, library: ${GFLAGS_LIBRARY})")

    # 将 cmake 缓存变量标记为高级。
    # 传递给此命令但尚未在缓存中的变量将被忽略
    mark_as_advanced(GFLAGS_LIBRARY_DEBUG GFLAGS_LIBRARY_RELEASE
                     GFLAGS_LIBRARY GFLAGS_INCLUDE_DIR GFLAGS_ROOT_DIR)
endif()
