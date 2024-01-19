########################################################################################################
# 判断平台的两种方式
message(STATUS "operation system is ${CMAKE_SYSTEM}")

if(CMAKE_SYSTEM_NAME MATCHES "Linux")
    message(STATUS "current platform: Linux")
elseif(CMAKE_SYSTEM_NAME MATHCES "Windows")
    message(STATUS "current platform: Windows")
elseif(CMAKE_SYSTEM_NAME MATCHES "FreeBSD")
    message(STATUS "current platform: FreeBSD")
else()
    message(STATUS "other platform: ${CMAKE_SYSTEM_NAME}")
endif()


IF (WIN32)
	MESSAGE(STATUS "Now is windows")
ELSEIF (APPLE)
	MESSAGE(STATUS "Now is Apple systens.")
ELSEIF (UNIX)
	MESSAGE(STATUS "Now is UNIX-like OS's.")
ENDIF ()

########################################################################################################
# An option that the user can select. Can accept condition to control when option is available for user.
# Usage:
#   test_option(<option_variable> "doc string" <initial value or boolean expression> [IF <condition>])
# Example:
#   test_option(TEST_CUDA "Build project with CUDA" ON if ON)
#       test_option description : Build project with CUDA
#       [build] -- test_option value : ON
#       [build] -- test_option ARGN : if;ON
function(test_option variable description value)
    message(STATUS "--------------- test option ------------------")
    message(STATUS "test_option description : ${description}")
    message(STATUS "test_option value : ${value}")
    message(STATUS "test_option ARGN : ${ARGN}")
    set(__value ${value})
    set(__condition "")
    set(__varname "__value")
    foreach(arg ${ARGN})
        if(arg STREQUAL "IF" OR arg STREQUAL "if")
            set(__varname "__condition")
        else()
            list(APPEND ${__varname} ${arg})
        endif()
    endforeach(arg ${ARGN})
    message(STATUS "test_option __varname : ${__varname}")
    unset(__varname)
    if("${__condition}" STREQUAL "")
        set(__condition 2 GREATER 1)
    endif()
    message(STATUS "test_option __condition : ${__condition}")
    message(STATUS "test_option __value : ${__value}")


    if(${__condition})
        if("${__value}" MATCHES ";")
            if(${__value})
                option(${variable} "${description}" ON)
            else()
                option(${variable} "${description}" OFF)
            endif()
        elseif(DEFINED ${__value})
            if(${__value})
                option(${variable} "${description}" ON)
            else()
                option(${variable} "${description}" OFF)
            endif()
        else()
            option(${variable} "${description}" ${__value})
        endif()
    else()
        unset(${variable} CACHE)
        message(STATUS "test_option unset variable")
    endif()
    
endfunction(test_option variable description value)

