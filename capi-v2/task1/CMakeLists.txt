cmake_minimum_required(VERSION 2.8)
project(task1)

include(CheckCXXCompilerFlag)
if(UNIX)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
add_definitions(-std=c++11)
elseif(COMPILER_SUPPORTS_CXX0X)
add_definitions(-std=c++0x)
endif()
endif()

# ---[ Find include dir
find_path(NCS_INCLUDE_DIR NAMES "mvnc.h" HINTS "/usr/local" PATHS "/usr/local" PATH_SUFFIXES "include" )

find_library(NCS_LIBRARY
        NAMES mvnc
        PATHS /usr/local/lib)

if(NCS_LIBRARY)
  message(STATUS "Found Movidius NCS (include: ${NCS_INCLUDE_DIR}, lib: ${NCS_LIBRARY}")
  include_directories(${NCS_INCLUDE_DIR})
else()
  message(FATAL " Intel NCS not located properly")
endif()

add_executable(task1 main.cpp )

#set(CMAKE_EXE_LINKER_FLAGS -Wl,-rpath,/usr/local/lib64)

target_link_libraries(task1 ${NCS_LIBRARY} )
