cmake_minimum_required(VERSION 2.8)
project(test-ncs)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
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

# GFLAGS
find_library(GFLAGS_LIB NAMES gflags)
find_path(GFLAGS_INCLUDES NAMES gflags.h PATHS /usr/include/gflags/)
message("GFLAGS INCLUDES: ${GFLAGS_INCLUDES}")
include_directories(SYSTEM ${GFLAGS_INCLUDES})

# ---[ Find include dir
find_path(NCS_INCLUDE_DIR NAMES "mvnc.h" HINTS "/opt/movidius-v2/NCSDK/ncsdk-x86_64/api/c/" PATHS "/opt/movidius-v2/NCSDK/ncsdk-x86_64/api/c/" )

message("NCS includes: ${NCS_INCLUDE_DIR}")

find_library(NCS_LIBRARY
        NAMES mvnc
        PATHS "/opt/movidius-v2/NCSDK/ncsdk-x86_64/api/c/")

message("NCS LIBRARY: ${NCS_LIBRARY}")

if(NCS_LIBRARY)
  message(STATUS "Found Movidius NCS (include: ${NCS_INCLUDE_DIR}, lib: ${NCS_LIBRARY}")
  include_directories(${NCS_INCLUDE_DIR})
else()
  message(FATAL " Intel NCS not located properly")
endif()

#OpenCV
find_package( OpenCV REQUIRED )
#set(OpenCV_INCLUDE_DIRS "/usr/local/include/opencv2/")
#set(OpenCV_LIBS "/usr/local/lib/libopencv_core.so")
include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})
message(STATUS "OpenCV found (${OpenCV_CONFIG_PATH})")

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
add_executable(test-ncs-v2 main.cpp)

set(CMAKE_EXE_LINKER_FLAGS -Wl,-rpath,/usr/local/lib64)

add_definitions(-Wall)

target_link_libraries(test-ncs-v2 ${NCS_LIBRARY} ${OpenCV_LIBS} ${GFLAGS_LIB})
