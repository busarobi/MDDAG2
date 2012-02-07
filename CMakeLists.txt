#The following command allows the use of the "file" command
cmake_minimum_required(VERSION 2.6)

#The declaration of the project
project(mddag2)

set(BASEPATH "${CMAKE_SOURCE_DIR}/src")

#This allows recursive parsing of the source files
# Including sub-directories recursively
file(GLOB_RECURSE project_SRCS "${BASEPATH}/*" )

# By Default AlphaReal is double and FeatureReal is float
# uncomment those lines to change that
#ADD_DEFINITIONS( "-DFEATUREREALDOUBLE" )
#ADD_DEFINITIONS( "-DALPHAREALFLOAT" )

#set(CMAKE_INCLUDE_PATH ./)
include_directories("${BASEPATH}" "/sps/appstat/busarobi/MDDAG2/TBB/tbb40_20111130oss/include")

link_directories("/sps/appstat/busarobi/MDDAG2/TBB/tbb40_20111130oss/lib/intel64/cc4.1.0_libc2.4_kernel2.6.16.21")

#This indicates the target (the executable)
add_executable(mddag2 ${project_SRCS})

target_link_libraries("tbb")

