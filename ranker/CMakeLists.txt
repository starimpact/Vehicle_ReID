cmake_minimum_required(VERSION 2.8)
project(ranker)

set(VERSION 0.1.0)

include(/usr/share/cmake-2.8/Modules/FindCUDA.cmake)

add_compile_options("-std=c++11;-O3")
set(CUDA_NVCC_FLAGS -arch=sm_35;-std=c++11)

include_directories(/usr/include/opencv /usr/include/opencv2)
link_directories(/usr/lib/x86_64-linux-gnu /usr/local/cuda/lib64)
set(LIBS pthread boost_thread boost_system cuda cudart opencv_imgproc opencv_video opencv_highgui opencv_core opencv_photo)

set(EXE_SRC main.cpp)
set(LIB_SRC database.cu do_ranker.cpp)

set(EXE_TGT test)
set(LIB_TGT ranker)

cuda_add_library(${LIB_TGT} SHARED ${LIB_SRC})
target_link_libraries(${LIB_TGT} ${LIBS})


add_executable(${EXE_TGT} ${EXE_SRC})
target_link_libraries(${EXE_TGT} ${LIBS} ranker)


