cmake_minimum_required(VERSION 2.8)

project(augment)

set(VERSION 0.1.0)

add_compile_options("-std=c++11")

include_directories(/usr/include/opencv /usr/include/opencv2)
#link_directories(/usr/local/lib /Users/starimpact/work/augmentation_threads/build)
link_directories(/usr/lib/x86_64-linux-gnu)
set(LIBS opencv_imgproc opencv_video opencv_highgui opencv_core opencv_photo)

#file(GLOB_RECURSE SRCS *.cc)
set(EXE_SRC main.cc)
set(LIB_SRC augmentor.cc)

set(EXE_TGT hello)
set(LIB_TGT augment)
add_executable(${EXE_TGT} ${EXE_SRC})

add_library(${LIB_TGT} SHARED ${LIB_SRC})
target_link_libraries(${LIB_TGT} ${LIBS})

target_link_libraries(${EXE_TGT} ${LIBS} augment)





