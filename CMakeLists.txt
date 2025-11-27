cmake_minimum_required(VERSION 3.05)

project(DlnneTest)

set(dl_include_root /dl/DlDeepToolkit-master/c++)

# 封装库的安装路径
set(DLHELPER_ROOT_DIR ${dl_include_root}/install)
set(DLHELPER_INCLUDE_DIR ${DLHELPER_ROOT_DIR}/include)
set(DLHELPER_LIB_DIR ${DLHELPER_ROOT_DIR}/lib)
set(DLHELPER_LIB dldevice dlvidhelper dlnnehelper dlpreprocess)

# FFMPEG dev
set(FFMPEG_ROOT_DIR ${dl_include_root}/3rdparty/ffmpeg-4.3.1)
set(FFMPEG_INCLUDE_DIR ${FFMPEG_ROOT_DIR}/include)
set(FFMPEG_LIB_DIR ${FFMPEG_ROOT_DIR}/lib)
set(FFMPEG_LIB avcodec avformat avutil swresample)

# x264 dev
set(X264_ROOT_DIR ${dl_include_root}/3rdparty/x264)
set(X264_INCLUDE_DIR ${X264_ROOT_DIR}/include)
set(X264_LIB_DIR ${X264_ROOT_DIR}/lib)
set(X264_LIB x264)

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# 指定头文件目录
include_directories(${DLHELPER_INCLUDE_DIR})
include_directories(${FFMPEG_INCLUDE_DIR})
include_directories(${X264_INCLUDE_DIR})
set(SDK_DIR $ENV{DLICC_PATH}/../)
include_directories(${SDK_DIR}/include)
include_directories(${SDK_DIR}/include/dlnne)
include_directories(${dl_include_root})

# 根据需要设置相应的编译选项
set(CXX_FLAGS -std=c++11 -fpermissive -Wno-attributes)
# 兼容老版本gcc
add_compile_options(-std=c++11)

# 添加可执行目标
link_directories(${SDK_DIR}/lib ${DLHELPER_LIB_DIR} ${FFMPEG_LIB_DIR} ${X264_LIB_DIR})
set(CMAKE_INSTALL_RPATH ${DLHELPER_LIB_DIR} ${FFMPEG_LIB_DIR} ${X264_LIB_DIR})
set(DESTINATION /dl/DlDeepToolkit-master/c++/test/dlnne/install/)

# compile all test
# test1
file(GLOB dlnne_tests "test.cc")
file(GLOB_RECURSE cc_files "dlnne_algo_unit_yolov5.cc")

foreach(test ${dlnne_tests})
    string(REGEX REPLACE ".+/(.+)\\..*" "\\1" exe_name ${test})
    add_executable(
        ${exe_name} ${test} ${cc_files}
    )
    target_compile_options(${exe_name} PRIVATE -I${SDK_DIR}/include -I${CMAKE_BINARY_DIR} ${CXX_FLAGS})
    target_link_libraries(${exe_name} PRIVATE curt pthread b64
                          ${DLHELPER_LIB} ${FFMPEG_LIB} ${X264_LIB} ${OpenCV_LIBS} "-Wl,--disable-new-dtags")
    install(TARGETS ${exe_name} DESTINATION ${DESTINATION})
endforeach()
