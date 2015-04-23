cmake_minimum_required(VERSION 2.8)
PROJECT(vid_works)
#SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_BUILD_TYPE "Release")
FIND_PACKAGE(OpenCV REQUIRED)
FIND_PACKAGE(Threads REQUIRED) #for Cimg
FIND_PACKAGE(X11) # for Cimg
FIND_PACKAGE(Boost REQUIRED date_time program_options thread filesystem system ) 

include_directories(${X11_INCLUDE_DIRS}) #for Cimg
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR})

SET(all_srcs main.cpp)
ADD_EXECUTABLE(vid_works ${all_srcs})
TARGET_LINK_LIBRARIES(vid_works ${OpenCV_LIBS})
TARGET_LINK_LIBRARIES(vid_works ${CMAKE_THREAD_LIBS_INIT} ${X11_LIBRARIES}) #cimg
TARGET_LINK_LIBRARIES(vid_works ${Boost_SYSTEM_LIBRARIES} ${Boost_FILESYSTEM_LIBRARIES})
