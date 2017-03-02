cmake_minimum_required(VERSION 2.8.3)
project(meshing)

find_package(catkin REQUIRED cmake_modules)

find_package(PCL 1.8 REQUIRED)

find_package(Eigen3 REQUIRED)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES meshing
  DEPENDS
    Eigen
    PCL

)

include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    ${EIGEN_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
)

add_library(meshing
    src/afront_meshing.cpp

)

target_link_libraries(meshing
    ${catkin_LIBRARIES}
    ${PCL_LIBRARIES}

)

#catkin_add_gtest(${PROJECT_NAME}-test test/utest.cpp)
#if(TARGET ${PROJECT_NAME}-test)
#    target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
#endif()
