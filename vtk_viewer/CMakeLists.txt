cmake_minimum_required(VERSION 2.8.3)
project(vtk_viewer)

if(CMAKE_VERSION VERSION_LESS 3.1)
    add_compile_options(-std=c++11)
else()
    set(CMAKE_CXX_STANDARD 11)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS ON)
endif()

find_package(catkin REQUIRED cmake_modules meshing)

find_package(VTK 7.1 REQUIRED NO_MODULE)
include(${VTK_USE_FILE})

find_package(PCL 1.8 REQUIRED)

find_package(Eigen3 REQUIRED)

find_package(OpenMP)
if(OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else(OPENMP_FOUND)
  message (STATUS "Not found OpenMP")
endif()

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES vtk_viewer
  DEPENDS
    Eigen
    PCL
    VTK
)

include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    ${EIGEN_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
)

add_library(vtk_viewer
    src/vtk_viewer.cpp
    src/vtk_utils.cpp
    src/mouse_interactor.cpp
)

target_link_libraries(vtk_viewer
    ${catkin_LIBRARIES}
    ${PCL_LIBRARIES}
    ${VTK_LIBRARIES}
)

catkin_add_gtest(${PROJECT_NAME}-test test/utest.cpp)
if(TARGET ${PROJECT_NAME}-test)
    target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
endif()

#############
## Install ##
#############
install(
  DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION})

install(
  TARGETS 
    vtk_viewer
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
