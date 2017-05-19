/*
 * Copyright (c) 2016, Southwest Research Institute
 * All rights reserved.
 *
 */

#include <meshing/afront_meshing.h>
#include <boost/thread/thread.hpp>
#include <pcl/io/pcd_io.h>
#include <vtkMath.h>
#include <gtest/gtest.h>

// This test shows the results of meshing on a square grid that has a sinusoidal
// variability in the z axis.  Red arrows show the surface normal for each triangle
// in the mesh, and cyan boxes show the points used to seed the mesh creation algorithm

TEST(AfrontTest, TestCase1)
{
  pcl::PointCloud<pcl::PointXYZ> cloud;
  int gridSize = 50;
  sleep(3);

  for(unsigned int x = 0; x < gridSize; x++)
  {
    for(unsigned int y = 0; y < gridSize; y++)
    {
      pcl::PointXYZ pt(x / 10.0  , y / 10.0 , 0.5 * cos(double(x)/10.0) - 0.5 * sin(double(y)/10.0) + vtkMath::Random(0.0, 0.001));
      cloud.push_back(pt);
    }
  }
  cloud.is_dense = false;
//  if (pcl::io::loadPCDFile<pcl::PointXYZ>("/home/larmstrong/Downloads/godel_point_cloud_data/test.pcd", cloud) == -1) //* load the file
//  {
//    PCL_ERROR ("Couldn't read file test_pcd.pcd \n");
//    return;
//  }

  std::cout << "starting test\n";
  afront_meshing::AfrontMeshing mesher;
  pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud(new pcl::PointCloud<pcl::PointXYZ>(cloud));
  std::cout << "number of cloud points: " << in_cloud->points.size() << "\n";

  mesher.setRho(0.05);
  mesher.setTriangleQuality(1.2);
  mesher.setRadius(1);
  mesher.enableSnap(false);

  if(!mesher.initMesher(in_cloud))
    std::cout << "failed to initialize mesher\n";

}

// Run all the tests that were declared with TEST()
int main(int argc, char **argv)
{
  //ros::init(argc, argv, "test");  // some tests need ROS framework
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
