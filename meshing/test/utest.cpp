/*
 * Copyright (c) 2016, Southwest Research Institute
 * All rights reserved.
 *
 */

#include <meshing/afront_meshing.h>
#include <boost/thread/thread.hpp>
#include <pcl/visualization/pcl_visualizer.h>
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

  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);

  std::cout << "starting test\n";
  boost::shared_ptr<afront_meshing::AfrontMeshing> mesher(new afront_meshing::AfrontMeshing);
  pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud(new pcl::PointCloud<pcl::PointXYZ>(cloud));
  std::cout << "number of cloud points: " << in_cloud->points.size() << "\n";
  mesher->setInputCloud(in_cloud);
  mesher->setRho(0.125);
  mesher->setTriangleQuality(1.2);
  mesher->setRadius(1.0);
  pcl::PolygonMesh out_mesh;

  if(mesher->computeGuidanceField())
  {
    mesher->startMeshing();
    out_mesh = mesher->getMesh();
    viewer->addPolygonMesh(out_mesh);
  }
  else
  {
    std::cout << "failed to generate guidance field\n";
  }

  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(in_cloud, 0, 255, 0);
  viewer->addPointCloud<pcl::PointXYZ> (in_cloud, single_color, "sample cloud");

  viewer->addCoordinateSystem(1.0);
  viewer->initCameraParameters();

  boost::function<void (const pcl::visualization::KeyboardEvent&)> keyboardEventOccurred = [viewer, mesher] (const pcl::visualization::KeyboardEvent &event)
  {
    if (event.getKeySym () == "n" && event.keyDown ())
    {
      pcl::PolygonMesh out_mesh;
      mesher->stepMesh();
      out_mesh = mesher->getMesh();
      viewer->removePolygonMesh();
      viewer->addPolygonMesh(out_mesh);
    }
  };

  viewer->registerKeyboardCallback(keyboardEventOccurred);
  viewer->spin();
}

// Run all the tests that were declared with TEST()
int main(int argc, char **argv)
{
  //ros::init(argc, argv, "test");  // some tests need ROS framework
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
