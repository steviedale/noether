/*
 * Copyright (c) 2016, Southwest Research Institute
 * All rights reserved.
 *
 */

#include <meshing/afront_meshing.h>
#include <meshing/afront_utils.h>
#include <boost/thread/thread.hpp>
#include <pcl/io/pcd_io.h>
#include <vtkMath.h>
#include <gtest/gtest.h>

TEST(AfrontUtilsTest, distPoint2Line)
{
  using afront_meshing::utils::DistPoint2LineResults;
  using afront_meshing::utils::distPoint2Line;
  Eigen::Vector3f lp1(0.0, 0.0, 0.0);
  Eigen::Vector3f lp2(1.0, 0.0, 0.0);
  Eigen::Vector3f p(0.5, 1.0, 0.0);
  DistPoint2LineResults results = distPoint2Line(lp1, lp2, p);
  EXPECT_FLOAT_EQ(results.d, 1.0);
  EXPECT_FLOAT_EQ(results.mu, 0.5);
  EXPECT_TRUE(results.p.isApprox(Eigen::Vector3f(0.5, 0.0, 0.0), 1e-10));
}

TEST(AfrontUtilsTest, distPoint2LineLeftBound)
{
  using afront_meshing::utils::DistPoint2LineResults;
  using afront_meshing::utils::distPoint2Line;
  Eigen::Vector3f lp1(0.0, 0.0, 0.0);
  Eigen::Vector3f lp2(1.0, 0.0, 0.0);
  Eigen::Vector3f p(-0.5, 1.0, 0.0);
  DistPoint2LineResults results = distPoint2Line(lp1, lp2, p);
  EXPECT_FLOAT_EQ(results.d, std::sqrt(1.0 * 1.0 + 0.5 * 0.5));
  EXPECT_FLOAT_EQ(results.mu, 0.0);
  EXPECT_TRUE(results.p.isApprox(lp1, 1e-10));
}

TEST(AfrontUtilsTest, distPoint2LineRightBound)
{
  using afront_meshing::utils::DistPoint2LineResults;
  using afront_meshing::utils::distPoint2Line;
  Eigen::Vector3f lp1(0.0, 0.0, 0.0);
  Eigen::Vector3f lp2(1.0, 0.0, 0.0);
  Eigen::Vector3f p(1.5, 1.0, 0.0);
  DistPoint2LineResults results = distPoint2Line(lp1, lp2, p);
  EXPECT_FLOAT_EQ(results.d, std::sqrt(1.0 * 1.0 + 0.5 * 0.5));
  EXPECT_FLOAT_EQ(results.mu, 1.0);
  EXPECT_TRUE(results.p.isApprox(lp2, 1e-10));
}

TEST(AfrontUtilsTest, distLine2Line)
{
  using afront_meshing::utils::DistLine2LineResults;
  using afront_meshing::utils::distLine2Line;

  Eigen::Vector3f l1[2], l2[2];

  l1[0] << 0.0, 0.0, 0.0;
  l1[1] << 1.0, 0.0, 0.0;

  l2[0] << 0.0, -0.5, 0.5;
  l2[1] << 0.0, 0.5, 0.5;


  DistLine2LineResults results = distLine2Line(l1[0], l1[1], l2[0], l2[1]);
  EXPECT_FLOAT_EQ(results.mu[0], 0.0);
  EXPECT_FLOAT_EQ(results.mu[1], 0.5);

  EXPECT_TRUE(results.p[0].isApprox(l1[0], 1e-10));
  EXPECT_TRUE(results.p[1].isApprox(Eigen::Vector3f(0.0, 0.0, 0.5), 1e-10));
  EXPECT_FALSE(results.parallel);
}

TEST(AfrontUtilsTest, distLine2LineParallel)
{
  using afront_meshing::utils::DistLine2LineResults;
  using afront_meshing::utils::distLine2Line;

  Eigen::Vector3f l1[2], l2[2];

  l1[0] << 0.0, 0.0, 0.0;
  l1[1] << 1.0, 0.0, 0.0;

  l2[0] << -0.5, 0.0, 0.5;
  l2[1] << 0.5, 0.0, 0.5;


  DistLine2LineResults results = distLine2Line(l1[0], l1[1], l2[0], l2[1]);
  EXPECT_FLOAT_EQ(results.mu[0], 0.0);
  EXPECT_FLOAT_EQ(results.mu[1], 0.5);

  EXPECT_TRUE(results.p[0].isApprox(l1[0], 1e-10));
  EXPECT_TRUE(results.p[1].isApprox(Eigen::Vector3f(0.0, 0.0, 0.5), 1e-10));
  EXPECT_TRUE(results.parallel);
}

TEST(AfrontUtilsTest, intersectionLine2Plane)
{
  using afront_meshing::utils::IntersectionLine2PlaneResults;
  using afront_meshing::utils::intersectionLine2Plane;

  Eigen::Vector3f l[2], u, v, origin;
  l[0] << 0.5, 0.5, -0.5;
  l[1] << 0.5, 0.5, 0.5;

  origin << 0.0, 0.0, 0.0;
  u << 1.0, 0.0, 0.0;
  v << 0.0, 1.0, 0.0;

  IntersectionLine2PlaneResults results = intersectionLine2Plane(l[0], l[1], origin, u, v);

  EXPECT_FLOAT_EQ(results.mu, 0.5);
  EXPECT_FLOAT_EQ(results.mv, 0.5);
  EXPECT_FLOAT_EQ(results.mw, 0.5);
  EXPECT_TRUE(results.p.isApprox(Eigen::Vector3f(0.5, 0.5, 0.0), 1e-10));
  EXPECT_FALSE(results.parallel);
}

TEST(AfrontUtilsTest, intersectionLine2PlaneParallel)
{
  using afront_meshing::utils::IntersectionLine2PlaneResults;
  using afront_meshing::utils::intersectionLine2Plane;

  Eigen::Vector3f l[2], u, v, origin;
  l[0] << 0.0, 0.0, 0.5;
  l[1] << 1.0, 0.0, 0.5;

  origin << 0.0, 0.0, 0.0;
  u << 1.0, 0.0, 0.0;
  v << 0.0, 1.0, 0.0;

  IntersectionLine2PlaneResults results = intersectionLine2Plane(l[0], l[1], origin, u, v);

  EXPECT_TRUE(results.parallel);
}

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

  std::cout << "starting test\n";
  afront_meshing::AfrontMeshing mesher;
  pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud(new pcl::PointCloud<pcl::PointXYZ>(cloud));
  std::cout << "number of cloud points: " << in_cloud->points.size() << "\n";

  mesher.setRho(0.5);
  mesher.setReduction(0.8);
  mesher.setSearchRadius(1);
  mesher.setNumberOfThreads(1);
  mesher.setInputCloud(in_cloud);

  if(!mesher.initialize())
    std::cout << "failed to initialize mesher\n";

  mesher.reconstruct();
}

// Run all the tests that were declared with TEST()
int main(int argc, char **argv)
{
  //ros::init(argc, argv, "test");  // some tests need ROS framework
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
