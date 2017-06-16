#include <ros/ros.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/filters/filter.h>
#include <meshing/afront_meshing.h>

int main(int argc, char **argv)
{
  ros::init(argc, argv, "afront_node");
  ros::NodeHandle nh("~");
  std::string pcd_file;
  std::string ply_file;
  double rho;
  double reduction;
  double radius;
  bool snap;
  int sample;

  if (nh.hasParam("pcd_file"))
  {
    nh.param<std::string>("pcd_file", pcd_file, "");
  }
  else
  {
    ROS_ERROR("Must provide a input pcd file!");
    return 0;
  }
  ROS_INFO("%s", pcd_file.c_str());


  if (nh.hasParam("ply_file"))
  {
    nh.param<std::string>("ply_file", ply_file, "");
  }
  else
  {
    ROS_ERROR("Must provide a output ply file!");
    return 0;
  }
  ROS_INFO("%s", ply_file.c_str());

  nh.param<double>("rho", rho, 0.5);
  nh.param<double>("radius", radius, 0.1);
  nh.param<bool>("snap", snap, false);
  nh.param<int>("sample", sample, 0);
  nh.param<double>("reduction", reduction, 0.8);
  if (reduction >= 1 || reduction <= 0)
  {
    ROS_ERROR("Reduction must be (0 < reduction < 1)");
    return 0;
  }

  pcl::PointCloud<pcl::PointXYZ> cloud, filtered_cloud;
  afront_meshing::AfrontMeshing mesher;
  if (pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_file, cloud) == -1) //* load the file
  {
    ROS_ERROR("Couldn't read pcd file!");
    return 0;
  }

  std::vector<int> indices;
  pcl::removeNaNFromPointCloud(cloud, filtered_cloud, indices);

  pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud(new pcl::PointCloud<pcl::PointXYZ>(filtered_cloud));
  mesher.setRho(rho);
  mesher.setReduction(reduction);
  mesher.setRadius(radius);
  mesher.enableSnap(snap);

  sleep(8);
  if (mesher.initMesher(in_cloud))
  {
    if (sample == 0)
      mesher.generateMesh();
    else
      for (auto i = 0; i < sample; ++i)
        mesher.stepMesh();

    pcl::io::savePLYFile(ply_file, mesher.getMesh());
  }

  return 0;
}
