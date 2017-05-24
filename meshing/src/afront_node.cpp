#include <ros/ros.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <meshing/afront_meshing.h>

int main(int argc, char **argv)
{
  ros::init(argc, argv, "afront_node");
  ros::NodeHandle nh("~");
  std::string pcd_file;
  std::string ply_file;
  double rho;
  double quality;
  double radius;
  bool snap;

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
  nh.param<double>("quality", quality, 1.2);
  if (quality < 1)
  {
    ROS_ERROR("Quality must be greater than or equal to 1.");
    return 0;
  }

  pcl::PointCloud<pcl::PointXYZ> cloud;
  afront_meshing::AfrontMeshing mesher;
  if (pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_file, cloud) == -1) //* load the file
  {
    ROS_ERROR("Couldn't read pcd file!");
    return 0;
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud(new pcl::PointCloud<pcl::PointXYZ>(cloud));
  mesher.setRho(rho);
  mesher.setTriangleQuality(quality);
  mesher.setRadius(radius);
  mesher.enableSnap(snap);

  if(mesher.initMesher(in_cloud))
    mesher.generateMesh();

  pcl::io::savePLYFile(ply_file, mesher.getMesh());

  return 0;
}
