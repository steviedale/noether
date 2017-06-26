#include <ros/ros.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/filters/filter.h>
#include <meshing/afront_meshing.h>

int main(int argc, char **argv)
{
  ros::init(argc, argv, "afront_node");
  ros::NodeHandle nh("~");
  std::string input_file;
  std::string output_file;
  double rho;
  double reduction;
  double radius;
  int sample;
  int threads;
  std::string extension;

  if (nh.hasParam("input_file"))
  {
    nh.param<std::string>("input_file", input_file, "");
    extension = boost::filesystem::extension(input_file);
    if (extension != ".pcd" && extension != ".ply")
    {
      ROS_ERROR("Only file types supported are pcd and ply.");
      return 0;
    }
  }
  else
  {
    ROS_ERROR("Must provide a input file!");
    return 0;
  }

  if (nh.hasParam("output_file"))
  {
    nh.param<std::string>("output_file", output_file, "");
  }
  else
  {
    ROS_ERROR("Must provide a output ply file!");
    return 0;
  }

  nh.param<double>("rho", rho, 0.5);
  nh.param<double>("radius", radius, 0.1);
  nh.param<int>("sample", sample, 0);
  nh.param<double>("reduction", reduction, 0.8);
  nh.param<int>("threads", threads, 1);

  if (reduction >= 1 || reduction <= 0)
  {
    ROS_ERROR("Reduction must be (0 < reduction < 1)");
    return 0;
  }

  pcl::PointCloud<pcl::PointXYZ> cloud, filtered_cloud;
  afront_meshing::AfrontMeshing mesher;

  if (extension == ".pcd" && pcl::io::loadPCDFile<pcl::PointXYZ>(input_file, cloud) == -1)
  {
    ROS_ERROR("Couldn't read pcd file!");
    return 0;
  }
  else if (extension == ".ply" && pcl::io::loadPLYFile<pcl::PointXYZ>(input_file, cloud) == -1)
  {
    ROS_ERROR("Couldn't read ply file!");
    return 0;
  }

  std::vector<int> indices;
  pcl::removeNaNFromPointCloud(cloud, filtered_cloud, indices);

  pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud(new pcl::PointCloud<pcl::PointXYZ>(filtered_cloud));
  mesher.setRho(rho);
  mesher.setReduction(reduction);
  mesher.setRadius(radius);
  mesher.setNumberOfThreads(threads);

  if (mesher.initMesher(in_cloud))
  {
    if (sample == 0)
      mesher.generateMesh();
    else
      for (auto i = 0; i < sample; ++i)
        mesher.stepMesh();

    pcl::io::savePLYFile(output_file, mesher.getMesh());
  }

  return 0;
}
