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
  double boundary_angle;
  int order;
  int sample;
  int threads;
  std::string extension;

  pcl::console::setVerbosityLevel(pcl::console::L_DEBUG);

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

  nh.param<double>("rho", rho, afront_meshing::AFRONT_DEFAULT_RHO);
  nh.param<double>("radius", radius, 0);
  nh.param<int>("sample", sample, 0);
  nh.param<double>("reduction", reduction, afront_meshing::AFRONT_DEFAULT_REDUCTION);
  nh.param<int>("threads", threads, afront_meshing::AFRONT_DEFAULT_THREADS);
  nh.param<int>("order", order, afront_meshing::AFRONT_DEFAULT_POLYNOMIAL_ORDER);
  nh.param<double>("boundary_angle", boundary_angle, afront_meshing::AFRONT_DEFAULT_BOUNDARY_ANGLE_THRESHOLD);

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

  Eigen::Vector4f centroid;
  pcl::compute3DCentroid(*in_cloud, centroid);
  Eigen::Vector3f view_pt (0.0f, 0.0f, 0.0f);
  Eigen::Vector3f view_norm (centroid.head<3>());
  view_norm.normalize();
  ROS_DEBUG_STREAM("View Normal: " << view_norm);

  mesher.setRho(rho);
  mesher.setReduction(reduction);
  mesher.setSearchRadius(radius);
  mesher.setNumberOfThreads(threads);
  mesher.setPolynomialOrder(order);
  mesher.setBoundaryAngleThreshold(boundary_angle);
  mesher.setInputCloud(in_cloud);

  if (mesher.initialize())
  {
    if (sample == 0)
      mesher.reconstruct();
    else
      for (auto i = 0; i < sample; ++i)
        mesher.stepReconstruction();

    pcl::PolygonMesh mesh = mesher.getMesh();

    if(!mesher.setNormalsFromViewPoint(view_pt, view_norm, mesh))
    {
      ROS_WARN("Failed to set mesh normals from given viewpoint (origin to cloud centroid)");
    }
    else
    {
      pcl::io::savePLYFile(output_file, mesh);
    }
  }
  else
  {
    ROS_ERROR("Failed to initialize AFront Mesher!");
    return -1;
  }

  return 0;
}
