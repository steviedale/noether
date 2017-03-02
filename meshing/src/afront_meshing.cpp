#include <meshing/afront_meshing.h>

//template class PCL_EXPORTS afront_meshing::MLSSampling<pcl::PointXYZ, pcl::PointNormal>;
namespace afront_meshing
{
  pcl::PointNormal MLSSampling::samplePoint(const pcl::PointXYZ& pt)
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    cloud->push_back(pt);

    setUpsamplingMethod(pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointNormal>::DISTINCT_CLOUD);
    setDistinctCloud(cloud);

    pcl::PointCloud<pcl::PointNormal> cloud_out;
    performUpsampling(cloud_out);

    return cloud_out.points[0];
  }

  void AfrontMeshing::setInputCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
  {
    input_cloud_ = cloud;//pcl::PointCloud<pcl::PointXYZ>::Ptr(cloud);
  }

  void AfrontMeshing::computeGuidanceField()
  {
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);

    // Calculate MLS
    cloud_normals_ = pcl::PointCloud<pcl::PointNormal>::Ptr(new pcl::PointCloud<pcl::PointNormal>());
    pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointNormal> mls;
    mls.setInputCloud (input_cloud_);
    mls.setPolynomialFit (true);
    mls.setSearchMethod (tree);
    mls.setSearchRadius (0.03);
    mls.process (*cloud_normals_);

    // Calculate Curvatures
    pcl::PrincipalCurvaturesEstimation<pcl::PointXYZ, pcl::PointNormal, pcl::PrincipalCurvatures> principal_curvatures_estimation;

    // Provide the original point cloud (without normals)
    principal_curvatures_estimation.setInputCloud (input_cloud_);

    // Provide the point cloud with normals
    principal_curvatures_estimation.setInputNormals (cloud_normals_);

    // Use the same KdTree from the normal estimation
    principal_curvatures_estimation.setSearchMethod (tree);
    principal_curvatures_estimation.setRadiusSearch (0.03);

    curvatures_ = pcl::PointCloud<pcl::PrincipalCurvatures>::Ptr(new pcl::PointCloud<pcl::PrincipalCurvatures>());
    principal_curvatures_estimation.compute (*curvatures_);
  }

  void startMeshing()
  {

  }

  double AfrontMeshing::getCurvature(int index)
  {
    if(index >= curvatures_->points.size() )
    {
      return -1.0;
    }

    double x = curvatures_->points[index].principal_curvature[0];
    double y = curvatures_->points[index].principal_curvature[1];
    double z = curvatures_->points[index].principal_curvature[2];
    double min;
    min = x > y ? x : y;
    min = z > min ? z : min;

    double curv = min / (x + y + z);
    return curv;
  }


}

