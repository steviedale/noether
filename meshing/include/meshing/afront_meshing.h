#ifndef AFRONT_MESHING_H
#define AFRONT_MESHING_H

#include <pcl/common/common.h>
#include <pcl/point_traits.h>
#include <pcl/point_types.h>
#include <pcl/PolygonMesh.h>

#include <pcl/surface/mls.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/principal_curvatures.h>

namespace afront_meshing
{
  //template <typename PointInT, typename PointOutT> void

  class MLSSampling : public pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointNormal>
  {
  public:
    // expose protected function 'performUpsampling' from MLS
    pcl::PointNormal samplePoint(const pcl::PointXYZ& pt);

  private:
    pcl::PointCloud<pcl::PointXYZ> cloud_;

  };


  class AfrontMeshing
  {
  public:
     void setInputCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);
     void computeGuidanceField();
     void startMeshing();

     /**
      * @brief setRho The primary variable used to control mesh triangulation size
      * @param val
      */
     void setRho(double val){rho_ = val;}

  private:
     pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud_;
     pcl::PointCloud<pcl::PointNormal>::Ptr cloud_normals_;

     /**
      * @brief curvatures_ each point contains: principal_curvature[3], pc1, and pc2
      * principal_curvature contains the eigenvector for the minimum eigen value
      * pc1 = eigenvalues_ [2] * indices_size;
      * pc2 = eigenvalues_ [1] * indices_size;
      * curvature change calculation = eig_val_min/(sum(eig_vals))
      */
     pcl::PointCloud<pcl::PrincipalCurvatures>::Ptr curvatures_;

     double getCurvature(int index);

     double rho_;
  };


}

#endif // AFRONT_MESHING_H
