#ifndef MLS_SAMPLING_H
#define MLS_SAMPLING_H

#include <pcl/surface/mls.h>

namespace afront_meshing
{
  class MLSSampling : public pcl::MovingLeastSquaresOMP<pcl::PointXYZ, pcl::PointNormal>
  {
  public:
    struct SamplePointResults
    {
      pcl::PointNormal point;   /**< @brief The point projected on to the MLS surface */
      int              closest; /**< @brief The closest point index on the MLS surface to the project point */
      double           dist;    /**< @brief The distance squared between point and closest */
    };

    void process(pcl::PointCloud<pcl::PointNormal> &output);

    // expose protected function 'performUpsampling' from MLS
    MLSSampling::SamplePointResults samplePoint(const pcl::PointXYZ& pt) const;
    MLSSampling::SamplePointResults samplePoint(const pcl::PointNormal& pt) const;

  private:

    Eigen::Vector2f calculateCurvature(const float &u, const float &v, const MLSResult &mls_result) const;
    Eigen::Vector2f calculateCurvature(const int &index) const;

    pcl::PointCloud<pcl::PointXYZ> cloud_;
  };
}

#endif // MLS_SAMPLING_H
