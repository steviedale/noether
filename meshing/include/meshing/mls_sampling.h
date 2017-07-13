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
      pcl::PointXYZ orig;           /**< @brief The point to be projected on to the MLS surface */
      pcl::PointXYZINormal point;   /**< @brief The point projected on to the MLS surface */
      int              closest;     /**< @brief The closest point index on the MLS surface to the project point */
      MLSResult        mls;         /**< @brief The MLS Results for the closest point */
      double           dist;        /**< @brief The distance squared between point and closest */
    };

    struct PolynomialPartialDerivative
    {
      double z;     /**< @brief The z component of the polynomial evaluated at z(u, v). */
      double z_u;   /**< @brief The partial derivative dz/du. */
      double z_v;   /**< @brief The partial derivative dz/dv. */
      double z_uu;  /**< @brief The partial derivative d^2z/du^2. */
      double z_vv;  /**< @brief The partial derivative d^2z/dv^2. */
      double z_uv;  /**< @brief The partial derivative d^2z/dudv. */
    };

    void process(pcl::PointCloud<pcl::PointNormal> &output);

    // expose protected function 'performUpsampling' from MLS
    MLSSampling::SamplePointResults samplePoint(const pcl::PointXYZ& pt) const;
    MLSSampling::SamplePointResults samplePoint(const pcl::PointNormal& pt) const;

    pcl::PointXYZINormal projectPoint(double u, double v, double w, const MLSResult &mls_result) const;

    double getMaxCurvature() const {return max_curvature_;}
    double getMinCurvature() const {return min_curvature_;}

    MLSResult getMLSResult(const int index) const {return mls_results_[index];}

    double getPolynomialValue(double u, double v, const MLSResult &mls_result) const;
    PolynomialPartialDerivative getPolynomialPartialDerivative(const double u, const double v, const MLSResult &mls_result) const;
  private:

    Eigen::Vector2f calculateCurvature(const double u, const double v, const MLSResult &mls_result) const;
    Eigen::Vector2f calculateCurvature(const Eigen::Vector3f pt, const int &index) const;

    pcl::PointCloud<pcl::PointXYZ> cloud_;
    double max_curvature_;
    double min_curvature_;
  };
}

#endif // MLS_SAMPLING_H
