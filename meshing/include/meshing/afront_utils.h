#ifndef AFRONT_UTILS_H
#define AFRONT_UTILS_H

#include <pcl/point_types.h>

namespace afront_meshing
{
namespace utils
{

  /** @brief Convert Eigen Vector3f to PCL PointXYZ */
  pcl::PointXYZ convertEigenToPCL(const Eigen::Vector3f &p)
  {
    return pcl::PointXYZ(p(0), p(1), p(2));
  }

  /**
  * @brief Get the mid point of a half edge given it's verticies
  * @param p1 Vertex of half edge
  * @param p2 Vectex of half edge
  * @return The mid point of the half edge
  */
  Eigen::Vector3f getMidPoint(const Eigen::Vector3f &p1, const Eigen::Vector3f &p2)
  {
    return (p1 + p2) / 2.0;
  }

  /**
  * @brief Get the length of a half edge given it's verticies
  * @param p1 Vertex of half edge
  * @param p2 Vectex of half edge
  * @return The lenght of the half edge
  */
  double getEdgeLength(const Eigen::Vector3f &p1, const Eigen::Vector3f &p2)
  {
    return (p2-p1).norm();
  }

} // namespace utils

} // namespace afront_meshing
#endif // AFRONT_UTILS_H
