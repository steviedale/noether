#include <meshing/mls_sampling.h>

namespace afront_meshing
{
  void MLSSampling::process(pcl::PointCloud<pcl::PointNormal> &output)
  {
    MovingLeastSquares::process(output);

    // Calculate the max principle curvature using mls result polynomial data
    for(int i = 0; i < output.size(); ++i)
    {
      Eigen::Vector2f k = calculateCurvature(i);
      output[i].curvature = k.cwiseAbs().maxCoeff();
      if (output[i].curvature < 1e-5)
      {
        output[i].curvature = 1e-5;
        std::cout << "Warning: Very Small Curvature!\n";
      }
    }
  }

  MLSSampling::SamplePointResults MLSSampling::samplePoint(const pcl::PointXYZ& pt) const
  {
    if (!pcl_isfinite(pt.x))
      std::cout << "Error: Sample point is not finite\n";

    SamplePointResults result;

    // Get 3D position of point
    //Eigen::Vector3f pos = distinct_cloud_->points[dp_i].getVector3fMap ();
    std::vector<int> nn_indices;
    std::vector<float> nn_dists;
    tree_->nearestKSearch(pt, 1, nn_indices, nn_dists);
    result.closest = nn_indices.front();
    result.dist = nn_dists.front();

    // If the closest point did not have a valid MLS fitting result
    // OR if it is too far away from the sampled point
    if (mls_results_[result.closest].valid == false)
      std::printf("\x1B[31m\tMLS Results Not Valid!\n");

    Eigen::Vector3d add_point = pt.getVector3fMap().template cast<double>();
    float u_disp = static_cast<float> ((add_point - mls_results_[result.closest].mean).dot(mls_results_[result.closest].u_axis)),
    v_disp = static_cast<float> ((add_point - mls_results_[result.closest].mean).dot(mls_results_[result.closest].v_axis));

    pcl::Normal result_normal;
    MLSResult result_mls = mls_results_[result.closest];
    projectPointToMLSSurface(u_disp, v_disp,
                             result_mls.u_axis,
                             result_mls.v_axis,
                             result_mls.plane_normal,
                             result_mls.mean,
                             result_mls.curvature,
                             result_mls.c_vec,
                             result_mls.num_neighbors,
                             result.point, result_normal);

    // Copy additional point information if available
    copyMissingFields(input_->points[result.closest], result.point);

    // Calculate principal curvature
    Eigen::Vector2f k = calculateCurvature(u_disp, v_disp, result_mls);

    result.point.normal_x = result_normal.normal_x;
    result.point.normal_y = result_normal.normal_y;
    result.point.normal_z = result_normal.normal_z;

    result.point.curvature = k.cwiseAbs().maxCoeff();
    if (result.point.curvature < 1e-5)
    {
      result.point.curvature = 1e-5;
      std::cout << "Warning: Very Small Curvature!\n";
    }

    return result;
  }

  MLSSampling::SamplePointResults MLSSampling::samplePoint(const pcl::PointNormal& pt) const
  {
    pcl::PointXYZ search_pt(pt.x, pt.y, pt.z);
    return samplePoint(search_pt);
  }

  Eigen::Vector2f MLSSampling::calculateCurvature(const float &u, const float &v, const MLSResult &mls_result) const
  {
    Eigen::Vector2f k;
    Eigen::VectorXd coeff = mls_result.c_vec;

    k << 1e-5, 1e-5;
    // Note: this use the Monge Patch to derive the Gaussian curvature and Mean Curvature found here http://mathworld.wolfram.com/MongePatch.html
    // Then:
    //      k1 = H + sqrt(H^2 - K)
    //      k1 = H - sqrt(H^2 - K)
    if (coeff.size() != 0) // coeff.size() == 0 When all sample points are on a plane.
    {
      // z = a + b*y + c*y^2 + d*x + e*x*y + f*x^2
      double a = coeff[0];
      double b = coeff[1];
      double c = coeff[2];
      double d = coeff[3];
      double e = coeff[4];
      double f = coeff[5];

      double Zx = d + e * v + 2.0 * f * u;
      double Zy = b + e * u + 2.0 * c * v;
      double Zxx = 2.0 * f;
      double Zxy = e; //Note: Zyx = Zxy
      double Zyy = 2.0 * c;

      double Z = 1 + Zx * Zx + Zy * Zy;
      double Zlen = sqrt(1 + Zx * Zx + Zy * Zy);
      double K = (Zxx * Zyy - Zxy * Zxy) / (Z * Z);
      double H = ((1.0 + Zy * Zy) * Zxx - 2.0 * Zx * Zy * Zxy + (1.0 + Zx * Zx) * Zyy) / (2.0 * Zlen * Zlen * Zlen);
      double disc2 = H * H - K;
      assert (disc2>=0.0);
      double disc = sqrt(disc2);
      k[0] = H + disc;
      k[1] = H - disc;

      if (std::abs(k[0]) > std::abs(k[1])) std::swap(k[0], k[1]);
    }
    else
    {
      std::cout << "Error: Polynomial fit failed! Neighbors: " << mls_result.num_neighbors << "\n";
    }

    return k;
  }

  Eigen::Vector2f MLSSampling::calculateCurvature(const int &index) const
  {
    return calculateCurvature(0.001, 0.001, mls_results_[index]);
  }
}
