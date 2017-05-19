#include <meshing/mls_sampling.h>

namespace afront_meshing
{
  void MLSSampling::process(pcl::PointCloud<pcl::PointNormal> &output)
  {
    MovingLeastSquares::process(output);

    // Calculate the max principle curvature using mls result polynomial data
    float min = std::numeric_limits<float>::max();
    for(int i = 0; i < output.size(); ++i)
    {
      Eigen::Vector2f k = calculateCurvature(i);
      output[i].curvature = k.cwiseAbs().maxCoeff();
      if (output[i].curvature < min)
        min = output[i].curvature;
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
    double a = coeff[0];
    double b = coeff[1];
    double c = coeff[2];
    double d = coeff[3];
    double e = coeff[4];
    double f = coeff[5];

    double nx = b + 2*d*u + e*v;
    double ny = c + e*u + 2*f*v;
    double nlen = sqrt(nx*nx+ny*ny+1);

    double b11 = 2*d/nlen;
    double b12 = e/nlen;
    double b22 = 2*f/nlen;
    double disc = (b11+b22)*(b11+b22) - 4*(b11*b22-b12*b12);
    assert (disc>=0);
    double disc2 = sqrt(disc);
    k[0] = (b11+b22+disc2)/2.0;
    k[1] = (b11+b22-disc2)/2.0;
    if (std::abs(k[0]) > std::abs(k[1])) std::swap(k[0], k[1]);
    return k;
  }

  Eigen::Vector2f MLSSampling::calculateCurvature(const int &index) const
  {
    return calculateCurvature(0.0, 0.0, mls_results_[index]);
  }
}
