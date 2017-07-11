#include <meshing/mls_sampling.h>

namespace afront_meshing
{
  void MLSSampling::process(pcl::PointCloud<pcl::PointNormal> &output)
  {
    MovingLeastSquares::process(output);

    // Calculate the max principle curvature using mls result polynomial data
    max_curvature_ = std::numeric_limits<double>::min();
    min_curvature_ = std::numeric_limits<double>::max();
    for(int i = 0; i < output.size(); ++i)
    {
      double k = calculateCurvature(output[i].getVector3fMap(), i).cwiseAbs().maxCoeff();
      if (k < 1e-5)
        k = 1e-5;

      output[i].curvature = k;
      if (k > max_curvature_)
        max_curvature_ = k;

      if (k < min_curvature_)
        min_curvature_ = k;
    }
  }

//  pcl::PointXYZINormal MLSSampling::projectPoint(double u, double v, double w, const MLSResult &mls_result) const
//  {
//    // z = a + b*y + c*y^2 + d*x + e*x*y + f*x^2
//    double a = mls_result.c_vec[0];
//    double b = mls_result.c_vec[1];
//    double c = mls_result.c_vec[2];
//    double d = mls_result.c_vec[3];
//    double e = mls_result.c_vec[4];
//    double f = mls_result.c_vec[5];

//    double gu = u;
//    double gv = v;
//    double gw;
//    double perr;
//    for (auto i = 0; i < 100; i++)
//    {
//      gw = a + b*gv + c*gv*gv + d*gu + e*gu*gv + f*gu*gu;
//      double err = (u - gu) * (u - gu) + (v - gv) * (v - gv) + (w - gw) * (w - gw);
//      double du = -2.0 * (u - gu) - 2 * (w - gw) * (d + e * gv + 2 * f *gu);
//      double dv = -2.0 * (v - gv) - 2 * (w - gw) * (b + e * gu + 2 * c *gv);

//      Eigen::MatrixXd J, Jpinv;
//      J.resize(1, 2);
//      J(0, 0) = du;
//      J(0, 1) = dv;

//      if (!dampedPInv(J, Jpinv))
//        break;

//      Eigen::MatrixXd update = Jpinv * err;
//      gu -= update(0);
//      gv -= update(1);

//      double dist = std::sqrt(err);
//      if (i != 0)
//      {
////        double delta = dist - perr;
////        double pchg = std::abs(delta / perr);
////        std::printf("Delta Error: %.10e\t Percent Change: %f\n", delta, pchg);
//      }
//      perr = dist;
//    }

//    double Zx = d + e * gv + 2.0 * f * gu;
//    double Zy = b + e * gu + 2.0 * c * gv;
////    Eigen::Vector3d dx(1, 0, Zx);
////    Eigen::Vector3d dy(0, 1, Zy);
////    Eigen::Vector3d normal = (dx.cross(dy)).normalized();

//    pcl::PointXYZINormal result;
//    result.x = static_cast<float> (mls_result.mean[0] + mls_result.u_axis[0] * gu + mls_result.v_axis[0] * gv + mls_result.plane_normal[0] * gw);
//    result.y = static_cast<float> (mls_result.mean[1] + mls_result.u_axis[1] * gu + mls_result.v_axis[1] * gv + mls_result.plane_normal[1] * gw);
//    result.z = static_cast<float> (mls_result.mean[2] + mls_result.u_axis[2] * gu + mls_result.v_axis[2] * gv + mls_result.plane_normal[2] * gw);

//    Eigen::Vector3d normal = mls_result.plane_normal - Zx * mls_result.u_axis - Zy * mls_result.v_axis;
//    normal.normalize();

//    result.normal_x = static_cast<float> (normal[0]);
//    result.normal_y = static_cast<float> (normal[1]);
//    result.normal_z = static_cast<float> (normal[2]);

//    Eigen::Vector2f k = calculateCurvature(gu, gv, mls_result);
//    result.curvature = k.cwiseAbs().maxCoeff();
//    if (result.curvature < 1e-5)
//      result.curvature = 1e-5;

//    return result;
//  }

  pcl::PointXYZINormal MLSSampling::projectPoint(double u, double v, double w, const MLSResult &mls_result) const
  {
    // This was implemented based on this https://math.stackexchange.com/questions/1497093/shortest-distance-between-point-and-surface
    // z = a + b*y + c*y^2 + d*x + e*x*y + f*x^2
    double a = mls_result.c_vec[0];
    double b = mls_result.c_vec[1];
    double c = mls_result.c_vec[2];
    double d = mls_result.c_vec[3];
    double e = mls_result.c_vec[4];
    double f = mls_result.c_vec[5];

    double gu = u;
    double gv = v;
    double gw = a + b*gv + c*gv*gv + d*gu + e*gu*gv + f*gu*gu;
    double tol = 1e-16;
    double err_total;
    double dist1 = std::abs(gw - w);
    double dist2;
    do
    {
      double Zx = d + e * v + 2.0 * f * u;
      double Zy = b + e * u + 2.0 * c * v;
      double Zxx = 2.0 * f;
      double Zxy = e; //Note: Zyx = Zxy
      double Zyy = 2.0 * c;

      double e1 = (gu - u) + Zx * gw - Zx * w;
      double e2 = (gv - v) + Zy * gw - Zy * w;

      double F1u = 1 + Zxx * gw + Zx * Zx - Zxx * w;
      double F1v = Zxy * gw + Zx * Zy - Zxy * w;

      double F2u = Zxy * gw + Zy * Zx - Zxy * w;
      double F2v = 1 + Zyy * gw + Zy * Zy - Zyy * w;

      Eigen::MatrixXd J(2, 2);
      J(0, 0) = F1u;
      J(0, 1) = F1v;
      J(1, 0) = F2u;
      J(1, 1) = F2v;

      Eigen::Vector2d err(e1, e2);
      Eigen::MatrixXd update = J.inverse() * err;
      gu -= update(0);
      gv -= update(1);
      gw = a + b*gv + c*gv*gv + d*gu + e*gu*gv + f*gu*gu;
      dist2 = std::sqrt((gu - u) * (gu - u) + (gv - v) * (gv - v) + (gw - w) * (gw - w));

      err_total = std::sqrt(e1 * e1 + e2 * e2);
      std::printf("Distance: %.10e\n", err_total);
    } while (err_total > tol && dist2 < dist1);

    if (dist2 > dist1) // the optimization was diverging
    {
      gu = u;
      gv = v;
    }

    std::printf("Start Dist: %.10e\t End Dist: %.10e\n", dist1, dist2);

    double Zx = d + e * gv + 2.0 * f * gu;
    double Zy = b + e * gu + 2.0 * c * gv;

    pcl::PointXYZINormal result;
    result.x = static_cast<float> (mls_result.mean[0] + mls_result.u_axis[0] * gu + mls_result.v_axis[0] * gv + mls_result.plane_normal[0] * gw);
    result.y = static_cast<float> (mls_result.mean[1] + mls_result.u_axis[1] * gu + mls_result.v_axis[1] * gv + mls_result.plane_normal[1] * gw);
    result.z = static_cast<float> (mls_result.mean[2] + mls_result.u_axis[2] * gu + mls_result.v_axis[2] * gv + mls_result.plane_normal[2] * gw);

    Eigen::Vector3d normal = mls_result.plane_normal - Zx * mls_result.u_axis - Zy * mls_result.v_axis;
    normal.normalize();

    result.normal_x = static_cast<float> (normal[0]);
    result.normal_y = static_cast<float> (normal[1]);
    result.normal_z = static_cast<float> (normal[2]);

    Eigen::Vector2f k = calculateCurvature(gu, gv, mls_result);
    result.curvature = k.cwiseAbs().maxCoeff();
    if (result.curvature < 1e-5)
      result.curvature = 1e-5;

    return result;
  }

  MLSSampling::SamplePointResults MLSSampling::samplePoint(const pcl::PointXYZ& pt) const
  {
    if (!pcl_isfinite(pt.x))
      std::cout << "Error: Sample point is not finite\n";

    SamplePointResults result;
    result.orig = pt;

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
    double u_disp = (add_point - mls_results_[result.closest].mean).dot(mls_results_[result.closest].u_axis);
    double v_disp = (add_point - mls_results_[result.closest].mean).dot(mls_results_[result.closest].v_axis);
    double w_disp = (add_point - mls_results_[result.closest].mean).dot(mls_results_[result.closest].plane_normal);

    pcl::PointNormal result_point;
    pcl::Normal result_normal;
    result.mls = mls_results_[result.closest];
//    projectPointToMLSSurface(static_cast<float> (u_disp),
//                             static_cast<float> (v_disp),
//                             result.mls.u_axis,
//                             result.mls.v_axis,
//                             result.mls.plane_normal,
//                             result.mls.mean,
//                             result.mls.curvature,
//                             result.mls.c_vec,
//                             result.mls.num_neighbors,
//                             result_point, result_normal);

    result.point = projectPoint(u_disp, v_disp, w_disp, result.mls);


//    // Copy additional point information if available
//    copyMissingFields(input_->points[result.closest], result_point);

//    // Calculate principal curvature
//    Eigen::Vector2f k = calculateCurvature(u_disp, v_disp, result.mls);

//    result.point.x = result_point.x;
//    result.point.y = result_point.y;
//    result.point.z = result_point.z;
//    result.point.normal_x = result_normal.normal_x;
//    result.point.normal_y = result_normal.normal_y;
//    result.point.normal_z = result_normal.normal_z;

//    result.point.curvature = k.cwiseAbs().maxCoeff();
//    if (result.point.curvature < 1e-5)
//      result.point.curvature = 1e-5;


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
      double Zlen = sqrt(Z);
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

  Eigen::Vector2f MLSSampling::calculateCurvature(const Eigen::Vector3f pt, const int &index) const
  {
    Eigen::Vector3d point = pt.template cast<double>();
    float u_disp = static_cast<float> ((point - mls_results_[index].mean).dot(mls_results_[index].u_axis)),
    v_disp = static_cast<float> ((point - mls_results_[index].mean).dot(mls_results_[index].v_axis));
    return calculateCurvature(u_disp, v_disp, mls_results_[index]);
  }

}
