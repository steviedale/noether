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

    double gu = u;
    double gv = v;
    PolynomialPartialDerivative d = getPolynomialPartialDerivative(gu, gv, mls_result);
    double gw = d.z;
    double tol = 1e-8;
    double err_total;
    double dist1 = std::abs(gw - w);
    double dist2;
    do
    {
      double e1 = (gu - u) + d.z_u * gw - d.z_u * w;
      double e2 = (gv - v) + d.z_v * gw - d.z_v * w;

      double F1u = 1 + d.z_uu * gw + d.z_u * d.z_u - d.z_uu * w;
      double F1v = d.z_uv * gw + d.z_u * d.z_v - d.z_uv * w;

      double F2u = d.z_uv * gw + d.z_v * d.z_u - d.z_uv * w;
      double F2v = 1 + d.z_vv * gw + d.z_v * d.z_v - d.z_vv * w;

      Eigen::MatrixXd J(2, 2);
      J(0, 0) = F1u;
      J(0, 1) = F1v;
      J(1, 0) = F2u;
      J(1, 1) = F2v;

      Eigen::Vector2d err(e1, e2);
      Eigen::MatrixXd update = J.inverse() * err;
      gu -= update(0);
      gv -= update(1);

      d = getPolynomialPartialDerivative(gu, gv, mls_result);
      gw = d.z;
      dist2 = std::sqrt((gu - u) * (gu - u) + (gv - v) * (gv - v) + (gw - w) * (gw - w));

      err_total = std::sqrt(e1 * e1 + e2 * e2);
      //std::printf("Distance: %.10e\n", err_total);
    } while (err_total > tol && dist2 < dist1);

    if (dist2 > dist1) // the optimization was diverging
    {
      gu = u;
      gv = v;
      d = getPolynomialPartialDerivative(gu, gv, mls_result);
      gw = d.z;
    }

    //std::printf("Start Dist: %.10e\t End Dist: %.10e\n", dist1, dist2);

    pcl::PointXYZINormal result;
    result.x = static_cast<float> (mls_result.mean[0] + mls_result.u_axis[0] * gu + mls_result.v_axis[0] * gv + mls_result.plane_normal[0] * gw);
    result.y = static_cast<float> (mls_result.mean[1] + mls_result.u_axis[1] * gu + mls_result.v_axis[1] * gv + mls_result.plane_normal[1] * gw);
    result.z = static_cast<float> (mls_result.mean[2] + mls_result.u_axis[2] * gu + mls_result.v_axis[2] * gv + mls_result.plane_normal[2] * gw);

    Eigen::Vector3d normal = mls_result.plane_normal - d.z_u * mls_result.u_axis - d.z_v * mls_result.v_axis;
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

    result.mls = mls_results_[result.closest];
    result.point = projectPoint(u_disp, v_disp, w_disp, result.mls);
    return result;
  }

  MLSSampling::SamplePointResults MLSSampling::samplePoint(const pcl::PointNormal& pt) const
  {
    pcl::PointXYZ search_pt(pt.x, pt.y, pt.z);
    return samplePoint(search_pt);
  }

  Eigen::Vector2f MLSSampling::calculateCurvature(const double u, const double v, const MLSResult &mls_result) const
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
      PolynomialPartialDerivative d = getPolynomialPartialDerivative(u, v, mls_result);
      double Z = 1 + d.z_u * d.z_u + d.z_v * d.z_v;
      double Zlen = sqrt(Z);
      double K = (d.z_uu * d.z_vv - d.z_uv * d.z_uv) / (Z * Z);
      double H = ((1.0 + d.z_v * d.z_v) * d.z_uu - 2.0 * d.z_u * d.z_v * d.z_uv + (1.0 + d.z_u * d.z_u) * d.z_vv) / (2.0 * Zlen * Zlen * Zlen);
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

  double MLSSampling::getPolynomialValue(double u, double v, const MLSResult &mls_result) const
  {
      // Compute the polynomial's terms at the current point
      // Example for second order: z = a + b*y + c*y^2 + d*x + e*x*y + f*x^2
      double u_pow, v_pow, result;
      int j = 0;
      u_pow = 1;
      result = 0;
      for (int ui = 0; ui <= order_; ++ui)
      {
        v_pow = 1;
        for (int vi = 0; vi <= order_ - ui; ++vi)
        {
          result += mls_result.c_vec[j++] * u_pow * v_pow;
          v_pow *= v;
        }
        u_pow *= u;
      }

      return result;
  }

  MLSSampling::PolynomialPartialDerivative MLSSampling::getPolynomialPartialDerivative(const double u, const double v, const MLSResult &mls_result) const
  {
    // Compute the displacement along the normal using the fitted polynomial
    // and compute the partial derivatives needed for estimating the normal
    PolynomialPartialDerivative d;
    Eigen::VectorXd u_pow(order_ + 2), v_pow(order_ + 2);
    int j = 0;

    d.z = d.z_u = d.z_v = d.z_uu = d.z_vv = d.z_uv = 0;
    u_pow(0) = v_pow(0) = 1;
    for (int ui = 0; ui <= order_; ++ui)
    {
      for (int vi = 0; vi <= order_ - ui; ++vi)
      {
        // Compute displacement along normal
        d.z += u_pow(ui) * v_pow(vi) * mls_result.c_vec[j];

        // Compute partial derivatives
        if (ui >= 1)
          d.z_u += mls_result.c_vec[j] * ui * u_pow(ui - 1) * v_pow(vi);

        if (vi >= 1)
          d.z_v += mls_result.c_vec[j] * vi * u_pow(ui) * v_pow(vi - 1);

        if (ui >= 1 && vi >= 1)
          d.z_uv += mls_result.c_vec[j] * ui * u_pow(ui - 1) * vi * v_pow(vi - 1);

        if (ui >= 2)
          d.z_uu += mls_result.c_vec[j] * ui * (ui - 1) * u_pow(ui - 2) * v_pow(vi);

        if (vi >= 2)
          d.z_vv += mls_result.c_vec[j] * vi * (vi - 1) * u_pow(ui) * v_pow(vi - 2);

        if (ui == 0)
          v_pow(vi + 1) = v_pow(vi) * v;

        ++j;
      }
      u_pow(ui + 1) = u_pow(ui) * u;
    }

    return d;
  }
}
