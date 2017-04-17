#include <pcl/geometry/mesh_conversion.h>
#include <pcl/conversions.h>
#include <meshing/afront_meshing.h>

//template class PCL_EXPORTS afront_meshing::MLSSampling<pcl::PointXYZ, pcl::PointNormal>;
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

    std::cout << "Minimum Curvature: " << min;
  }

  pcl::PointNormal MLSSampling::samplePoint(const pcl::PointXYZ& pt) const
  {
    if (!pcl_isfinite(pt.x))
      std::cout << "Error: Sample point is not finite\n";

    // Get 3D position of point
    //Eigen::Vector3f pos = distinct_cloud_->points[dp_i].getVector3fMap ();
    std::vector<int> nn_indices;
    std::vector<float> nn_dists;
    tree_->nearestKSearch(pt, 1, nn_indices, nn_dists);
    int input_index = nn_indices.front ();

    // If the closest point did not have a valid MLS fitting result
    // OR if it is too far away from the sampled point
//    if (mls_results_[input_index].valid == false)
//      continue;

    Eigen::Vector3d add_point = pt.getVector3fMap().template cast<double>();
    float u_disp = static_cast<float> ((add_point - mls_results_[input_index].mean).dot(mls_results_[input_index].u_axis)),
    v_disp = static_cast<float> ((add_point - mls_results_[input_index].mean).dot(mls_results_[input_index].v_axis));

    pcl::PointNormal result_point;
    pcl::Normal result_normal;
    MLSResult result_mls = mls_results_[input_index];
    projectPointToMLSSurface(u_disp, v_disp,
                             result_mls.u_axis,
                             result_mls.v_axis,
                             result_mls.plane_normal,
                             result_mls.mean,
                             result_mls.curvature,
                             result_mls.c_vec,
                             result_mls.num_neighbors,
                             result_point, result_normal);

    // Copy additional point information if available
    copyMissingFields(input_->points[input_index], result_point);

    // Calculate principal curvature
    Eigen::Vector2f k = calculateCurvature(u_disp, v_disp, result_mls);

    result_point.normal_x = result_normal.normal_x;
    result_point.normal_y = result_normal.normal_y;
    result_point.normal_z = result_normal.normal_z;
    result_point.curvature = k.cwiseAbs().maxCoeff();

    return result_point;
  }

  pcl::PointNormal MLSSampling::samplePoint(const pcl::PointNormal& pt) const
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

  void AfrontMeshing::setInputCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
  {
    input_cloud_ = cloud;//pcl::PointCloud<pcl::PointXYZ>::Ptr(cloud);
  }

  bool AfrontMeshing::computeGuidanceField()
  {
    mesh_vertex_data_ = pcl::PointCloud<MeshTraits::VertexData>::Ptr(&mesh_.getVertexDataCloud());

    input_cloud_tree_ = pcl::search::KdTree<pcl::PointXYZ>::Ptr(new pcl::search::KdTree<pcl::PointXYZ>);
    input_cloud_tree_->setInputCloud(input_cloud_);

    // Calculate MLS
    mls_cloud_ = pcl::PointCloud<pcl::PointNormal>::Ptr(new pcl::PointCloud<pcl::PointNormal>());

    //pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointNormal> mls;
    mls_.setComputeNormals(true);
    mls_.setInputCloud(input_cloud_);
    mls_.setPolynomialFit(true);
    mls_.setSearchMethod(input_cloud_tree_);
    mls_.setSearchRadius(r_);
    mls_.setUpsamplingMethod(pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointNormal>::DISTINCT_CLOUD);
    mls_.setDistinctCloud(input_cloud_);
    mls_.process(*mls_cloud_);

    mls_cloud_tree_ = pcl::search::KdTree<pcl::PointNormal>::Ptr(new pcl::search::KdTree<pcl::PointNormal>);
    mls_cloud_tree_->setInputCloud(mls_cloud_);

    return true;
  }

  pcl::PolygonMesh AfrontMeshing::getMesh() const
  {
    pcl::PolygonMesh out_mesh;
    pcl::geometry::toFaceVertexMesh(mesh_, out_mesh);
    return out_mesh;
  }

  pcl::PointCloud<pcl::Normal>::ConstPtr AfrontMeshing::getNormals() const
  {
    pcl::PointCloud<pcl::Normal>::Ptr pn(new pcl::PointCloud<pcl::Normal>());;
    pcl::copyPointCloud(*mls_cloud_, *pn);

    return pn;
  }

  void AfrontMeshing::startMeshing()
  {
    std::vector<int> K;
    std::vector<float> K_dist;

    std::cout << "starting meshing\n";

    // Create first triangle
    createFirstTriangle(2.5, 2.2, 0);

    const HalfEdgeIndex& idx_he_boundary = mesh_.getOutgoingHalfEdgeIndex(mesh_.getVertexIndex(mesh_.getVertexDataCloud()[0]));
    IHEAFC       circ_iheaf     = mesh_.getInnerHalfEdgeAroundFaceCirculator(idx_he_boundary);
    const IHEAFC circ_iheaf_end = circ_iheaf;
    do
    {
      HalfEdgeIndex he = circ_iheaf.getTargetIndex();
      queue_.push_back(he);
    } while (++circ_iheaf != circ_iheaf_end);
  }

  void AfrontMeshing::stepMesh()
  {
    HalfEdgeIndex half_edge = queue_.front();
    queue_.pop_front();

    updateKdTree();

    CanCutEarResults ccer = canCutEar(half_edge);
    if (ccer.valid)
    {
      cutEar(*ccer.valid);
      return;
    }

    // If we can not cut ear then try and grow.
    PredictVertexResults pv = predictVertex(half_edge);
    TriangleToCloseResults tc = isTriangleToClose(ccer, pv);
    if (tc.valid)
    {
      grow(ccer, pv);
    }
    else
    {
//      current_index_ = merge(tc);
    }
  }

  void AfrontMeshing::updateKdTree()
  {
    mesh_tree_.reset();
    mesh_tree_ = pcl::search::KdTree<MeshTraits::VertexData>::Ptr(new pcl::search::KdTree<MeshTraits::VertexData>);
    mesh_tree_->setInputCloud(mesh_vertex_data_);
  }

  void AfrontMeshing::createFirstTriangle(const double &x, const double &y, const double &z)
  {
    std::vector<int> K;
    std::vector<float> K_dist;

    pcl::PointNormal middle_pt;
    middle_pt.x = x;
    middle_pt.y = y;
    middle_pt.z = z;
    mls_cloud_tree_->nearestKSearch(middle_pt, 1, K, K_dist);

    pcl::PointNormal p1 = mls_.samplePoint(mls_cloud_->points[K[0]]);

    // Get the allowed grow distance
    double d = 2 * std::sin(rho_/2) / p1.curvature;
    GrowDistanceResults gdr = getGrowDistance(p1, d, d, d);

    // search for the nearest neighbor
    mls_cloud_tree_->nearestKSearch(p1, 2, K, K_dist);

    // use l1 and nearest neighbor to extend edge
    pcl::PointNormal mp, dp, p2, p3;;
    Eigen::Vector3d v1, v2, norm;

    dp = mls_cloud_->points[K[1]];
    v1 << (dp.x - p1.x), (dp.y - p1.y), (dp.z - p1.z);
    v1 = v1.normalized();
    norm << dp.normal_x, dp.normal_y, dp.normal_z;

    p2 = getPredictedVertex(p1, v1, gdr.l);
    mp = getMidPoint(p1, p2);
    d = getEdgeLength(p1, p2);

    v2 = norm.cross(v1).normalized();

    gdr = getGrowDistance(mp, d, d, d);
    p3 = getPredictedVertex(mp, v2, gdr.l);

    MeshTraits::FaceData fd = createFaceData(p1, p2, p3);
    VertexIndices vi;
    vi.push_back(mesh_.addVertex(pcl::PointXYZ(p1.x, p1.y, p1.z)));
    vi.push_back(mesh_.addVertex(pcl::PointXYZ(p2.x, p2.y, p2.z)));
    vi.push_back(mesh_.addVertex(pcl::PointXYZ(p3.x, p3.y, p3.z)));
    mesh_.addFace(vi[0], vi[1], vi[2], fd);
  }

  AfrontMeshing::CanCutEarResults AfrontMeshing::canCutEar(const HalfEdgeIndex &half_edge) const
  {
    CanCutEarResults result;
    result.he = half_edge;
    result.next = canCutEarHelper(half_edge, getNextHalfEdge(half_edge));
    result.prev = canCutEarHelper(getPrevHalfEdge(half_edge), half_edge);

    if (result.next.valid && result.prev.valid)
      if (result.next.aspect_ratio < result.prev.aspect_ratio)
        result.valid = &result.next;
      else
        result.valid = &result.prev;
    else if (result.next.valid)
      result.valid = &result.next;
    else if (result.prev.valid)
      result.valid = &result.prev;

    return result;
  }

  AfrontMeshing::CanCutEarResult AfrontMeshing::canCutEarHelper(const HalfEdgeIndex &half_edge1, const HalfEdgeIndex &half_edge2) const
  {
    CanCutEarResult result;
    pcl::PointXYZ p1, p2, p3;
    Eigen::Vector3d v1, v2, v3, cross;
    double dot, sina, cosa, bottom;

    result.first = half_edge1;
    result.second = half_edge2;
    result.valid = false;
    result.same_face = false;

    // First check and make sure both half edges are not associated the same face
    if (mesh_.getOppositeFaceIndex(result.first) == mesh_.getOppositeFaceIndex(result.second))
    {
      result.same_face = true;
      return result;
    }

    result.vi.push_back(mesh_.getOriginatingVertexIndex(result.first));
    result.vi.push_back(mesh_.getTerminatingVertexIndex(result.first));
    result.vi.push_back(mesh_.getTerminatingVertexIndex(result.second));
    p1 = mesh_.getVertexDataCloud()[result.vi[0].get()];
    p2 = mesh_.getVertexDataCloud()[result.vi[1].get()];
    p3 = mesh_.getVertexDataCloud()[result.vi[2].get()];

    v1 << (p1.x - p2.x), (p1.y - p2.y), (p1.z - p2.z);
    v2 << (p3.x - p2.x), (p3.y - p2.y), (p3.z - p2.z);
    v3 << (p1.x - p3.x), (p1.y - p3.y), (p1.z - p3.z);

    result.A = v1.norm();
    result.B = v2.norm();
    result.C = v3.norm();

    result.aspect_ratio = std::max(std::max(result.A, result.B), result.C)/std::min(std::min(result.A, result.B), result.C);

    // Check first angle of triangle
    dot = v1.dot(v2);
    cross = v1.cross(v2);
    bottom = v1.norm() * v2.norm();
    sina = cross.norm()/bottom;
    cosa = dot/bottom;
    result.c = atan2(sina, cosa);

    // Check second angle of triangle
    v2 *= -1.0;
    dot = v2.dot(v3);
    cross = v2.cross(v3);
    bottom = v2.norm() * v3.norm();
    sina = cross.norm()/bottom;
    cosa = dot/bottom;
    result.a = atan2(sina, cosa);

    result.b = M_PI - result.a - result.c;

    if (result.a < 1.22173 && result.b < 1.22173 && result.c < 1.22173)
      result.valid = true;

    return result;
  }

  void AfrontMeshing::cutEar(const CanCutEarResult &data)
  {
    // Add new face
    std::cout << "Ear Cut" << std::endl;
    MeshTraits::FaceData new_fd = createFaceData(mesh_.getVertexDataCloud()[data.vi[0].get()], mesh_.getVertexDataCloud()[data.vi[1].get()], mesh_.getVertexDataCloud()[data.vi[2].get()]);
    mesh_.addFace(data.vi[0], data.vi[1], data.vi[2], new_fd);

    queue_.push_back(getNextHalfEdge(getPrevHalfEdge(data.first)));
    std::remove_if(queue_.begin(), queue_.end(), [data](HalfEdgeIndex he){ return ((he == data.first)^(he == data.second));});
  }

  AfrontMeshing::PredictVertexResults AfrontMeshing::predictVertex(const HalfEdgeIndex &half_edge) const
  {
    // Local Variables
    PredictVertexResults result;

    // Get Afront FaceData
    result.he = half_edge;
    FaceIndex face_indx = mesh_.getOppositeFaceIndex(half_edge);
    MeshTraits::FaceData fd = mesh_.getFaceDataCloud()[face_indx.get()];

    // Get Half Edge Vertexs
    result.vi.push_back(mesh_.getOriginatingVertexIndex(half_edge));
    result.vi.push_back(mesh_.getTerminatingVertexIndex(half_edge));
    result.p[0] = mesh_.getVertexDataCloud()[result.vi[0].get()];
    result.p[1] = mesh_.getVertexDataCloud()[result.vi[1].get()];

    // Calculate min max of all lengths attached to half edge
    Eigen::Vector2d mm = getMinMaxEdgeLength(result.vi[0], result.vi[1]);

    // find the mid point of the edge
    result.mp = getMidPoint(result.p[0], result.p[1]);

    // Calculate the edge length
    double l = getEdgeLength(result.p[0], result.p[1]);

    // Calculate direction vector to move
    result.d = getGrowDirection(result.p[0], result.mp, fd);

    // Get the allowed grow distance
    result.gdr = getGrowDistance(result.mp, l, mm[0], mm[1]);

    // Return predicted vertex
    result.pv = getPredictedVertex(result.mp, result.d, result.gdr.l);
    result.p[2].x = result.pv.x;
    result.p[2].y = result.pv.y;
    result.p[2].z = result.pv.z;

    return result;
  }

  AfrontMeshing::TriangleToCloseResults AfrontMeshing::isTriangleToClose(const CanCutEarResults &ccer, const PredictVertexResults &pvr) const
  {
      TriangleToCloseResults result;
      std::vector<int> K;
      std::vector<float> K_dist;

      // search for the nearest neighbor
      pcl::PointXYZ search_pt(pvr.p[2]);
      mesh_tree_->radiusSearch(search_pt, 3.0 * pvr.gdr.ideal, K, K_dist);

      // Next need to loop through each vertice and find the half edges then check for fence violations and distance to edge






//    TriangleToCloseResults result;
//    std::vector<int> K;
//    std::vector<float> K_dist;

//    // search for the nearest neighbor
//    pcl::PointXYZ search_pt(pvr.p[2]);
//    mesh_tree_->nearestKSearch(search_pt, 1, K, K_dist);

//    result.pvr = pvr;
//    result.dist = K_dist[0];
//    MeshTraits::VertexData &data = mesh_vertex_data_->at(K[0]);
//    result.closest = mesh_.getVertexIndex(data);
//    result.valid = true;

//    if (result.dist < pow(pvr.l * 0.5, 2))
//      result.valid = false;

//    return result;

  }

  void AfrontMeshing::grow(const CanCutEarResults &ccer, const PredictVertexResults &pvr)
  {
    // Add new face
    MeshTraits::FaceData new_fd = createFaceData(pvr.p[0], pvr.p[1], pvr.p[2]);
    mesh_.addFace(pvr.vi[0], pvr.vi[1], mesh_.addVertex(pvr.p[2]), new_fd);
    mesh_tree_->setInputCloud(mesh_vertex_data_); // This may need to be replaced with using an octree.

    // Add new half edges to the queue
    queue_.push_back(getNextHalfEdge(ccer.prev.first));
    queue_.push_back(getPrevHalfEdge(ccer.next.second));
  }

  AfrontMeshing::VertexIndex AfrontMeshing::merge(const TriangleToCloseResults &tc)
  {
    MeshTraits::FaceData new_fd = createFaceData(tc.pvr.p[0], tc.pvr.p[1], mesh_.getVertexDataCloud()[tc.closest.get()]);
    mesh_.addFace(tc.pvr.vi[0], tc.pvr.vi[1], tc.closest, new_fd);
    mesh_tree_->setInputCloud(mesh_vertex_data_); // This may need to be replaced with using an octree.
    return tc.closest;
  }

  float AfrontMeshing::getCurvature(const int &index) const
  {
    if(index >= mls_cloud_->points.size())
      return -1.0;

    return mls_cloud_->at(index).curvature;
  }

  pcl::PointNormal AfrontMeshing::getMidPoint(const pcl::PointXYZ &p1, const pcl::PointXYZ &p2) const
  {
    pcl::PointNormal p;
    p.x = (p1.x + p2.x)/2.0;
    p.y = (p1.y + p2.y)/2.0;
    p.z = (p1.z + p2.z)/2.0;
    return p;
  }

  pcl::PointNormal AfrontMeshing::getMidPoint(const pcl::PointNormal &p1, const pcl::PointNormal &p2) const
  {
    pcl::PointNormal p;
    p.x = (p1.x + p2.x)/2.0;
    p.y = (p1.y + p2.y)/2.0;
    p.z = (p1.z + p2.z)/2.0;
    return p;
  }

  double AfrontMeshing::getEdgeLength(const pcl::PointXYZ &p1, const pcl::PointXYZ &p2) const
  {
    double dx2 = pow(p2.x - p1.x, 2);
    double dy2 = pow(p2.y - p1.y, 2);
    double dz2 = pow(p2.z - p1.z, 2);
    return std::sqrt(dx2 + dy2 + dz2);
  }

  double AfrontMeshing::getEdgeLength(const pcl::PointNormal &p1, const pcl::PointNormal &p2) const
  {
    double dx2 = pow(p2.x - p1.x, 2);
    double dy2 = pow(p2.y - p1.y, 2);
    double dz2 = pow(p2.z - p1.z, 2);
    return std::sqrt(dx2 + dy2 + dz2);
  }

  Eigen::Vector3d AfrontMeshing::getGrowDirection(const pcl::PointXYZ &p, const pcl::PointNormal &mp, const MeshTraits::FaceData &fd) const
  {
    Eigen::Vector3d v1, v2, v3, norm;
    v1 << (mp.x - p.x), (mp.y - p.y), (mp.z - p.z);
    norm << fd.normal_x, fd.normal_y, fd.normal_z;
    v2 = norm.cross(v1).normalized();

    // Check direction from origin of triangle
    v3 << (fd.x - mp.x), (fd.y - mp.y), (fd.z - mp.z);
    if (v2.dot(v3) > 0.0)
      v2 *= -1.0;

    return v2;
  }

  AfrontMeshing::GrowDistanceResults AfrontMeshing::getGrowDistance(const pcl::PointNormal &mp, const double &edge_length, const double &min_length, const double &max_length) const
  {
    GrowDistanceResults gdr;
    std::vector<int> K;
    std::vector<float> K_dist;

    int cnt = mls_cloud_tree_->radiusSearch(mp, reduction_ * max_length, K, K_dist);
    gdr.valid = false;
    gdr.max_curv = getMaxCurvature(K);
    if (cnt > 0)
    {
      gdr.valid = true;
      gdr.ideal = 2.0 * std::sin(rho_ / 2.0) / gdr.max_curv;
      double estimated = rho_ / gdr.max_curv;
      double max = min_length * reduction_;
      double min = max_length / reduction_;
      if (estimated > gdr.ideal)
        estimated = gdr.ideal;

      if (estimated > edge_length)
      {
        double ratio = estimated / edge_length;
        if (ratio > reduction_)
          estimated = reduction_ * edge_length;

      }
      else
      {
        double ratio = edge_length / estimated;
        if (ratio > reduction_)
          estimated = edge_length / reduction_;
      }

      gdr.estimated = estimated;
      gdr.l = std::sqrt(pow(estimated, 2.0) - pow(edge_length / 2.0, 2.0));

      //      if (min < max)
      //      {
      //        if (estimated < min)
      //          estimated = min;
      //        else if (estimated > max)
      //          estimated = max;
      //      }
      //      else
      //      {
      //        if (estimated > edge_length)
      //        {
      //          double ratio = estimated / edge_length;
      //          if (ratio > reduction_)
      //            estimated = reduction_ * edge_length;

      //        }
      //        else
      //        {
      //          double ratio = edge_length / estimated;
      //          if (ratio > reduction_)
      //            estimated = edge_length / reduction_;
      //        }
      //      }
    }

    return gdr;
  }

  pcl::PointNormal AfrontMeshing::getPredictedVertex(const pcl::PointNormal &mp, const Eigen::Vector3d &d, const double &l) const
  {
    pcl::PointXYZ p;
    p.x = mp.x + l * d(0);
    p.y = mp.y + l * d(1);
    p.z = mp.z + l * d(2);

    // Project new point onto the mls surface
    return mls_.samplePoint(p);
  }

  float AfrontMeshing::getMaxCurvature(const std::vector<int>& indices) const
  {
    float curv = 0.0;
    for(int i = 0; i < indices.size(); ++i)
    {
      float c = getCurvature(indices[i]);
      if (c > curv)
        curv = c;
    }
    return curv;
  }

  Eigen::Vector2d AfrontMeshing::getMinMaxEdgeLength(const VertexIndex &v1, const VertexIndex &v2) const
  {
    Eigen::Vector2d result;
    result[0] = std::numeric_limits<double>::max();
    result[1] = 0.0;

    // Get half edges for v1
    {
      OHEAVC       circ_oheav     = mesh_.getOutgoingHalfEdgeAroundVertexCirculator(v1);
      const OHEAVC circ_oheav_end = circ_oheav;
      do
      {
        HalfEdgeIndex he = circ_oheav.getTargetIndex();
        pcl::PointXYZ p1, p2;
        p1 = mesh_.getVertexDataCloud()[mesh_.getOriginatingVertexIndex(he).get()];
        p2 = mesh_.getVertexDataCloud()[mesh_.getTerminatingVertexIndex(he).get()];
        double d = getEdgeLength(p1, p2);
        if (d > result[1])
          result[1] = d;

        if (d < result[0])
          result[0] = d;
      } while (++circ_oheav != circ_oheav_end);
    }
    // Get half edges for v2
    {
      OHEAVC       circ_oheav     = mesh_.getOutgoingHalfEdgeAroundVertexCirculator(v2);
      const OHEAVC circ_oheav_end = circ_oheav;
      do
      {
        HalfEdgeIndex he = circ_oheav.getTargetIndex();
        pcl::PointXYZ p1, p2;
        p1 = mesh_.getVertexDataCloud()[mesh_.getOriginatingVertexIndex(he).get()];
        p2 = mesh_.getVertexDataCloud()[mesh_.getTerminatingVertexIndex(he).get()];
        double d = getEdgeLength(p1, p2);
        if (d > result[1])
          result[1] = d;

        if (d < result[0])
          result[0] = d;
      } while (++circ_oheav != circ_oheav_end);
    }

    return result;
  }


  AfrontMeshing::MeshTraits::FaceData AfrontMeshing::createFaceData(const pcl::PointXYZ &p1, const pcl::PointXYZ &p2, const pcl::PointXYZ &p3) const
  {
    MeshTraits::FaceData center_pt;
    Eigen::Vector3d v1, v2, norm;
    center_pt.x = (p1.x + p2.x + p3.x)/3.0;
    center_pt.y = (p1.y + p2.y + p3.y)/3.0;
    center_pt.z = (p1.z + p2.z + p3.z)/3.0;

    v1 << (p2.x - p1.x), (p2.y - p1.y), (p2.z - p1.z);
    v2 << (center_pt.x - p1.x), (center_pt.y - p1.y), (center_pt.z - p1.z);
    norm = v2.cross(v1).normalized();

    center_pt.normal_x = norm(0);
    center_pt.normal_y = norm(1);
    center_pt.normal_z = norm(2);

    return center_pt;
  }

  AfrontMeshing::MeshTraits::FaceData AfrontMeshing::createFaceData(const pcl::PointNormal &p1, const pcl::PointNormal &p2, const pcl::PointNormal &p3) const
  {
    MeshTraits::FaceData center_pt;
    Eigen::Vector3d v1, v2, norm;
    center_pt.x = (p1.x + p2.x + p3.x)/3.0;
    center_pt.y = (p1.y + p2.y + p3.y)/3.0;
    center_pt.z = (p1.z + p2.z + p3.z)/3.0;

    v1 << (p2.x - p1.x), (p2.y - p1.y), (p2.z - p1.z);
    v2 << (center_pt.x - p1.x), (center_pt.y - p1.y), (center_pt.z - p1.z);
    norm = v2.cross(v1).normalized();

    center_pt.normal_x = norm(0);
    center_pt.normal_y = norm(1);
    center_pt.normal_z = norm(2);

    return center_pt;
  }

  void AfrontMeshing::printVertices() const
  {
    std::cout << "Vertices:\n   ";
    for (unsigned int i=0; i<mesh_.sizeVertices(); ++i)
    {
      std::cout << mesh_.getVertexDataCloud()[i] << " ";
    }
    std::cout << std::endl;
  }

  void AfrontMeshing::printFaces() const
  {
    std::cout << "Faces:\n";
    for (unsigned int i=0; i<mesh_.sizeFaces(); ++i)
    {
      printFace(FaceIndex(i));
    }
  }

  void AfrontMeshing::printEdge(const HalfEdgeIndex &half_edge) const
  {
    std::cout << "  "
              << mesh_.getVertexDataCloud() [mesh_.getOriginatingVertexIndex(half_edge).get()]
              << " "
              << mesh_.getVertexDataCloud() [mesh_.getTerminatingVertexIndex(half_edge).get()]
              << std::endl;
  }

  void AfrontMeshing::printFace(const FaceIndex &idx_face) const
  {
    // Circulate around all vertices in the face
    VAFC       circ     = mesh_.getVertexAroundFaceCirculator(idx_face);
    const VAFC circ_end = circ;
    std::cout << "  ";
    do
    {
      std::cout << mesh_.getVertexDataCloud() [circ.getTargetIndex().get()] << " ";
    } while (++circ != circ_end);
    std::cout << std::endl;
  }

}

