#include <pcl/geometry/mesh_conversion.h>
#include <pcl/conversions.h>
#include <meshing/afront_meshing.h>

//template class PCL_EXPORTS afront_meshing::MLSSampling<pcl::PointXYZ, pcl::PointNormal>;
namespace afront_meshing
{
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

    result_point.normal_x = result_normal.normal_x;
    result_point.normal_y = result_normal.normal_y;
    result_point.normal_z = result_normal.normal_z;
    result_point.curvature = result_normal.curvature;

    return result_point;
  }

  pcl::PointNormal MLSSampling::samplePoint(const pcl::PointNormal& pt) const
  {
    pcl::PointXYZ search_pt(pt.x, pt.y, pt.z);
    return samplePoint(search_pt);
  }

  void AfrontMeshing::setInputCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
  {
    input_cloud_ = cloud;//pcl::PointCloud<pcl::PointXYZ>::Ptr(cloud);
  }

  bool AfrontMeshing::computeGuidanceField()
  {
    input_cloud_tree_ = pcl::search::KdTree<pcl::PointXYZ>::Ptr(new pcl::search::KdTree<pcl::PointXYZ>);
    //input_cloud_tree_->setInputCloud(input_cloud_);

    std::cout << "here1\n";
    //pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);

    // Calculate MLS
    cloud_normals_ = pcl::PointCloud<pcl::PointNormal>::Ptr(new pcl::PointCloud<pcl::PointNormal>());
    //pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointNormal> mls;
    mls_.setComputeNormals(true);
    mls_.setInputCloud(input_cloud_);
    mls_.setPolynomialFit(true);
    mls_.setSearchMethod(input_cloud_tree_);
    mls_.setSearchRadius(r_);
    mls_.setUpsamplingMethod(pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointNormal>::DISTINCT_CLOUD);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    cloud->push_back(input_cloud_->points[0]);
    mls_.setDistinctCloud(input_cloud_);
    mls_.process(*cloud_normals_);

    std::cout << "here2\n";
    // Calculate Curvatures
    pcl::PrincipalCurvaturesEstimation<pcl::PointXYZ, pcl::PointNormal, pcl::PrincipalCurvatures> principal_curvatures_estimation;

    // Provide the original point cloud (without normals)
    principal_curvatures_estimation.setInputCloud (input_cloud_);

    // Provide the point cloud with normals
    if(cloud_normals_->points.size() != input_cloud_->points.size())
    {
      return false;
    }
    std::cout << "done calculating normals\n";
    principal_curvatures_estimation.setInputNormals (cloud_normals_);

    // Use the same KdTree from the normal estimation
    principal_curvatures_estimation.setSearchMethod (input_cloud_tree_);
    principal_curvatures_estimation.setRadiusSearch (r_);

    curvatures_ = pcl::PointCloud<pcl::PrincipalCurvatures>::Ptr(new pcl::PointCloud<pcl::PrincipalCurvatures>());
    principal_curvatures_estimation.compute (*curvatures_);

    std::cout << "done calculating curvatures\n";
    input_cloud_tree_->setInputCloud(input_cloud_);
    return true;
  }

  pcl::PolygonMesh AfrontMeshing::getMesh()
  {
    pcl::PolygonMesh out_mesh;
    pcl::geometry::toFaceVertexMesh(mesh_, out_mesh);
    return out_mesh;
  }

  void AfrontMeshing::startMeshing()
  {
    std::vector<int> K;
    std::vector<float> K_dist;

    std::cout << "starting meshing\n";

    // Create first triangle
    VertexIndex start = createFirstTriangle(2.5, 2.2, 0);

    mesh_vertex_data_ = pcl::PointCloud<MeshTraits::VertexData>::Ptr(&mesh_.getVertexDataCloud());
    mesh_tree_ = pcl::search::KdTree<MeshTraits::VertexData>::Ptr(new pcl::search::KdTree<MeshTraits::VertexData>);
    mesh_tree_->setInputCloud(mesh_vertex_data_);

//    // Print out half-edges
//    {
//      std::cout << "Circulate around the boundary half-edges:" << std::endl;
//      const HalfEdgeIndex& idx_he_boundary = mesh_.getOutgoingHalfEdgeIndex (vi[0]);
//      IHEAFC       circ_iheaf     = mesh_.getInnerHalfEdgeAroundFaceCirculator (idx_he_boundary);
//      const IHEAFC circ_iheaf_end = circ_iheaf;
//      do
//      {
//        printEdge (mesh_, circ_iheaf.getTargetIndex());
//      } while (++circ_iheaf != circ_iheaf_end);
//    }

    // Grow triangle from each each half edge
    for (int i = 0; i < 250; ++i)
    {
      const HalfEdgeIndex& idx_he_boundary = mesh_.getOutgoingHalfEdgeIndex(start);
      IHEAFC       circ_iheaf     = mesh_.getInnerHalfEdgeAroundFaceCirculator(idx_he_boundary);
      const IHEAFC circ_iheaf_end = circ_iheaf;

//      do
//      {
        std::cout << "Created new face\n";

        // Check if it can create a triangle with previous half edge
        HalfEdgeIndex half_edge = circ_iheaf.getTargetIndex();
        --circ_iheaf;
        HalfEdgeIndex prev_half_edge = circ_iheaf.getTargetIndex();

        if (cutEar(prev_half_edge, half_edge))
        {
          start = mesh_.getOriginatingVertexIndex(prev_half_edge);
          continue;
        }

        // Check if it can create a triangle with next half edge
        ++circ_iheaf;
        ++circ_iheaf;
        HalfEdgeIndex next_half_edge = circ_iheaf.getTargetIndex();

        if (cutEar(half_edge, next_half_edge))
        {
          start = mesh_.getOriginatingVertexIndex(half_edge);
          continue;
        }

        // If we can not cut ear then try and grow.
        PredictVertexResults pv = predictVertex(half_edge);
        TriangleToCloseResults tc = isTriangleToClose(pv);
        if (tc.valid)
        {
          start = grow(pv);
        }
        else
        {
          start = merge(tc);
        }

//      } while (++circ_iheaf != circ_iheaf_end);
    }
  }

  AfrontMeshing::VertexIndex AfrontMeshing::createFirstTriangle(const double &x, const double &y, const double &z)
  {
    std::vector<int> K;
    std::vector<float> K_dist;

    pcl::PointXYZ middle_pt(x, y, z);
    input_cloud_tree_->nearestKSearch(middle_pt, 1, K, K_dist);

    pcl::PointNormal p1_norm = mls_.samplePoint(cloud_normals_->points[K[0]]);
    pcl::PointXYZ p1(p1_norm.x, p1_norm.y, p1_norm.z);

    // Get the allowed grow distance
    double l = getGrowDistance(p1);

    // search for the nearest neighbor
    input_cloud_tree_->nearestKSearch(p1, 2, K, K_dist);

    // use l1 and nearest neighbor to extend edge
    pcl::PointNormal dp = cloud_normals_->points[K[1]];
    pcl::PointXYZ p2, p3, mp;
    Eigen::Vector3d v1, v2, norm;

    v1 << (dp.x - p1.x), (dp.y - p1.y), (dp.z - p1.z);
    v1 = v1.normalized();
    norm << dp.normal_x, dp.normal_y, dp.normal_z;

    p2 = getPredictedVertex(p1, v1, l);
    mp = getMidPoint(p1, p2);

    v2 = norm.cross(v1).normalized();

    l = getGrowDistance(mp);
    p3 = getPredictedVertex(mp, v2, l);

    MeshTraits::FaceData fd = createFaceData(p1, p2, p3);
    VertexIndices vi;
    vi.push_back(mesh_.addVertex(p1));
    vi.push_back(mesh_.addVertex(p2));
    vi.push_back(mesh_.addVertex(p3));
    mesh_.addFace(vi[0], vi[1], vi[2], fd);

    return vi[0];
  }

  bool AfrontMeshing::cutEar(const HalfEdgeIndex &half_edge1, const HalfEdgeIndex &half_edge2)
  {
    VertexIndices vi;
    pcl::PointXYZ p1, p2, p3;
    Eigen::Vector3d v1, v2, v3, cross;
    double dot, sina, cosa, bottom, theta1, theta2;

    // first check and make sure both half edges are not associated the same face
    if (mesh_.getOppositeFaceIndex(half_edge1) == mesh_.getOppositeFaceIndex(half_edge2))
      return false;

    // Debug info
    std::cout << "Attempting to perform ear cut: " << std::endl;
    printEdge(mesh_, half_edge1);
    printEdge(mesh_, half_edge2);

    vi.push_back(mesh_.getOriginatingVertexIndex(half_edge1));
    vi.push_back(mesh_.getTerminatingVertexIndex(half_edge1));
    vi.push_back(mesh_.getTerminatingVertexIndex(half_edge2));
    p1 = mesh_.getVertexDataCloud()[vi[0].get()];
    p2 = mesh_.getVertexDataCloud()[vi[1].get()];
    p3 = mesh_.getVertexDataCloud()[vi[2].get()];

    v1 << (p1.x - p2.x), (p1.y - p2.y), (p1.z - p2.z);
    v2 << (p3.x - p2.x), (p3.y - p2.y), (p3.z - p2.z);
    v3 << (p1.x - p3.x), (p1.y - p3.y), (p1.z - p3.z);

    // Check first angle of triangle
    dot = v1.dot(v2);
    cross = v1.cross(v2);
    bottom = v1.norm() * v2.norm();
    sina = cross.norm()/bottom;
    cosa = dot/bottom;
    theta1 = atan2(sina, cosa);

    if (theta1 > 1.22173) // The paper only allows ear cutting if all angles are less than 70 degress
      return false;

    // Check second angle of triangle
    v2 *= -1.0;
    dot = v2.dot(v3);
    cross = v2.cross(v3);
    bottom = v2.norm() * v3.norm();
    sina = cross.norm()/bottom;
    cosa = dot/bottom;
    theta2 = atan2(sina, cosa);

    if (theta2 > 1.22173) // The paper only allows ear cutting if all angles are less than 70 degress
      return false;

    // Check third angle of triangle
    if ((M_PI - theta1 - theta2) > 1.22173)
      return false;

    // The criteria has been met to perform ear cutting;
    // Add new face
    std::cout << "Ear Cut" << std::endl;
    MeshTraits::FaceData new_fd = createFaceData(p1, p2, p3);
    mesh_.addFace(vi[0], vi[1], vi[2], new_fd);

    return true;
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

    // find the mid point of the edge
    result.mp = getMidPoint(result.p[0], result.p[1]);

    // Calculate direction vector to move
    result.d = getGrowDirection(result.p[0], result.mp, fd);

    // Get the allowed grow distance
    result.l = getGrowDistance(result.mp);

    // Return predicted vertex
    result.p[2] = getPredictedVertex(result.mp, result.d, result.l);

    return result;
  }

  AfrontMeshing::TriangleToCloseResults AfrontMeshing::isTriangleToClose(const PredictVertexResults &pvr) const
  {
    TriangleToCloseResults result;
    std::vector<int> K;
    std::vector<float> K_dist;

    // search for the nearest neighbor
    pcl::PointXYZ search_pt(pvr.p[2]);
    mesh_tree_->nearestKSearch(search_pt, 1, K, K_dist);

    result.pvr = pvr;
    result.dist = K_dist[0];
    MeshTraits::VertexData &data = mesh_vertex_data_->at(K[0]);
    result.closest = mesh_.getVertexIndex(data);
    result.valid = true;

    if (result.dist < pow(pvr.l/2.0, 2))
      result.valid = false;

    return result;
  }

  AfrontMeshing::VertexIndex AfrontMeshing::grow(const PredictVertexResults &pvr)
  {
    // Add new face
    MeshTraits::FaceData new_fd = createFaceData(pvr.p[0], pvr.p[1], pvr.p[2]);
    VertexIndex vi = mesh_.addVertex(pvr.p[2]);
    mesh_.addFace(pvr.vi[0], pvr.vi[1], vi, new_fd);
    mesh_tree_->setInputCloud(mesh_vertex_data_); // This may need to be replaced with using an octree.
    return vi;
  }

  AfrontMeshing::VertexIndex AfrontMeshing::merge(const TriangleToCloseResults &tc)
  {
    MeshTraits::FaceData new_fd = createFaceData(tc.pvr.p[0], tc.pvr.p[1], mesh_.getVertexDataCloud()[tc.closest.get()]);
    mesh_.addFace(tc.pvr.vi[0], tc.pvr.vi[1], tc.closest, new_fd);
    mesh_tree_->setInputCloud(mesh_vertex_data_); // This may need to be replaced with using an octree.
    return tc.closest;
  }

  double AfrontMeshing::getCurvature(const int index) const
  {
    if(index >= curvatures_->points.size())
    {
      return -1.0;
    }

    double x = std::abs(curvatures_->points[index].principal_curvature[0]);
    double y = std::abs(curvatures_->points[index].principal_curvature[1]);
    double z = std::abs(curvatures_->points[index].principal_curvature[2]);
    double min;
    min = x > y ? x : y;
    min = z > min ? z : min;

    double curv = min / (x + y + z);
    return curv;
  }

  pcl::PointXYZ AfrontMeshing::getMidPoint(const pcl::PointXYZ &p1, const pcl::PointXYZ &p2) const
  {
    return pcl::PointXYZ((p1.x + p2.x)/2.0, (p1.y + p2.y)/2.0,(p1.z + p2.z)/2.0);
  }

  Eigen::Vector3d AfrontMeshing::getGrowDirection(const pcl::PointXYZ &p, const pcl::PointXYZ &mp, const MeshTraits::FaceData &fd) const
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

  double AfrontMeshing::getGrowDistance(const pcl::PointXYZ &mp) const
  {
    std::vector<int> K;
    std::vector<float> K_dist;

    int cnt = input_cloud_tree_->radiusSearch(mp, rho_, K, K_dist);
    double curv = getAverageCurvature(K);
    if (cnt == 0)
      return 0.0;
    else
      return rho_ / curv;

  }

  pcl::PointXYZ AfrontMeshing::getPredictedVertex(const pcl::PointXYZ &mp, const Eigen::Vector3d &d, const double &l) const
  {
    pcl::PointXYZ p;
    p.x = mp.x + l * d(0);
    p.y = mp.y + l * d(1);
    p.z = mp.z + l * d(2);

    // Project new point onto the mls surface
    pcl::PointNormal new_vert = mls_.samplePoint(p);
    p.x = new_vert.x;
    p.y = new_vert.y;
    p.z = new_vert.z;

    return p;
  }

  double AfrontMeshing::getAverageCurvature(const std::vector<int>& indices) const
  {
    double curv = 0;
    for(int i = 0; i < indices.size(); ++i)
    {
      curv += getCurvature(indices[i]);
    }
    return (curv/indices.size());
  }


  AfrontMeshing::MeshTraits::FaceData AfrontMeshing::createFaceData(const pcl::PointXYZ p1, const pcl::PointXYZ p2, const pcl::PointXYZ p3)
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

}

