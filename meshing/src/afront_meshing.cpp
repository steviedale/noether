#include <pcl/geometry/mesh_conversion.h>
#include <pcl/conversions.h>
#include <meshing/afront_meshing.h>

//template class PCL_EXPORTS afront_meshing::MLSSampling<pcl::PointXYZ, pcl::PointNormal>;
namespace afront_meshing
{
  pcl::PointNormal MLSSampling::samplePoint(const pcl::PointXYZ& pt)
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
    projectPointToMLSSurface(u_disp, v_disp,
                             mls_results_[input_index].u_axis, mls_results_[input_index].v_axis,
                             mls_results_[input_index].plane_normal,
                             mls_results_[input_index].mean,
                             mls_results_[input_index].curvature,
                             mls_results_[input_index].c_vec,
                             mls_results_[input_index].num_neighbors,
                             result_point, result_normal);

    // Copy additional point information if available
    copyMissingFields(input_->points[input_index], result_point);

    result_point.normal_x = result_normal.normal_x;
    result_point.normal_y = result_normal.normal_y;
    result_point.normal_z = result_normal.normal_z;
    result_point.curvature = result_normal.curvature;

    return result_point;
  }

  pcl::PointNormal MLSSampling::samplePoint(const pcl::PointNormal& pt)
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
    std::cout << "starting meshing\n";
    int start_pt = input_cloud_->points.size() / 2;

    // use firt point to create first edge.  Get curvature and ideal length (rho/k)
    pcl::PointNormal pt1_norm = cloud_normals_->points[start_pt];
    pcl::PointXYZ pt1(pt1_norm.x, pt1_norm.y, pt1_norm.z);
    double k1 = getCurvature(start_pt); //pt1.curvature;
    double l1 = rho_ / k1;

    std::cout << "k1 and L1 value " << k1 << " " << l1 <<  "\n";

    std::vector<int> K;
    std::vector<float> K_dist;

    // search for the nearest neighbor
    pcl::PointXYZ search_pt(pt1.x, pt1.y, pt1.z);
    input_cloud_tree_->nearestKSearch(search_pt, 2, K, K_dist);

    // use l1 and nearest neighbor to extend edge
    pcl::PointNormal pt2_norm = cloud_normals_->points[K[1]];
    pcl::PointXYZ pt2(pt2_norm.x, pt2_norm.y, pt2_norm.z);
    pcl::PointXYZ diff;
    diff.x = pt2.x - pt1.x;
    diff.y = pt2.y - pt1.y;
    diff.z = pt2.z - pt1.z;

    double length = sqrt(diff.x * diff.x +  diff.y * diff.y + diff.z * diff.z);

    // modify point 2
    pt2.x = diff.x * l1 / length + pt1.x;
    pt2.y = diff.y * l1 / length + pt1.y;
    pt2.z = diff.z * l1 / length + pt1.z;

    // Project point 2 onto the MLS surface
    pt2_norm = mls_.samplePoint(pt2);
    pt2.x = pt2_norm.x;
    pt2.y = pt2_norm.y;
    pt2.z = pt2_norm.z;

    std::cout << "pt2 done\n";

    //TODO: need to check pt2 curvature to make sure that the distance traveled is valid
    // (or precompute by finding average curvature around pt1)

    // based on pt1 and pt2, perform radius search of points, average curvatures, and make new triangle
    pcl::PointXYZ pt3;
    pt3.x = (pt1.x + pt2.x) / 2.0;
    pt3.y = (pt1.y + pt2.y) / 2.0;
    pt3.z = (pt1.z + pt2.z) / 2.0;

    // radius search using pt3 (midpoint) and l1
    input_cloud_tree_->radiusSearch(pt3, l1, K, K_dist);

    std::cout << "pt2 radius search\n";

    // calculate average curvature
    double curv = getAverageCurvature(K);
    l1 = rho_ / curv;

    pcl::PointNormal pt3_norm = mls_.samplePoint(pt3);

    // peform cross product of diff and pt3 normal to get the new triangle extension
    // because diff is not normalized, need to normalize the length prior to modifying pt3
    pt3.x = pt3.x + ((diff.y * pt3_norm.z) - (diff.z * pt3_norm.y)) * l1 / length;
    pt3.y = pt3.y - ((diff.x * pt3_norm.z) + (diff.z * pt3_norm.x)) * l1 / length;
    pt3.z = pt3.z + ((diff.x * pt3_norm.y) - (diff.y * pt3_norm.x)) * l1 / length;

    // Now that we have a new point, need to project the point onto the MLS surface
    pt3_norm = mls_.samplePoint(pt3);
    pt3.x = pt3_norm.x;
    pt3.y = pt3_norm.y;
    pt3.z = pt3_norm.z;

    std::cout << "pt3 done\n";

    // add vertex points
    VertexIndices vi;
    vi.push_back(mesh_.addVertex(pt1));
    vi.push_back(mesh_.addVertex(pt2));
    vi.push_back(mesh_.addVertex(pt3));

    // Find center of face and normal direction
    MeshTraits::FaceData face_data = createFaceData(pt1, pt2, pt3);
    std::cout << "Face Data:" << face_data << std::endl;

    // Add new face
    mesh_.addFace(vi[0], vi[1], vi[2], face_data);


    // Print out half-edges
    {
      std::cout << "Circulate around the boundary half-edges:" << std::endl;
      const HalfEdgeIndex& idx_he_boundary = mesh_.getOutgoingHalfEdgeIndex (vi[0]);
      IHEAFC       circ_iheaf     = mesh_.getInnerHalfEdgeAroundFaceCirculator (idx_he_boundary);
      const IHEAFC circ_iheaf_end = circ_iheaf;
      do
      {
        printEdge (mesh_, circ_iheaf.getTargetIndex());
      } while (++circ_iheaf != circ_iheaf_end);
    }

    // Grow triangle from each each half edge
    VertexIndex start = vi[0];
    for (int i = 0; i < 11; ++i)
    {
      const HalfEdgeIndex& idx_he_boundary = mesh_.getOutgoingHalfEdgeIndex(start);
      IHEAFC       circ_iheaf     = mesh_.getInnerHalfEdgeAroundFaceCirculator(idx_he_boundary);
      const IHEAFC circ_iheaf_end = circ_iheaf;

//      do
//      {
        std::cout << "Created new face\n";
        HalfEdgeIndex half_edge = circ_iheaf.getTargetIndex();
        start = growEdge(half_edge);

//      } while (++circ_iheaf != circ_iheaf_end);
    }

    //////////////////////////////////////////////////////////////////////////////

    std::cout << "Outgoing half-edges of vertex 0:" << std::endl;
    OHEAVC       circ_oheav     = mesh_.getOutgoingHalfEdgeAroundVertexCirculator(vi[0]);
    const OHEAVC circ_oheav_end = circ_oheav;
    do
    {
      printEdge (mesh_, circ_oheav.getTargetIndex());
    } while (++circ_oheav != circ_oheav_end);



  }

  AfrontMeshing::VertexIndex AfrontMeshing::growEdge(const HalfEdgeIndex &half_edge)
  {
    // Local Variables
    VertexIndices vi;
    pcl::PointXYZ p[3], mid_pt;
    std::vector<int> K;
    std::vector<float> K_dist;
    double curv, l;

    // Print Half Edge Data
    printEdge(mesh_, half_edge);

    // Get Afront FaceData
    FaceIndex face_indx = mesh_.getFaceIndex(mesh_.getOppositeHalfEdgeIndex(half_edge));
    MeshTraits::FaceData fd = mesh_.getFaceDataCloud()[face_indx.get()];

    // Get Half Edge Vertexs
    vi.push_back(mesh_.getOriginatingVertexIndex(half_edge));
    vi.push_back(mesh_.getTerminatingVertexIndex(half_edge));
    p[0] = mesh_.getVertexDataCloud()[vi[0].get()];
    p[1] = mesh_.getVertexDataCloud()[vi[1].get()];

    // find the mid point of the edge
    mid_pt.x = (p[0].x + p[1].x)/2.0;
    mid_pt.y = (p[0].y + p[1].y)/2.0;
    mid_pt.z = (p[0].z + p[1].z)/2.0;

    // Calculate direction vector to move (v2)
    Eigen::Vector3d v1, v2, v3, norm;
    v1 << (p[1].x - p[0].x), (p[1].y - p[0].y), (p[1].z - p[0].z);
    norm << fd.normal_x, fd.normal_y, fd.normal_z;
    v2 = norm.cross(v1).normalized();

    // Check direction from origin of triangle
    v3 << (fd.x - mid_pt.x), (fd.y - mid_pt.y), (fd.z - mid_pt.z);
    if (v2.dot(v3) > 0.0)
      v2 *= -1.0;

    // radius search using pt3 (midpoint) and l1
    input_cloud_tree_->radiusSearch(mid_pt, rho_, K, K_dist);

    // calculate average curvature
    curv = getAverageCurvature(K);
    l = rho_ / curv;

    // Get new point
    p[2].x = mid_pt.x + l * v2(0);
    p[2].y = mid_pt.y + l * v2(1);
    p[2].z = mid_pt.z + l * v2(2);

    // Project new point onto the mls surface
    pcl::PointNormal new_vert = mls_.samplePoint(p[2]);
    p[2].x = new_vert.x;
    p[2].y = new_vert.y;
    p[2].z = new_vert.z;

    // Add new vertex
    vi.push_back(mesh_.addVertex(p[2]));

    // Add new face
    MeshTraits::FaceData new_fd = createFaceData(p[0], p[1], p[2]);
    FaceIndex findx =  mesh_.addFace(vi[0], vi[1], vi[2], new_fd);
    return vi.back();
  }

  double AfrontMeshing::getCurvature(int index)
  {
    if(index >= curvatures_->points.size() )
    {
      return -1.0;
    }

    double x = curvatures_->points[index].principal_curvature[0];
    double y = curvatures_->points[index].principal_curvature[1];
    double z = curvatures_->points[index].principal_curvature[2];
    double min;
    min = x > y ? x : y;
    min = z > min ? z : min;

    double curv = min / (x + y + z);
    return curv;
  }

  double AfrontMeshing::getAverageCurvature(std::vector<int>& indices)
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

