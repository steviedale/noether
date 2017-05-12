#include <pcl/geometry/mesh_conversion.h>
#include <pcl/conversions.h>
#include <meshing/afront_meshing.h>
#include <eigen3/Eigen/LU>

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

    std::printf("Minimum Curvature: %f\n", min);
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



  AfrontMeshing::AfrontMeshing() : mesh_vertex_data_(mesh_.getVertexDataCloud())
  {
    counter_ = 0;
    finished_ = false;
    snap_ = false;
  }

  bool AfrontMeshing::initMesher(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
  {
    std::printf("\x1B[36mInitializing Mesher!\n");
    input_cloud_ = cloud;

    // Generate the MLS surface
    if (!computeGuidanceField())
      return false;

    // Create first triangle
    createFirstTriangle(rand() % mls_cloud_->size());

    // Get the inital boundary of the mesh
    const HalfEdgeIndex& idx_he_boundary = mesh_.getOutgoingHalfEdgeIndex(mesh_.getVertexIndex(mesh_vertex_data_[0]));
    IHEAFC       circ_iheaf     = mesh_.getInnerHalfEdgeAroundFaceCirculator(idx_he_boundary);
    const IHEAFC circ_iheaf_end = circ_iheaf;
    do
    {
      HalfEdgeIndex he = circ_iheaf.getTargetIndex();
      queue_.push_back(he);
    } while (++circ_iheaf != circ_iheaf_end);

    return true;
  }

  bool AfrontMeshing::computeGuidanceField()
  {
    std::printf("\x1B[36mComputing Guidance Field!\n");

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
    if (mls_cloud_->empty())
      return false;

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

  void AfrontMeshing::generateMesh()
  {
    std::printf("\x1B[36mMeshing Started!\n");
    while (!finished_)
      stepMesh();
  }

  void AfrontMeshing::stepMesh()
  {
    if (finished_)
    {
      std::printf("\x1B[31mTried to step mesh after it has finished meshing!\n");
      return;
    }

    counter_ += 1;
    std::printf("\x1B[35mAdvancing Front: %lu\n", counter_);
    HalfEdgeIndex half_edge = queue_.front();
    queue_.pop_front();

    updateKdTree();

    FrontData front = getAdvancingFrontData(half_edge);
    CanCutEarResults ccer = canCutEar(front);
    if(viewer_)
    {
      // remove previouse iterations lines
      viewer_->removeShape("HalfEdge");
      viewer_->removeShape("NextHalfEdge");
      viewer_->removeShape("PrevHalfEdge");

      pcl::PointXYZ p1, p2, p3, p4;
      p1 = convertEigenToPCL(ccer.next.tri.p[0]);
      p2 = convertEigenToPCL(ccer.next.tri.p[1]);
      p3 = convertEigenToPCL(ccer.next.tri.p[2]);
      p4 = convertEigenToPCL(ccer.prev.tri.p[2]);

      viewer_->addLine<pcl::PointXYZ, pcl::PointXYZ>(p1, p2, 0, 255, 0, "HalfEdge");       // Green
      viewer_->addLine<pcl::PointXYZ, pcl::PointXYZ>(p2, p3, 255, 0, 0, "NextHalfEdge");   // Red
      viewer_->addLine<pcl::PointXYZ, pcl::PointXYZ>(p1, p4, 255, 0, 255, "PrevHalfEdge"); // Magenta
      viewer_->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 8, "HalfEdge");
      viewer_->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 8, "NextHalfEdge");
      viewer_->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 8, "PrevHalfEdge");
    }

    if (ccer.valid)
    {
      std::printf("\x1B[34m  Performed Ear Cut Opperation\n");
      cutEar(*ccer.valid);
    }
    else
    {
      // If we can not cut ear then try and grow.
      PredictVertexResults pv = predictVertex(front);
      if (pv.at_boundary)
      {
        std::printf("\x1B[32m  At Point Cloud Boundary\n");
        boundary_.push_back(front.he);
      }
      else
      {
        TriangleToCloseResults ttcr = isTriangleToClose(ccer, pv);
        if (ttcr.type == TriangleToCloseTypes::None)
        {
          std::printf("\x1B[32m  Performed Grow Opperation\n");
          grow(ccer, pv);
        }
        else
        {
          std::printf("\x1B[33m  Performed Topology Event Opperation\n");
          topologyEvent(ttcr);
        }
      }
    }

    if(queue_.size() == 0)
      finished_ = true;
  }

  void AfrontMeshing::updateKdTree()
  {
    mesh_vertex_data_copy_.reset();
    mesh_vertex_data_copy_ = pcl::PointCloud<MeshTraits::VertexData>::Ptr(new pcl::PointCloud<MeshTraits::VertexData>(mesh_.getVertexDataCloud()));

    mesh_tree_.reset();
    mesh_tree_ = pcl::search::KdTree<MeshTraits::VertexData>::Ptr(new pcl::search::KdTree<MeshTraits::VertexData>);
    mesh_tree_->setInputCloud(mesh_vertex_data_copy_);
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
    createFirstTriangle(K[0]);
  }

  void AfrontMeshing::createFirstTriangle(const int &index)
  {
    MLSSampling::SamplePointResults sp = mls_.samplePoint(mls_cloud_->points[index]);
    Eigen::Vector3f p1 = sp.point.getVector3fMap();

    // Get the allowed grow distance
    double d = 2 * std::sin(rho_/2) / sp.point.curvature;
    GrowDistanceResults gdr = getGrowDistance(p1, d, d, d);

    // search for the nearest neighbor
    std::vector<int> K;
    std::vector<float> K_dist;
    mls_cloud_tree_->nearestKSearch(sp.point, 2, K, K_dist);

    // use l1 and nearest neighbor to extend edge
    pcl::PointNormal dp;
    Eigen::Vector3f p2, p3, v1, v2, mp, norm;

    dp = mls_cloud_->points[K[1]];
    v1 = dp.getVector3fMap() - p1;
    v1 = v1.normalized();
    norm = dp.getNormalVector3fMap();

    if (snap_)
      p2 = (mls_cloud_->points[getPredictedVertex(p1, v1, gdr.l).closest]).getVector3fMap();
    else
      p2 = getPredictedVertex(p1, v1, gdr.l).point.getVector3fMap();

    mp = getMidPoint(p1, p2);
    d = getEdgeLength(p1, p2);

    v2 = norm.cross(v1).normalized();

    gdr = getGrowDistance(mp, d, d, d);
    p3 = getPredictedVertex(mp, v2, gdr.l).point.getVector3fMap();

    if (snap_)
      p3 = (mls_cloud_->points[getPredictedVertex(mp, v2, gdr.l).closest]).getVector3fMap();
    else
      p3 = getPredictedVertex(mp, v2, gdr.l).point.getVector3fMap();

    MeshTraits::FaceData fd = createFaceData(p1, p2, p3);
    VertexIndices vi;
    vi.push_back(mesh_.addVertex(pcl::PointXYZ(p1(0), p1(1), p1(2))));
    vi.push_back(mesh_.addVertex(pcl::PointXYZ(p2(0), p2(1), p2(2))));
    vi.push_back(mesh_.addVertex(pcl::PointXYZ(p3(0), p3(1), p3(2))));
    mesh_.addFace(vi[0], vi[1], vi[2], fd);
  }

  AfrontMeshing::CanCutEarResults AfrontMeshing::canCutEar(const FrontData &front) const
  {
    CanCutEarResults result;
    Eigen::Vector3f p3;

    result.front = front;

    //////////////////////////
    // Check Next Half Edge //
    //////////////////////////
    result.next.primary = front.he;
    result.next.secondary = getNextHalfEdge(front.he);
    result.next.valid = false;
    result.next.vi.push_back(front.vi[0]);
    result.next.vi.push_back(front.vi[1]);
    result.next.vi.push_back(mesh_.getTerminatingVertexIndex(result.next.secondary));
    p3 = (mesh_vertex_data_[result.next.vi[2].get()]).getVector3fMap();
    result.next.tri = getTriangleData(front, p3);

    // First check and make sure both half edges are not associated the same face
    if (result.next.tri.valid)
      if (result.next.tri.a < 1.22173 && result.next.tri.b < 1.22173 && result.next.tri.c < 1.22173)
        result.next.valid = true;

    //////////////////////////
    // Check Prev Half Edge //
    //////////////////////////
    result.prev.primary = front.he;
    result.prev.secondary = getPrevHalfEdge(front.he);
    result.prev.valid = false;
    result.prev.vi.push_back(front.vi[0]);
    result.prev.vi.push_back(front.vi[1]);
    result.prev.vi.push_back(mesh_.getOriginatingVertexIndex(result.prev.secondary));
    p3 = (mesh_vertex_data_[result.prev.vi[2].get()]).getVector3fMap();
    result.prev.tri = getTriangleData(front, p3);

    // First check and make sure both half edges are not associated the same face
    if (result.prev.tri.valid)
      if (result.prev.tri.a < 1.22173 && result.prev.tri.b < 1.22173 && result.prev.tri.c < 1.22173)
        result.prev.valid = true;

    ////////////////////////////////////////////
    // Review results and choose the best one //
    ////////////////////////////////////////////
    if (result.next.valid && result.prev.valid)
      if (result.next.tri.aspect_ratio < result.prev.tri.aspect_ratio)
        result.valid = &result.prev;
      else
        result.valid = &result.next;
    else if (result.next.valid)
      result.valid = &result.next;
    else if (result.prev.valid)
      result.valid = &result.prev;

    return result;
  }

  void AfrontMeshing::cutEar(const CanCutEarResult &data)
  {
    // Add new face
    MeshTraits::FaceData new_fd = createFaceData((mesh_vertex_data_[data.vi[0].get()]).getVector3fMap(),
                                                 (mesh_vertex_data_[data.vi[1].get()]).getVector3fMap(),
                                                 (mesh_vertex_data_[data.vi[2].get()]).getVector3fMap());
    HalfEdgeIndex temp;
    if (data.vi[2] == mesh_.getOriginatingVertexIndex(data.secondary))
        temp = getPrevHalfEdge(data.secondary);
    else
        temp = getPrevHalfEdge(data.primary);

    mesh_.addFace(data.vi[0], data.vi[1], data.vi[2], new_fd);

    queue_.push_back(getNextHalfEdge(temp));
    queue_.erase(std::remove_if(queue_.begin(), queue_.end(), [data](HalfEdgeIndex he){ return (he == data.secondary);}), queue_.end());
    boundary_.erase(std::remove_if(boundary_.begin(), boundary_.end(), [data](HalfEdgeIndex he){ return (he == data.secondary);}), boundary_.end());
  }

  AfrontMeshing::FrontData AfrontMeshing::getAdvancingFrontData(const HalfEdgeIndex &half_edge) const
  {
    FrontData result;
    result.he = half_edge;

    FaceIndex face_indx = mesh_.getOppositeFaceIndex(half_edge);
    MeshTraits::FaceData fd = mesh_.getFaceDataCloud()[face_indx.get()];

    // Get Half Edge Vertexs
    result.vi[0] = mesh_.getOriginatingVertexIndex(half_edge);
    result.vi[1] = mesh_.getTerminatingVertexIndex(half_edge);
    result.p[0] = (mesh_vertex_data_[result.vi[0].get()]).getVector3fMap();
    result.p[1] = (mesh_vertex_data_[result.vi[1].get()]).getVector3fMap();

    // Calculate the half edge length
    result.length = getEdgeLength(result.p[0], result.p[1]);

    // Get half edge midpoint
    result.mp = getMidPoint(result.p[0], result.p[1]);

    // Calculate the grow direction vector
    result.d = getGrowDirection(result.p[0], result.mp, fd);

    return result;
  }

  AfrontMeshing::PredictVertexResults AfrontMeshing::predictVertex(const FrontData &front) const
  {
    // Local Variables
    PredictVertexResults result;
    result.front = front;

    // Calculate min max of all lengths attached to half edge
    Eigen::Vector2d mm = getMinMaxEdgeLength(front.vi[0], front.vi[1]);

    // Get the allowed grow distance
    result.gdr = getGrowDistance(front.mp, front.length, mm[0], mm[1]);

    // Get predicted vertex
    result.pv = getPredictedVertex(front.mp, front.d, result.gdr.l);

    // Check and see if there any point in the grow direction of the front.
    // If not then it is at the boundary of the point cloud.
    result.at_boundary = true;
    Eigen::Vector3f v1 = (front.p[1] - front.mp).normalized();
    Eigen::Vector3f closest = (mls_cloud_->at(result.pv.closest)).getVector3fMap();
    Eigen::Vector3f v2 = closest - front.mp;
    double dot1 = v2.dot(front.d);
    double dot2 = v2.dot(v1);
    if ((dot1 > 0.00001) && (std::abs(dot2) <= front.length/2.0))
      result.at_boundary = false;

    // Get triangle Data
    if (snap_)
    {
      result.tri = getTriangleData(front, mls_cloud_->points[result.pv.closest].getVector3fMap());
//      std::printf("  At Boundary: %i\n", result.at_boundary);
//      std::printf("  Aspect Ratio: %f\n", result.tri.aspect_ratio);
      if (result.at_boundary == false && result.tri.aspect_ratio < 0.1)
        result.at_boundary = true;
//      std::printf("  At Boundary: %i\n", result.at_boundary);
    }
    else
      result.tri = getTriangleData(front, result.pv.point.getVector3fMap());

    return result;
  }

  AfrontMeshing::TriangleToCloseResults AfrontMeshing::isTriangleToClose(const CanCutEarResults &ccer, const PredictVertexResults &pvr) const
  {
      TriangleToCloseResults result;
      std::vector<int> K;
      std::vector<float> K_dist;

      result.pvr = pvr;
      result.ccer = ccer;
      result.type = TriangleToCloseTypes::None;

      // Before checking fence violation lets check previous and next half edge and make sure there is not an issue.
      // TODO: Need to improve such that two triangles get created for this case (ccer.next.tri.c <= 1.5708)
      // TODO: Need to improve where a new triangle gets created while modify the other for this case (pvr.tri.c >= ccer.next.tri.c)
      if (ccer.next.tri.valid)
        if (pvr.tri.c >= ccer.next.tri.c || ccer.next.tri.c <= 1.5708)
        {
          result.type = TriangleToCloseTypes::NextHalfEdge;
          return result;
        }

      if (ccer.prev.tri.valid)
        if(pvr.tri.b >= ccer.prev.tri.b || ccer.prev.tri.b <= 1.5708)
        {
          result.type = TriangleToCloseTypes::PrevHalfEdge;
          return result;
        }

      // search for the nearest neighbor
      pcl::PointXYZ search_pt(pvr.tri.p[2](0), pvr.tri.p[2](1), pvr.tri.p[2](2));
      mesh_tree_->radiusSearch(search_pt, 3.0 * pvr.gdr.ideal, K, K_dist);

      for( auto i = 0; i < K.size(); ++i)
      {
        MeshTraits::VertexData &data = mesh_vertex_data_.at(K[i]);
        VertexIndex vi = mesh_.getVertexIndex(data);

        OHEAVC       circ_oheav     = mesh_.getOutgoingHalfEdgeAroundVertexCirculator(vi);
        const OHEAVC circ_oheav_end = circ_oheav;

        do
        {

          HalfEdgeIndex he = circ_oheav.getTargetIndex();
          if (mesh_.isBoundary(he) && (he != pvr.front.he) && (he != ccer.prev.secondary) && (he != ccer.next.secondary))
          {
            DistPointToHalfEdgeResults new_dist = distPointToHalfEdge(pvr.tri.p[2], he);

            if (!checkFence(pvr.tri.p[0], pvr.tri.p[2], he))
            {
              if (result.type == TriangleToCloseTypes::None)
              {
                result.type = TriangleToCloseTypes::FenceViolation;
                result.dist = new_dist;
              }
              else
              {
                if (new_dist.line < result.dist.line)
                  result.dist = new_dist;
              }
            }

            if (result.type != TriangleToCloseTypes::FenceViolation && new_dist.line < pvr.gdr.l * 0.5)
            {
              if (result.type == TriangleToCloseTypes::None)
              {
                result.type = TriangleToCloseTypes::CloseProximity;
                result.dist = new_dist;
              }
              else
              {
                if (new_dist.line < result.dist.line)
                  result.dist = new_dist;
              }
            }
          }
        } while (++circ_oheav != circ_oheav_end);
      }

      return result;

//    TriangleToCloseRconst OHEAVC circ_oheav_end = circ_oheav;esults result;
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
  }

  AfrontMeshing::DistPointToHalfEdgeResults AfrontMeshing::distPointToHalfEdge(const Eigen::Vector3f p, const HalfEdgeIndex &half_edge) const
  {
    DistPointToHalfEdgeResults results;
    Eigen::Vector3f p1, p2;

    results.he = half_edge;
    p1 = (mesh_vertex_data_[mesh_.getOriginatingVertexIndex(half_edge).get()]).getVector3fMap();
    p2 = (mesh_vertex_data_[mesh_.getTerminatingVertexIndex(half_edge).get()]).getVector3fMap();
    Eigen::Vector3f v = p2 - p1;
    Eigen::Vector3f w = p - p1;

    double c1 = w.dot(v);
    double c2 = v.dot(v);
    results.start = getEdgeLength(p, p1);
    results.end = getEdgeLength(p, p2);

    if (c1 <= 0)
    {
      results.line = results.start;
    }
    else if (c2 <= c1)
    {
      results.line = results.end;
    }
    else
    {
      double b = c1 / c2;
      Eigen::Vector3f pb = p1 + b * v;
      results.line = getEdgeLength(p, pb);
    }

    return results;
  }

  bool AfrontMeshing::checkFence(const Eigen::Vector3f p1, const Eigen::Vector3f p2, const HalfEdgeIndex &half_edge) const
  {
    // Define parametric equation of fence plane
    Eigen::Vector3f he_p1, he_p2;
    he_p1 = (mesh_vertex_data_[mesh_.getOriginatingVertexIndex(half_edge).get()]).getVector3fMap();
    he_p2 = (mesh_vertex_data_[mesh_.getTerminatingVertexIndex(half_edge).get()]).getVector3fMap();

    MeshTraits::FaceData fd = mesh_.getFaceDataCloud()[mesh_.getOppositeFaceIndex(half_edge).get()];

    Eigen::Vector3f u = he_p2 - he_p1;
    Eigen::Vector3f v = fd.getNormalVector3fMap();
    v = v.normalized() * rho_;

    // Define parametric equation of line segment
    Eigen::Vector3f w = p2 - p1;

    Eigen::Matrix3f m;
    m << u, v, -w;

    Eigen::Vector3f x = p1 - he_p1;
    Eigen::Vector3f result = m.lu().solve(x);

    if (result(2) <= 1 && result(2) >= 0) // This checks if line segement intersects the plane
      if (result(0) <= 1 && result(0) >= 0) // This checks if intersection point is within the x range of the plane
        if (result(1) <= 1 && result(1) >= -1) // This checks if intersection point is within the y range of the plane
          return false;


    return true;
  }

  void AfrontMeshing::grow(const CanCutEarResults &ccer, const PredictVertexResults &pvr)
  {
    // Add new face
    MeshTraits::FaceData new_fd = createFaceData(pvr.tri.p[0], pvr.tri.p[1], pvr.tri.p[2]);
    pcl::PointXYZ np(pvr.tri.p[2](0), pvr.tri.p[2](1), pvr.tri.p[2](2));
    mesh_.addFace(pvr.front.vi[0], pvr.front.vi[1], mesh_.addVertex(np), new_fd);

    // Add new half edges to the queue
    queue_.push_back(getNextHalfEdge(ccer.prev.secondary));
    queue_.push_back(getPrevHalfEdge(ccer.next.secondary));
  }

  void AfrontMeshing::topologyEvent(const TriangleToCloseResults &ttcr)
  {
    if (ttcr.type == TriangleToCloseTypes::PrevHalfEdge)
    {
      std::printf("\x1B[32m\tForced Ear Cut Opperation with Previous Half Edge!\n");
      cutEar(ttcr.ccer.prev);
    }
    else if (ttcr.type == TriangleToCloseTypes::NextHalfEdge)
    {
      std::printf("\x1B[32m\tForced Ear Cut Opperation with Previous Half Edge!\n");
      cutEar(ttcr.ccer.next);
    }
    else if (ttcr.type == TriangleToCloseTypes::CloseProximity)
    {
      std::printf("\x1B[31m\tNeed to implement CloseProximity!\n");
    }
    else if (ttcr.type == TriangleToCloseTypes::FenceViolation)
    {
      std::printf("\x1B[31m\tMerging triangle due to FenceViolation!\n");
      merge(ttcr);
    }
  }

  void AfrontMeshing::merge(const TriangleToCloseResults &ttcr)
  {
    VertexIndex closest;
    if (ttcr.dist.end < ttcr.dist.start)
      closest = mesh_.getTerminatingVertexIndex(ttcr.dist.he);
    else
      closest = mesh_.getOriginatingVertexIndex(ttcr.dist.he);

    if (ttcr.dist.he == ttcr.ccer.prev.secondary || ttcr.dist.he == ttcr.ccer.next.secondary)
      std::printf("\x1B[31m\tError in AfrontMeshing::merge!\n");

    MeshTraits::FaceData new_fd = createFaceData(ttcr.pvr.tri.p[0], ttcr.pvr.tri.p[1], (mesh_vertex_data_[closest.get()]).getVector3fMap());
    mesh_.addFace(ttcr.pvr.front.vi[0], ttcr.pvr.front.vi[1], closest, new_fd);

    // Add new half edges to the queue
    queue_.push_back(getNextHalfEdge(ttcr.ccer.prev.secondary));
    queue_.push_back(getPrevHalfEdge(ttcr.ccer.next.secondary));
  }

  float AfrontMeshing::getCurvature(const int &index) const
  {
    if(index >= mls_cloud_->points.size())
      return -1.0;

    return mls_cloud_->at(index).curvature;
  }

  Eigen::Vector3f AfrontMeshing::getMidPoint(const Eigen::Vector3f &p1, const Eigen::Vector3f &p2) const
  {
    return (p1 + p2) / 2.0;
  }

  double AfrontMeshing::getEdgeLength(const Eigen::Vector3f &p1, const Eigen::Vector3f &p2) const
  {
    return (p2-p1).norm();
  }

  Eigen::Vector3f AfrontMeshing::getGrowDirection(const Eigen::Vector3f &p, const Eigen::Vector3f &mp, const MeshTraits::FaceData &fd) const
  {
    Eigen::Vector3f v1, v2, v3, norm;
    v1 = mp - p;
    norm = fd.getNormalVector3fMap();
    v2 = norm.cross(v1).normalized();

    // Check direction from origin of triangle
    v3 = fd.getVector3fMap() - mp;
    if (v2.dot(v3) > 0.0)
      v2 *= -1.0;

    return v2;
  }

  AfrontMeshing::GrowDistanceResults AfrontMeshing::getGrowDistance(const Eigen::Vector3f &mp, const double &edge_length, const double &min_length, const double &max_length) const
  {
    GrowDistanceResults gdr;
    pcl::PointNormal pn;

    pn.x = mp(0);
    pn.y = mp(1);
    pn.z = mp(2);
    int cnt = mls_cloud_tree_->radiusSearch(pn, reduction_ * max_length, gdr.k, gdr.k_dist);
    gdr.valid = false;
    gdr.max_curv = getMaxCurvature(gdr.k);
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

  MLSSampling::SamplePointResults AfrontMeshing::getPredictedVertex(const Eigen::Vector3f &mp, const Eigen::Vector3f &d, const double &l) const
  {
    Eigen::Vector3f p = mp + l * d;

    // Project new point onto the mls surface
    return mls_.samplePoint(pcl::PointXYZ(p(0), p(1), p(2)));
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
        p1 = mesh_vertex_data_[mesh_.getOriginatingVertexIndex(he).get()];
        p2 = mesh_vertex_data_[mesh_.getTerminatingVertexIndex(he).get()];
        double d = getEdgeLength(p1.getVector3fMap(), p2.getVector3fMap());
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
        p1 = mesh_vertex_data_[mesh_.getOriginatingVertexIndex(he).get()];
        p2 = mesh_vertex_data_[mesh_.getTerminatingVertexIndex(he).get()];
        double d = getEdgeLength(p1.getVector3fMap(), p2.getVector3fMap());
        if (d > result[1])
          result[1] = d;

        if (d < result[0])
          result[0] = d;
      } while (++circ_oheav != circ_oheav_end);
    }

    return result;
  }

  AfrontMeshing::TriangleData AfrontMeshing::getTriangleData(const FrontData &front, const Eigen::Vector3f p3) const
  {
    TriangleData result;
    Eigen::Vector3f v1, v2, v3, cross;
    double dot, sina, cosa, top, bottom;

    result.valid = true;

    v1 = front.p[1] - front.p[0];
    v2 = p3 - front.p[1];
    v3 = p3 - front.p[0];

    dot = v2.dot(front.d);
    if (dot <= 0.0)
      result.valid = false;

    result.A = v1.norm();
    result.B = v2.norm();
    result.C = v3.norm();


//    result.aspect_ratio = std::max(std::max(result.A, result.B), result.C)/std::min(std::min(result.A, result.B), result.C);


    // Calculate the first angle of triangle
    dot = v1.dot(v3);
    cross = v1.cross(v3);
    bottom = result.A * result.C;
    top = cross.norm();
    sina = cross.norm()/bottom;
    cosa = dot/bottom;
    result.b = atan2(sina, cosa);

    double area = 0.5 * top;
    result.aspect_ratio = (4 * area * std::sqrt(3)) / (result.A * result.A + result.B * result.B + result.C * result.C);

    // Check the second angle of triangle
    v1 *= -1.0;
    dot = v1.dot(v2);
    cross = v1.cross(v2);
    bottom = result.A * result.B;
    sina = cross.norm()/bottom;
    cosa = dot/bottom;
    result.c = atan2(sina, cosa);

    result.a = M_PI - result.b - result.c;

    // Store point information
    result.p[0] = front.p[0];
    result.p[1] = front.p[1];
    result.p[2] = p3;

    return result;
  }

  AfrontMeshing::MeshTraits::FaceData AfrontMeshing::createFaceData(const Eigen::Vector3f &p1, const Eigen::Vector3f &p2, const Eigen::Vector3f &p3) const
  {
    MeshTraits::FaceData center_pt;
    Eigen::Vector3f cp, v1, v2, norm;

    cp = (p1 + p2 + p3) / 3.0;
    v1 = p2 - p1;
    v2  = cp - p1;
    norm = v2.cross(v1).normalized();

    center_pt.x = cp(0);
    center_pt.y = cp(1);
    center_pt.z = cp(2);
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
      std::cout << mesh_vertex_data_[i] << " ";
    }
    std::cout << std::endl;
  }

  void AfrontMeshing::printFaces() const
  {
    std::cout << "Faces:\n";
    for (unsigned int i=0; i < mesh_.sizeFaces(); ++i)
      printFace(FaceIndex(i));
  }

  void AfrontMeshing::printEdge(const HalfEdgeIndex &half_edge) const
  {
    std::cout << "  "
              << mesh_vertex_data_ [mesh_.getOriginatingVertexIndex(half_edge).get()]
              << " "
              << mesh_vertex_data_ [mesh_.getTerminatingVertexIndex(half_edge).get()]
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
      std::cout << mesh_vertex_data_ [circ.getTargetIndex().get()] << " ";
    } while (++circ != circ_end);
    std::cout << std::endl;
  }

  pcl::PointXYZ AfrontMeshing::convertEigenToPCL(const Eigen::Vector3f &p) const
  {
    return pcl::PointXYZ(p(0), p(1), p(2));
  }

}

