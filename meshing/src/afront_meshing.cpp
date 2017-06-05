#include <pcl/geometry/mesh_conversion.h>
#include <eigen3/Eigen/LU>

#include <meshing/afront_meshing.h>
#include <chrono>

#include <pcl/io/pcd_io.h>

namespace afront_meshing
{
  AfrontMeshing::AfrontMeshing() : mesh_vertex_data_(mesh_.getVertexDataCloud())
  {
    #ifdef AFRONTDEBUG
    counter_ = 0;
    fence_counter_ = 0;
    #endif

    finished_ = false;
    snap_ = false;
  }

  bool AfrontMeshing::initMesher(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
  {
    std::printf("\x1B[36mInitializing Mesher (Size: %i)!\x1B[0m\n", (int)cloud->points.size());
    input_cloud_ = cloud;

    // Generate the MLS surface
    if (!computeGuidanceField())
    {
      std::printf("\x1B[31m\tFailed to compute Guidance Field! Try increasing radius.\x1B[0m\n");
      return false;
    }

    pcl::io::savePCDFile("mls.pcd", *mls_cloud_);

    // Create first triangle
    createFirstTriangle(rand() % mls_cloud_->size());

    // Get the inital boundary of the mesh
    const HalfEdgeIndex& idx_he_boundary = mesh_.getOutgoingHalfEdgeIndex(mesh_.getVertexIndex(mesh_vertex_data_[0]));
    IHEAFC       circ_iheaf     = mesh_.getInnerHalfEdgeAroundFaceCirculator(idx_he_boundary);
    const IHEAFC circ_iheaf_end = circ_iheaf;
    do
    {
      HalfEdgeIndex he = circ_iheaf.getTargetIndex();
      addToQueue(he);
    } while (++circ_iheaf != circ_iheaf_end);

    #ifdef AFRONTDEBUG
    viewer_ = pcl::visualization::PCLVisualizer::Ptr(new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer_->addCoordinateSystem(1.0);
    viewer_->initCameraParameters();
    viewer_->setBackgroundColor (0, 0, 0);
    viewer_->addPolygonMesh(getMesh());

    int v1 = 0;
    viewer_->createViewPort(0.0, 0.0, 0.5, 1.0, v1);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(input_cloud_, 0, 255, 0);
    viewer_->addPointCloud<pcl::PointXYZ> (input_cloud_, single_color, "sample cloud", v1);

    int v2 = 1;
    viewer_->createViewPort(0.5, 0.0, 1.0, 1.0, v2);
    viewer_->addPointCloud<pcl::PointNormal> (mls_cloud_, "mls_cloud", v2);

    viewer_->registerKeyboardCallback(&AfrontMeshing::keyboardEventOccurred, *this);
    viewer_->spin();
    #endif

    return true;
  }

  bool AfrontMeshing::computeGuidanceField()
  {
    std::printf("\x1B[36mComputing Guidance Field!\x1B[0m\n");
    auto start = std::chrono::high_resolution_clock::now();

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

    std::chrono::duration<double> elapsed = std::chrono::high_resolution_clock::now() - start;
    std::printf("\x1B[36mComputing Guidance Field Finished (%f sec)!\x1B[0m\n", elapsed.count());
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
    std::printf("\x1B[36mMeshing Started!\x1B[0m\n");
    auto start = std::chrono::high_resolution_clock::now();
    while (!finished_)
      stepMesh();

    std::chrono::duration<double> elapsed = std::chrono::high_resolution_clock::now() - start;
    std::printf("\x1B[36mMeshing Finished (%f sec)!\x1B[0m\n", elapsed.count());
  }

  void AfrontMeshing::stepMesh()
  {
    if (finished_)
    {
      std::printf("\x1B[31mTried to step mesh after it has finished meshing!\x1B[0m\n");
      return;
    }

    HalfEdgeIndex half_edge = queue_.front();
    queue_.pop_front();

    updateKdTree();

    FrontData front = getAdvancingFrontData(half_edge);
    CanCutEarResults ccer = canCutEar(front);

    #ifdef AFRONTDEBUG
    counter_ += 1;
    std::printf("\x1B[35mAdvancing Front: %lu\x1B[0m\n", counter_);

    // Remove previous fence from viewer
    for (auto j = 1; j <= fence_counter_; ++j)
      viewer_->removeShape("fence" + std::to_string(j));
    fence_counter_ = 0;

    // remove previouse iterations lines
    viewer_->removeShape("HalfEdge");
    viewer_->removeShape("NextHalfEdge");
    viewer_->removeShape("PrevHalfEdge");
    viewer_->removeShape("LeftSide");
    viewer_->removeShape("RightSide");
    viewer_->removeShape("Closest");


    pcl::PointXYZ p1, p2, p3, p4;
    p1 = utils::convertEigenToPCL(ccer.next.tri.p[0]);
    p2 = utils::convertEigenToPCL(ccer.next.tri.p[1]);
    p3 = utils::convertEigenToPCL(ccer.next.tri.p[2]);
    p4 = utils::convertEigenToPCL(ccer.prev.tri.p[2]);

    viewer_->addLine<pcl::PointXYZ, pcl::PointXYZ>(p1, p2, 0, 255, 0, "HalfEdge");       // Green
    viewer_->addLine<pcl::PointXYZ, pcl::PointXYZ>(p2, p3, 255, 0, 0, "NextHalfEdge");   // Red
    viewer_->addLine<pcl::PointXYZ, pcl::PointXYZ>(p1, p4, 255, 0, 255, "PrevHalfEdge"); // Magenta
    viewer_->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 8, "HalfEdge");
    viewer_->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 8, "NextHalfEdge");
    viewer_->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 8, "PrevHalfEdge");
    #endif

    if (ccer.type == CanCutEarResults::ClosedArea)
    {
      #ifdef AFRONTDEBUG
      std::printf("\x1B[34m  Performed Ear Cut Opperation\x1B[0m\n");
      #endif
      cutEar(ccer);
    }
    else
    {
      // If we can not cut ear then try and grow.
      PredictVertexResults pv = predictVertex(front);
      if (pv.at_boundary)
      {
        #ifdef AFRONTDEBUG
        std::printf("\x1B[32m  At Point Cloud Boundary\x1B[0m\n");
        #endif
        boundary_.push_back(front.he);
      }
      else
      {
        #ifdef AFRONTDEBUG
        p3 = utils::convertEigenToPCL(pv.tri.p[2]);
        viewer_->addLine<pcl::PointXYZ, pcl::PointXYZ>(p1, p3, 0, 255, 0, "RightSide");       // Green
        viewer_->addLine<pcl::PointXYZ, pcl::PointXYZ>(p2, p3, 0, 255, 0, "LeftSide");        // Green
        viewer_->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 8, "RightSide");
        viewer_->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 8, "LeftSide");
        #endif
        TriangleToCloseResults ttcr = isTriangleToClose(ccer, pv);
        if (!ttcr.found)
        {
          #ifdef AFRONTDEBUG
          std::printf("\x1B[32m  Performed Grow Opperation\x1B[0m\n");
          #endif
          grow(ccer, pv);
        }
        else
        {
          #ifdef AFRONTDEBUG
          std::printf("\x1B[33m  Performed Topology Event Opperation\x1B[0m\n");
          #endif
          merge(ttcr);
//          topologyEvent(ttcr);
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
    if (d > 0.5) // Need to make the max edge size a parameter.
      d = 0.5;
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

    mp = utils::getMidPoint(p1, p2);
    d = utils::distPoint2Point(p1, p2);

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
    result.type = CanCutEarResults::None;

    //////////////////////////
    // Check Next Half Edge //
    //////////////////////////
    result.next.primary = front.he;
    result.next.secondary = getNextHalfEdge(front.he);
    result.next.valid = false;
    result.next.vi[0] = front.vi[0];
    result.next.vi[1] = front.vi[1];
    result.next.vi[2] = mesh_.getTerminatingVertexIndex(result.next.secondary);
    p3 = (mesh_vertex_data_[result.next.vi[2].get()]).getVector3fMap();
    result.next.tri = getTriangleData(front, p3);

    // Check if triangle is valid
    if (result.next.tri.valid)
      if (result.next.tri.a < 1.22173 && result.next.tri.b < 1.22173 && result.next.tri.c < 1.22173)
        result.next.valid = true;

    //////////////////////////
    // Check Prev Half Edge //
    //////////////////////////
    result.prev.primary = front.he;
    result.prev.secondary = getPrevHalfEdge(front.he);
    result.prev.valid = false;
    result.prev.vi[0] = front.vi[0];
    result.prev.vi[1] = front.vi[1];
    result.prev.vi[2] = mesh_.getOriginatingVertexIndex(result.prev.secondary);
    p3 = (mesh_vertex_data_[result.prev.vi[2].get()]).getVector3fMap();
    result.prev.tri = getTriangleData(front, p3);

    // Check if triangle is valid
    if (result.prev.tri.valid)
      if (result.prev.tri.a < 1.22173 && result.prev.tri.b < 1.22173 && result.prev.tri.c < 1.22173)
        result.prev.valid = true;

    if (mesh_.getOppositeFaceIndex(result.next.secondary) == mesh_.getOppositeFaceIndex(result.prev.secondary)) // Check if front, previouse and next are associated to the same triangle
      result.type = CanCutEarResults::None;
    else if (result.next.vi[2] == result.prev.vi[2]) // Check if front, previouse and next create a closed area
      result.type = CanCutEarResults::ClosedArea;
    else // Review results and choose the best one
      if (result.next.valid && result.prev.valid)
        if (result.next.tri.aspect_ratio < result.prev.tri.aspect_ratio)
          result.type = CanCutEarResults::PrevHalfEdge;
        else
          result.type = CanCutEarResults::NextHalfEdge;
      else if (result.next.valid)
        result.type = CanCutEarResults::NextHalfEdge;
      else if (result.prev.valid)
        result.type = CanCutEarResults::PrevHalfEdge;

    return result;
  }

  void AfrontMeshing::cutEar(const CanCutEarResults &ccer)
  {
    assert(ccer.type != CanCutEarResults::None);

    if (ccer.type == CanCutEarResults::ClosedArea)
    {
      MeshTraits::FaceData new_fd = createFaceData((mesh_vertex_data_[ccer.prev.vi[0].get()]).getVector3fMap(),
                                                   (mesh_vertex_data_[ccer.prev.vi[1].get()]).getVector3fMap(),
                                                   (mesh_vertex_data_[ccer.prev.vi[2].get()]).getVector3fMap());

      mesh_.addFace(ccer.prev.vi[0], ccer.prev.vi[1], ccer.prev.vi[2], new_fd);
      removeFromQueue(ccer.prev.secondary, ccer.next.secondary);
      removeFromBoundary(ccer.prev.secondary, ccer.next.secondary);
    }
    else
    {
      // Add new face
      const CanCutEarResult *data;
      HalfEdgeIndex temp;
      if (ccer.type == CanCutEarResults::PrevHalfEdge)
      {
        data = &ccer.prev;
        temp = getPrevHalfEdge(data->secondary);
      }
      else
      {
        data = &ccer.next;
        temp = getPrevHalfEdge(data->primary);
      }

      MeshTraits::FaceData new_fd = createFaceData((mesh_vertex_data_[data->vi[0].get()]).getVector3fMap(),
                                                   (mesh_vertex_data_[data->vi[1].get()]).getVector3fMap(),
                                                   (mesh_vertex_data_[data->vi[2].get()]).getVector3fMap());


      mesh_.addFace(data->vi[0], data->vi[1], data->vi[2], new_fd);

      addToQueue(getNextHalfEdge(temp));
      removeFromQueue(data->secondary);
      removeFromBoundary(data->secondary);
    }
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
    result.length = utils::distPoint2Point(result.p[0], result.p[1]);

    // Get half edge midpoint
    result.mp = utils::getMidPoint(result.p[0], result.p[1]);

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
    Eigen::Vector3f closest = (mls_cloud_->at(result.pv.closest)).getVector3fMap();
    result.at_boundary = !isPointValid(front, closest);

    // Get triangle Data
    if (snap_)
    {
      result.tri = getTriangleData(front, mls_cloud_->points[result.pv.closest].getVector3fMap());
      if (result.at_boundary == false && result.tri.aspect_ratio < 0.1)
        result.at_boundary = true;
    }
    else
      result.tri = getTriangleData(front, result.pv.point.getVector3fMap());

    return result;
  }

  AfrontMeshing::CloseProximityResults AfrontMeshing::isCloseProximity(const CanCutEarResults &ccer, const PredictVertexResults &pvr) const
  {
    CloseProximityResults results;
//    std::vector<int> K;
//    std::vector<float> K_dist;

//    pcl::PointXYZ search_pt = utils::convertEigenToPCL(pvr.tri.p[2]);
//    mesh_tree_->radiusSearch(search_pt, 3.0 * pvr.gdr.l, K, K_dist);

//    results.fences.reserve(K.size());
//    results.verticies.reserve(K.size());
//    results.found = false;

//    // First check for closest proximity violation
//    for(auto i = 0; i < K.size(); ++i)
//    {
//      MeshTraits::VertexData &data = mesh_vertex_data_.at(K[i]);
//      VertexIndex vi = mesh_.getVertexIndex(data);

//      if (mesh_.isBoundary(vi))
//      {
//        HalfEdgeIndex he = mesh_.getOutgoingHalfEdgeIndex(vi);
//        if (mesh_.isBoundary(he) && (vi != pvr.front.vi[0]) && (vi != pvr.front.vi[1]))
//        {
//          assert(vi == mesh_.getOriginatingVertexIndex(he));
//          Eigen::Vector3f chkpt = mesh_vertex_data_[vi.get()].getVector3fMap();
//          Eigen::Vector3f endpt = mesh_vertex_data_[mesh_.getTerminatingVertexIndex(he).get()].getVector3fMap();
//          utils::DistPoint2LineResults left_side = utils::distPoint2Line(pvr.tri.p[0], pvr.tri.p[2], chkpt);
//          utils::DistPoint2LineResults right_side = utils::distPoint2Line(pvr.tri.p[1], pvr.tri.p[2], chkpt);

//          bool chkpt_valid = isPointValid(pvr.front, chkpt, false);
//          bool endpt_valid = isPointValid(pvr.front, endpt, false);

//          if (chkpt_valid || endpt_valid) // If either vertex is valid add as a valid fence check.
//          {
//            results.fences.push_back(he);

//            #ifdef AFRONTDEBUG
//            fence_counter_ += 1;
//            std::string fence_name = "fence" + std::to_string(fence_counter_);
//            viewer_->addLine<pcl::PointXYZ, pcl::PointXYZ>(utils::convertEigenToPCL(chkpt), utils::convertEigenToPCL(endpt), 0, 255, 255, fence_name) ; // pink
//            viewer_->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 8, fence_name);
//            #endif
//          }

//          if (!chkpt_valid)
//            continue;

//          results.verticies.push_back(vi);
//          utils::DistPoint2LineResults min_dist;
//          if (left_side.d < right_side.d)
//            min_dist = left_side;
//          else
//            min_dist = right_side;

//          if (min_dist.d < 0.5 * pvr.gdr.l) //pvr.gdr.l
//          {
//            if (!results.found)
//            {
//              results.found = true;
//              results.dist = min_dist;
//              results.closest = vi;
//            }
//            else
//            {
//              if (min_dist.d < results.dist.d)
//              {
//                results.dist = min_dist;
//                results.closest = vi;
//              }
//            }
//          }
//        }
//      }
//    }


    int lqueue = queue_.size();
    int lboundary = boundary_.size();
    int fronts = lqueue + lboundary;

    results.fences.reserve(fronts);
    results.verticies.reserve(fronts);
    results.found = false;

    for(auto i = 0; i < fronts; ++i)
    {
      HalfEdgeIndex he;
      if (i < lqueue)
        he = queue_[i];
      else
        he = boundary_[i-lqueue];

      VertexIndex vi[2];
      vi[0] = mesh_.getOriginatingVertexIndex(he);
      vi[1] = mesh_.getTerminatingVertexIndex(he);

      Eigen::Vector3f chkpt = mesh_vertex_data_[vi[0].get()].getVector3fMap();
      Eigen::Vector3f endpt = mesh_vertex_data_[vi[1].get()].getVector3fMap();

      bool chkpt_valid = isPointValid(pvr.front, chkpt, false) && (utils::distPoint2Point(pvr.tri.p[2], chkpt) < 3.0 * pvr.gdr.l);
      bool endpt_valid = isPointValid(pvr.front, endpt, false) && (utils::distPoint2Point(pvr.tri.p[2], endpt) < 3.0 * pvr.gdr.l);

      if (chkpt_valid || endpt_valid) // If either vertex is valid add as a valid fence check.
      {
        results.fences.push_back(he);

        #ifdef AFRONTDEBUG
        fence_counter_ += 1;
        std::string fence_name = "fence" + std::to_string(fence_counter_);
        viewer_->addLine<pcl::PointXYZ, pcl::PointXYZ>(utils::convertEigenToPCL(chkpt), utils::convertEigenToPCL(endpt), 0, 255, 255, fence_name) ; // pink
        viewer_->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 8, fence_name);
        #endif
      }

      if (!chkpt_valid)
        continue;

      utils::DistPoint2LineResults left_side = utils::distPoint2Line(pvr.tri.p[0], pvr.tri.p[2], chkpt);
      utils::DistPoint2LineResults right_side = utils::distPoint2Line(pvr.tri.p[1], pvr.tri.p[2], chkpt);

      results.verticies.push_back(vi[0]);
      utils::DistPoint2LineResults min_dist;
      if (left_side.d < right_side.d)
        min_dist = left_side;
      else
        min_dist = right_side;

      if (min_dist.d < 0.5 * pvr.gdr.l) //pvr.gdr.l
      {
        if (!results.found)
        {
          results.found = true;
          results.dist = min_dist;
          results.closest = vi[0];
        }
        else
        {
          if (min_dist.d < results.dist.d)
          {
            results.dist = min_dist;
            results.closest = vi[0];
          }
        }
      }
    }


    // TODO: Need to improve such that two triangles get created for this case (ccer.next.tri.c <= 1.5708)
    // TODO: Need to improve where a new triangle gets created while modify the other for this case (pvr.tri.c >= ccer.next.tri.c)
    if ((ccer.next.tri.valid && pvr.tri.c >= ccer.next.tri.c) && (ccer.prev.tri.valid && pvr.tri.b >= ccer.prev.tri.b))
    {
      if (ccer.next.tri.c < ccer.prev.tri.b)
      {
        results.closest = ccer.next.vi[2];
        results.dist = utils::distPoint2Line(pvr.tri.p[1], pvr.tri.p[2], ccer.next.tri.p[2]);
        results.found = true;
      }
      else
      {
        results.closest = ccer.prev.vi[2];
        results.dist = utils::distPoint2Line(pvr.tri.p[1], pvr.tri.p[2], ccer.prev.tri.p[2]);
        results.found = true;
      }
    }
    else if (ccer.next.tri.valid && pvr.tri.c >= ccer.next.tri.c)
    {
      results.closest = ccer.next.vi[2];
      results.dist = utils::distPoint2Line(pvr.tri.p[1], pvr.tri.p[2], ccer.next.tri.p[2]);
      results.found = true;
    }
    else if (ccer.prev.tri.valid && pvr.tri.b >= ccer.prev.tri.b)
    {
      results.closest = ccer.prev.vi[2];
      results.dist = utils::distPoint2Line(pvr.tri.p[1], pvr.tri.p[2], ccer.prev.tri.p[2]);
      results.found = true;
    }

//    if (!results.found || (results.found && results.closest != ccer.prev.vi[2] && results.closest != ccer.next.vi[2]))
//    {
//      if ((ccer.next.tri.valid && ccer.next.tri.c <= 1.5708) && (ccer.prev.tri.valid && ccer.prev.tri.b <= 1.5708))
//      {
//        if (ccer.next.tri.c < ccer.prev.tri.b)
//        {
//          results.closest = ccer.next.vi[2];
//          results.dist = utils::distPoint2Line(pvr.tri.p[1], pvr.tri.p[2], ccer.next.tri.p[2]);
//          results.found = true;
//        }
//        else
//        {
//          results.closest = ccer.prev.vi[2];
//          results.dist = utils::distPoint2Line(pvr.tri.p[1], pvr.tri.p[2], ccer.prev.tri.p[2]);
//          results.found = true;
//        }
//      }
//      else if (ccer.next.tri.valid && ccer.next.tri.c <= 1.5708)
//      {
//        results.closest = ccer.next.vi[2];
//        results.dist = utils::distPoint2Line(pvr.tri.p[1], pvr.tri.p[2], ccer.next.tri.p[2]);
//        results.found = true;
//      }
//      else if (ccer.prev.tri.valid && ccer.prev.tri.b <= 1.5708)
//      {
//        results.closest = ccer.prev.vi[2];
//        results.dist = utils::distPoint2Line(pvr.tri.p[1], pvr.tri.p[2], ccer.prev.tri.p[2]);
//        results.found = true;
//      }
//    }

    #ifdef AFRONTDEBUG
    Eigen::Vector3f p;
    p = pvr.tri.p[2];
    if (results.found)
      p = mesh_vertex_data_[results.closest.get()].getVector3fMap();

    viewer_->addSphere(utils::convertEigenToPCL(p), 0.2 * pvr.gdr.l, 255, 255, 0, "Closest");
    #endif

    return results;
  }

  AfrontMeshing::FenceViolationResults AfrontMeshing::isFenceViolated(const VertexIndex &vi, const Eigen::Vector3f &p, const CloseProximityResults &cpr, const PredictVertexResults &pvr) const
  {
    // Now need to check for fence violation
    FenceViolationResults results;
    results.found = false;
    Eigen::Vector3f sp = mesh_vertex_data_[vi.get()].getVector3fMap();

    for(auto i = 0; i < cpr.fences.size(); ++i)
    {
      // Need to ignore fence if either of its verticies are the same as the vi
      if (vi == mesh_.getOriginatingVertexIndex(cpr.fences[i]) || vi == mesh_.getTerminatingVertexIndex(cpr.fences[i]))
        continue;

      // Need to ignore fence check if closet is associated to the fence half edge
      if (cpr.found)
        if (cpr.closest == mesh_.getOriginatingVertexIndex(cpr.fences[i]) || cpr.closest == mesh_.getTerminatingVertexIndex(cpr.fences[i]))
          continue;

      // Check for fence intersection
      bool fence_violated = false;
      Eigen::Vector3f he_p1, he_p2;
      he_p1 = (mesh_vertex_data_[mesh_.getOriginatingVertexIndex(cpr.fences[i]).get()]).getVector3fMap();
      he_p2 = (mesh_vertex_data_[mesh_.getTerminatingVertexIndex(cpr.fences[i]).get()]).getVector3fMap();

      MeshTraits::FaceData fd = mesh_.getFaceDataCloud()[mesh_.getOppositeFaceIndex(cpr.fences[i]).get()];

      Eigen::Vector3f u = he_p2 - he_p1;
      Eigen::Vector3f v = fd.getNormalVector3fMap();
      v = v.normalized() * 3.0 * rho_; // Should the height of the fence be rho?

      utils::IntersectionLine2PlaneResults lpr;
      lpr = utils::intersectionLine2Plane(sp, p, he_p1, u, v);

      if (!lpr.parallel) // May need to add additional check if parallel
        if (lpr.mw <= 1 && lpr.mw >= 0)      // This checks if line segement intersects the plane
          if (lpr.mu <= 1 && lpr.mu >= 0)    // This checks if intersection point is within the x range of the plane
            if (lpr.mv <= 1 && lpr.mv >= -1) // This checks if intersection point is within the y range of the plane
              fence_violated = true;

      if (fence_violated)
      {
        double dist = (lpr.mw * lpr.w).dot(pvr.front.d);
        if (!results.found)
        {
          results.he = cpr.fences[i];
          results.index = i;
          results.lpr = lpr;
          results.dist = dist;
          results.found = true;
        }
        else
        {
          if (dist < results.dist)
          {
            results.he = cpr.fences[i];
            results.index = i;
            results.lpr = lpr;
            results.dist = dist;
          }
        }
      }
    }

    return results;
  }

  AfrontMeshing::TriangleToCloseResults AfrontMeshing::isTriangleToClose(const CanCutEarResults &ccer, const PredictVertexResults &pvr) const
  {
    TriangleToCloseResults results;
    results.ccer = ccer;
    results.pvr = pvr;

    // Check if new vertex is in close proximity to exising mesh verticies
    CloseProximityResults cpr = isCloseProximity(ccer, pvr);
    results.found = cpr.found;

    // Check if any fences are violated.
    FenceViolationResults fvr;
    if (cpr.found && cpr.closest == ccer.prev.vi[2])
    {
      results.closest = cpr.closest;
      fvr = isFenceViolated(ccer.prev.vi[1], ccer.prev.tri.p[2], cpr, pvr);
    }
    else if (cpr.found && cpr.closest == ccer.next.vi[2])
    {
      results.closest = cpr.closest;
      fvr = isFenceViolated(ccer.prev.vi[0], ccer.next.tri.p[2], cpr, pvr);
    }
    else
    {
      Eigen::Vector3f p = pvr.tri.p[2];
      if (cpr.found)
      {
        results.closest = cpr.closest;
        p = mesh_vertex_data_[cpr.closest.get()].getVector3fMap();
      }

      FenceViolationResults fvr1, fvr2, min_fvr;
      fvr1 = isFenceViolated(ccer.front.vi[0], p, cpr, pvr);
      fvr2 = isFenceViolated(ccer.front.vi[1], p, cpr, pvr);
      if (fvr1.found && fvr2.found)
      {
        fvr = fvr1;
        if (fvr2.dist < min_fvr.dist)
          fvr = fvr2;
      }
      else if (fvr1.found)
      {
        fvr = fvr1;
      }
      else
      {
        fvr = fvr2;
      }
    }

    if (fvr.found)
    {
      results.found = true;

      #ifdef AFRONTDEBUG
      assert(fence_counter_ != 0);
      std::string fence_name = "fence" + std::to_string(fvr.index + 1);
      viewer_->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 255, 140, 0, fence_name);
      #endif

      VertexIndex fvi[2];
      Eigen::Vector3f fp[2];
      fvi[0] = mesh_.getOriginatingVertexIndex(fvr.he);
      fvi[1] = mesh_.getTerminatingVertexIndex(fvr.he);
      fp[0] = (mesh_vertex_data_[fvi[0].get()]).getVector3fMap();
      fp[1] = (mesh_vertex_data_[fvi[1].get()]).getVector3fMap();

      bool vp1 = isPointValid(ccer.front, fp[0], false);
      bool vp2 = isPointValid(ccer.front, fp[1], false);
      assert(vp1 || vp2);
      if (vp1 && vp2)
        if (fvr.lpr.mu <= 0.5)
          results.closest = mesh_.getOriginatingVertexIndex(fvr.he);
        else
          results.closest = mesh_.getTerminatingVertexIndex(fvr.he);
      else if (vp1)
        results.closest = mesh_.getOriginatingVertexIndex(fvr.he);
      else
        results.closest = mesh_.getTerminatingVertexIndex(fvr.he);
    }

    return results;
  }

  utils::DistLine2LineResults AfrontMeshing::distLineToHalfEdge(const Eigen::Vector3f &p1, const Eigen::Vector3f &p2, const HalfEdgeIndex &half_edge) const
  {
    return utils::distLine2Line(p1, p2,
                                 (mesh_vertex_data_[mesh_.getOriginatingVertexIndex(half_edge).get()]).getVector3fMap(),
                                 (mesh_vertex_data_[mesh_.getTerminatingVertexIndex(half_edge).get()]).getVector3fMap());
  }

  bool AfrontMeshing::isPointValid(const FrontData &front, const Eigen::Vector3f p, bool limit) const
  {
    Eigen::Vector3f v1 = (front.p[1] - front.mp).normalized();
    Eigen::Vector3f v2 = p - front.mp;
    double dot1 = v2.dot(front.d);
    double dot2 = v2.dot(v1);
    if (limit)
    {
      if ((dot1 > 0.00001) && (std::abs(dot2) <= front.length/2.0))
        return true;
    }
    else
    {
      if ((dot1 > 0.00001))
        return true;
    }

    return false;
  }

  void AfrontMeshing::grow(const CanCutEarResults &ccer, const PredictVertexResults &pvr)
  {
    // Add new face
    MeshTraits::FaceData new_fd = createFaceData(pvr.tri.p[0], pvr.tri.p[1], pvr.tri.p[2]);
    pcl::PointXYZ np(pvr.tri.p[2](0), pvr.tri.p[2](1), pvr.tri.p[2](2));
    mesh_.addFace(pvr.front.vi[0], pvr.front.vi[1], mesh_.addVertex(np), new_fd);

    // Add new half edges to the queue
    addToQueue(getNextHalfEdge(ccer.prev.secondary));
    addToQueue(getPrevHalfEdge(ccer.next.secondary));
  }

  void AfrontMeshing::merge(const TriangleToCloseResults &ttcr)
  {
    assert(ttcr.closest != mesh_.getTerminatingVertexIndex(ttcr.ccer.prev.secondary));
    assert(ttcr.closest != mesh_.getOriginatingVertexIndex(ttcr.ccer.next.secondary));
    // Need to make sure at lease one vertex of the half edge is in the grow direction
    if (ttcr.closest == mesh_.getOriginatingVertexIndex(ttcr.ccer.prev.secondary))
    {
      #ifdef AFRONTDEBUG
      std::printf("\x1B[32m\tAborting Merge, Forced Ear Cut Opperation with Previous Half Edge!\x1B[0m\n");
      #endif
      CanCutEarResults force = ttcr.ccer;
      force.type = CanCutEarResults::PrevHalfEdge;
      cutEar(force);
      return;
    }
    else if (ttcr.closest == mesh_.getTerminatingVertexIndex(ttcr.ccer.next.secondary))
    {
      #ifdef AFRONTDEBUG
      std::printf("\x1B[32m\tAborting Merge, Forced Ear Cut Opperation with Next Half Edge!\x1B[0m\n");
      #endif
      CanCutEarResults force = ttcr.ccer;
      force.type = CanCutEarResults::NextHalfEdge;
      cutEar(force);
      return;
    }

    MeshTraits::FaceData new_fd = createFaceData(ttcr.pvr.tri.p[0], ttcr.pvr.tri.p[1], (mesh_vertex_data_[ttcr.closest.get()]).getVector3fMap());
    mesh_.addFace(ttcr.pvr.front.vi[0], ttcr.pvr.front.vi[1], ttcr.closest, new_fd);

    // Add new half edges to the queue
    addToQueue(getNextHalfEdge(ttcr.ccer.prev.secondary));
    addToQueue(getPrevHalfEdge(ttcr.ccer.next.secondary));
  }

  float AfrontMeshing::getCurvature(const int &index) const
  {
    if(index >= mls_cloud_->points.size())
      return -1.0;

    return mls_cloud_->at(index).curvature;
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
    int cnt = mls_cloud_tree_->radiusSearch(pn, 3 * reduction_ * max_length, gdr.k, gdr.k_dist);
    gdr.valid = false;
    gdr.max_curv = getMaxCurvature(gdr.k);
    if (cnt > 0)
    {
      gdr.valid = true;
      gdr.ideal = 2.0 * std::sin(rho_ / 2.0) / gdr.max_curv;
      double estimated = rho_ / gdr.max_curv;
      double max = min_length * reduction_;
      double min = max_length / reduction_;
//      if (estimated > gdr.ideal)
//        estimated = gdr.ideal;

//      if (estimated > edge_length)
//      {
//        double ratio = estimated / edge_length;
//        if (ratio > reduction_)
//          estimated = reduction_ * edge_length;

//      }
//      else
//      {
//        double ratio = edge_length / estimated;
//        if (ratio > reduction_)
//          estimated = edge_length / reduction_;
//      }
      if (min < max)
      {
        if (estimated < min)
          estimated = min;
        else if (estimated > max)
          estimated = max;
      }
      else
      {
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
      }

      gdr.estimated = estimated;
      gdr.l = std::sqrt(pow(estimated, 2.0) - pow(edge_length / 2.0, 2.0));
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
        double d = utils::distPoint2Point(p1.getVector3fMap(), p2.getVector3fMap());
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
        double d = utils::distPoint2Point(p1.getVector3fMap(), p2.getVector3fMap());
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
    double dot, sina, cosa, top, bottom, area;

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

    // Calculate the first angle of triangle
    dot = v1.dot(v3);
    cross = v1.cross(v3);
    bottom = result.A * result.C;
    top = cross.norm();
    sina = cross.norm()/bottom;
    cosa = dot/bottom;
    result.b = atan2(sina, cosa);

    area = 0.5 * top;
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

  void AfrontMeshing::addToQueue(const HalfEdgeIndex &half_edge)
  {
//    assert(std::find(queue_.begin(), queue_.end(), half_edge) == queue_.end());
    queue_.push_back(half_edge);
  }

  void AfrontMeshing::removeFromQueue(const HalfEdgeIndex &half_edge)
  {
    queue_.erase(std::remove_if(queue_.begin(), queue_.end(), [half_edge](HalfEdgeIndex he){ return (he == half_edge);}), queue_.end());
  }

  void AfrontMeshing::removeFromQueue(const HalfEdgeIndex &half_edge1, const HalfEdgeIndex &half_edge2)
  {
    queue_.erase(std::remove_if(queue_.begin(), queue_.end(), [half_edge1, half_edge2](HalfEdgeIndex he){ return (he == half_edge1 || he == half_edge2);}), queue_.end());
  }

  void AfrontMeshing::addToBoundary(const HalfEdgeIndex &half_edge)
  {
    assert(std::find(boundary_.begin(), boundary_.end(), half_edge) == boundary_.end());
    boundary_.push_back(half_edge);
  }

  void AfrontMeshing::removeFromBoundary(const HalfEdgeIndex &half_edge)
  {
    boundary_.erase(std::remove_if(boundary_.begin(), boundary_.end(), [half_edge](HalfEdgeIndex he){ return (he == half_edge);}), boundary_.end());
  }

  void AfrontMeshing::removeFromBoundary(const HalfEdgeIndex &half_edge1, const HalfEdgeIndex &half_edge2)
  {
    boundary_.erase(std::remove_if(boundary_.begin(), boundary_.end(), [half_edge1, half_edge2](HalfEdgeIndex he){ return (he == half_edge1 || he == half_edge2);}), boundary_.end());
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

#ifdef AFRONTDEBUG
  void AfrontMeshing::keyboardEventOccurred(const pcl::visualization::KeyboardEvent &event, void*)
  {
    if (event.getKeySym() == "n" && event.keyDown())
    {
      if (!isFinished())
      {
        stepMesh();

        viewer_->removePolygonMesh();
        viewer_->addPolygonMesh(getMesh());
      }
    }

    if (event.getKeySym() == "t" && event.keyDown())
    {
      if (!isFinished())
      {
        generateMesh();

        viewer_->removePolygonMesh();
        viewer_->addPolygonMesh(getMesh());
      }
    }

    if (event.getKeySym() == "m" && event.keyDown())
    {
      if (!isFinished())
      {
        for (auto i = 0; i < 25; ++i)
        {
          stepMesh();

          viewer_->removePolygonMesh();
          viewer_->addPolygonMesh(getMesh());
        }
      }
    }
  }
#endif
}

