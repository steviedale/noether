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
    threads_ = 1;
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
    viewer_->initCameraParameters();
    viewer_->setBackgroundColor (0, 0, 0);
    viewer_->addPolygonMesh(getMesh());

    int v1 = 1;
    viewer_->createViewPort(0.0, 0.5, 0.5, 1.0, v1);
    viewer_->addCoordinateSystem(1.0, v1);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(input_cloud_, 0, 255, 0);
    viewer_->addPointCloud<pcl::PointXYZ>(input_cloud_, single_color, "sample cloud", v1);

    //Show just mesh
    int v2 = 2;
    viewer_->createViewPort(0.5, 0.5, 1.0, 1.0, v2);
    viewer_->addCoordinateSystem(1.0, v2);

    //Show Final mesh results over the mls point cloud
    int v3 = 3;
    viewer_->createViewPort(0.0, 0.0, 0.5, 0.5, v3);
    viewer_->addCoordinateSystem(1.0, v3);
    pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointNormal> handler_k(mls_cloud_, "curvature");
    viewer_->addPointCloud<pcl::PointNormal>(mls_cloud_, handler_k, "mls_cloud", v3);

    //Show mls information
    int v4 = 4;
    viewer_->createViewPort(0.5, 0.0, 1.0, 0.5, v4);
    viewer_->addCoordinateSystem(1.0, v4);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointNormal> single_color2(mls_cloud_, 0, 255, 0);
    viewer_->addPointCloud<pcl::PointNormal>(mls_cloud_, single_color2, "mls_cloud2", v4);
    viewer_->addPointCloudNormals<pcl::PointNormal>(mls_cloud_, 1, 0.005, "mls_cloud_normals", v4);

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
    mls_.setNumberOfThreads(threads_);
    mls_.setComputeNormals(true);
    mls_.setInputCloud(input_cloud_);
    mls_.setPolynomialFit(true);
    mls_.setSearchMethod(input_cloud_tree_);
    mls_.setSearchRadius(r_);

    // Adding the Distinct cloud is to force the storing of the mls results.
    mls_.setUpsamplingMethod(pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointNormal>::DISTINCT_CLOUD);
    mls_.setDistinctCloud(input_cloud_);

    mls_.process(*mls_cloud_);
    if (mls_cloud_->empty())
      return false;

    mls_cloud_tree_ = pcl::search::KdTree<pcl::PointNormal>::Ptr(new pcl::search::KdTree<pcl::PointNormal>);
    mls_cloud_tree_->setSortedResults(true);
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

    AdvancingFrontData afront = getAdvancingFrontData(half_edge);

    #ifdef AFRONTDEBUG
    counter_ += 1;
    std::printf("\x1B[35mAdvancing Front: %lu\x1B[0m\n", counter_);

    // Remove previous fence from viewer
    for (auto j = 1; j <= fence_counter_; ++j)
      viewer_->removeShape("fence" + std::to_string(j), 2);
    fence_counter_ = 0;

    // remove previouse iterations objects
    viewer_->removeShape("HalfEdge");
    viewer_->removeShape("NextHalfEdge", 1);
    viewer_->removeShape("PrevHalfEdge", 1);
    viewer_->removeShape("LeftSide", 1);
    viewer_->removeShape("RightSide", 1);
    viewer_->removeShape("Closest", 1);
    viewer_->removeShape("ProxRadius", 1);
    viewer_->removePointCloud("Mesh_Vertex_Cloud", 2);
    viewer_->removeShape("MLSRadius", 4);
    viewer_->removeShape("MLSMean", 4);
    viewer_->removeShape("MLSProjection", 4);

    pcl::PointXYZ p1, p2, p3, p4;
    p1 = utils::convertEigenToPCL(afront.next.tri.p[0]);
    p2 = utils::convertEigenToPCL(afront.next.tri.p[1]);
    p3 = utils::convertEigenToPCL(afront.next.tri.p[2]);
    p4 = utils::convertEigenToPCL(afront.prev.tri.p[2]);

    viewer_->addLine<pcl::PointXYZ, pcl::PointXYZ>(p1, p2, 0, 255, 0, "HalfEdge");       // Green
    viewer_->addLine<pcl::PointXYZ, pcl::PointXYZ>(p2, p3, 255, 0, 0, "NextHalfEdge", 1);   // Red
    viewer_->addLine<pcl::PointXYZ, pcl::PointXYZ>(p1, p4, 255, 0, 255, "PrevHalfEdge", 1); // Magenta
    viewer_->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 8, "HalfEdge");
    viewer_->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 8, "NextHalfEdge", 1);
    viewer_->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 8, "PrevHalfEdge", 1);


    pcl::visualization::PointCloudColorHandlerCustom<MeshTraits::VertexData> single_color(mesh_vertex_data_copy_, 255, 128, 0);
    viewer_->addPointCloud<MeshTraits::VertexData>(mesh_vertex_data_copy_, single_color, "Mesh_Vertex_Cloud", 2);
    #endif

    if (mesh_.getOppositeFaceIndex(afront.next.secondary) != mesh_.getOppositeFaceIndex(afront.prev.secondary) && afront.next.vi[2] == afront.prev.vi[2]) // This indicates a closed area
    {
      #ifdef AFRONTDEBUG
      std::printf("\x1B[34m  Closing Area\x1B[0m\n");
      #endif
      MeshTraits::FaceData new_fd = createFaceData((mesh_vertex_data_[afront.prev.vi[0].get()]).getVector3fMap(),
                                                   (mesh_vertex_data_[afront.prev.vi[1].get()]).getVector3fMap(),
                                                   (mesh_vertex_data_[afront.prev.vi[2].get()]).getVector3fMap());

      mesh_.addFace(afront.prev.vi[0], afront.prev.vi[1], afront.prev.vi[2], new_fd);
      removeFromQueue(afront.prev.secondary, afront.next.secondary);
      removeFromBoundary(afront.prev.secondary, afront.next.secondary);
    }
    else
    {
      // If we can not cut ear then try and grow.
      PredictVertexResults pvr = predictVertex(afront);
      if (pvr.boundary)
      {
        #ifdef AFRONTDEBUG
        std::printf("\x1B[32m  At Point Cloud Boundary\x1B[0m\n");
        #endif
        boundary_.push_back(afront.front.he);
      }
      else if (!pvr.valid)
      {
        #ifdef AFRONTDEBUG
        std::printf("\x1B[32m  Unable to create valid triangle!\x1B[0m\n");
        #endif
        boundary_.push_back(afront.front.he);
      }
      else
      {
        #ifdef AFRONTDEBUG
        Eigen::Vector3f mean_pt = (mls_cloud_->at(pvr.pv.closest)).getVector3fMap();
        Eigen::Vector3f projected_pt = pvr.pv.point.getVector3fMap();
        viewer_->addSphere(utils::convertEigenToPCL(mean_pt), r_, 0, 255, 128, "MLSRadius", 4);
        viewer_->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 0.2, "MLSRadius", 4);

        viewer_->addSphere(utils::convertEigenToPCL(pvr.pv.mls.mean), 0.02 * r_, 255, 0, 0, "MLSMean", 4);
        viewer_->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 0.2, "MLSMean", 4);

        viewer_->addSphere(utils::convertEigenToPCL(projected_pt), 0.02 * r_, 255, 128, 0, "MLSProjection", 4);
        viewer_->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 0.2, "MLSProjection", 4);

        p3 = utils::convertEigenToPCL(pvr.tri.p[2]);
        viewer_->addLine<pcl::PointXYZ, pcl::PointXYZ>(p1, p3, 0, 255, 0, "RightSide", 1);       // Green
        viewer_->addLine<pcl::PointXYZ, pcl::PointXYZ>(p2, p3, 0, 255, 0, "LeftSide", 1);        // Green
        viewer_->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 8, "RightSide", 1);
        viewer_->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 8, "LeftSide", 1);
        #endif
        TriangleToCloseResults ttcr = isTriangleToClose(pvr);
        if (!ttcr.found)
        {
          #ifdef AFRONTDEBUG
          std::printf("\x1B[32m  Performed Grow Opperation\x1B[0m\n");
          #endif
          grow(pvr);
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
    MLSSampling::SamplePointResults sp1 = mls_.samplePoint(mls_cloud_->points[index]);
    Eigen::Vector3f p1 = sp1.point.getVector3fMap();

    // Get the allowed grow distance and control the first triangle size
    double max_step = getMaxStep(p1);

    // search for the nearest neighbor
    std::vector<int> K;
    std::vector<float> K_dist;
    mls_cloud_tree_->nearestKSearch(sp1.point, 2, K, K_dist);

    // use l1 and nearest neighbor to extend edge
    pcl::PointNormal dp;
    MLSSampling::SamplePointResults sp2, sp3;
    Eigen::Vector3f p2, p3, v1, v2, mp, norm;

    dp = mls_cloud_->points[K[1]];
    v1 = dp.getVector3fMap() - p1;
    v1 = v1.normalized();
    norm = dp.getNormalVector3fMap();

    sp2 = getPredictedVertex(p1, v1, max_step);
    p2 = sp2.point.getVector3fMap();

    mp = utils::getMidPoint(p1, p2);
    double d = utils::distPoint2Point(p1, p2);

    v2 = norm.cross(v1).normalized();

    GrowDistanceResults gdr = getGrowDistance(mp, d);
    sp3 = getPredictedVertex(mp, v2, gdr.l);
    p3 = sp3.point.getVector3fMap();

    std::cout << sp1.point.getNormalVector3fMap().transpose() << std::endl;
    std::cout << sp2.point.getNormalVector3fMap().transpose() << std::endl;
    std::cout << sp3.point.getNormalVector3fMap().transpose() << std::endl;

    MeshTraits::FaceData fd = createFaceData(p1, p2, p3);
    VertexIndices vi;
    vi.push_back(mesh_.addVertex(sp1.point));
    vi.push_back(mesh_.addVertex(sp2.point));
    vi.push_back(mesh_.addVertex(sp3.point));
    mesh_.addFace(vi[0], vi[1], vi[2], fd);
  }

  void AfrontMeshing::cutEar(const CutEarData &ccer)
  {
    assert(ccer.tri.valid);

    // Add new face
    HalfEdgeIndex temp;
    if (ccer.type == CutEarData::PrevHalfEdge)
      temp = getPrevHalfEdge(ccer.secondary);
    else
      temp = getPrevHalfEdge(ccer.primary);

    MeshTraits::FaceData new_fd = createFaceData((mesh_vertex_data_[ccer.vi[0].get()]).getVector3fMap(),
                                                 (mesh_vertex_data_[ccer.vi[1].get()]).getVector3fMap(),
                                                 (mesh_vertex_data_[ccer.vi[2].get()]).getVector3fMap());


    mesh_.addFace(ccer.vi[0], ccer.vi[1], ccer.vi[2], new_fd);

    addToQueue(getNextHalfEdge(temp));
    removeFromQueue(ccer.secondary);
    removeFromBoundary(ccer.secondary);
  }

  AfrontMeshing::AdvancingFrontData AfrontMeshing::getAdvancingFrontData(const HalfEdgeIndex &half_edge) const
  {
    AdvancingFrontData result;
    result.front.he = half_edge;

    FaceIndex face_indx = mesh_.getOppositeFaceIndex(half_edge);
    MeshTraits::FaceData fd = mesh_.getFaceDataCloud()[face_indx.get()];

    // Get Half Edge Vertexs
    result.front.vi[0] = mesh_.getOriginatingVertexIndex(half_edge);
    result.front.vi[1] = mesh_.getTerminatingVertexIndex(half_edge);
    result.front.p[0] = (mesh_vertex_data_[result.front.vi[0].get()]).getVector3fMap();
    result.front.p[1] = (mesh_vertex_data_[result.front.vi[1].get()]).getVector3fMap();

    // Calculate the half edge length
    result.front.length = utils::distPoint2Point(result.front.p[0], result.front.p[1]);

    // Get half edge midpoint
    result.front.mp = utils::getMidPoint(result.front.p[0], result.front.p[1]);

    // Calculate the grow direction vector
    result.front.d = getGrowDirection(result.front.p[0], result.front.mp, fd);

    Eigen::Vector3f p3;

    //////////////////////////
    // Check Next Half Edge //
    //////////////////////////
    result.next.type = CutEarData::NextHalfEdge;
    result.next.primary = result.front.he;
    result.next.secondary = getNextHalfEdge(result.front.he);
    result.next.valid = false;
    result.next.vi[0] = result.front.vi[0];
    result.next.vi[1] = result.front.vi[1];
    result.next.vi[2] = mesh_.getTerminatingVertexIndex(result.next.secondary);
    p3 = (mesh_vertex_data_[result.next.vi[2].get()]).getVector3fMap();
    result.next.tri = getTriangleData(result.front, p3);

    // Check if triangle is valid
    if (result.next.tri.valid)
      if (result.next.tri.a < 1.22173 && result.next.tri.b < 1.22173 && result.next.tri.c < 1.22173)
        result.next.valid = true;

    //////////////////////////
    // Check Prev Half Edge //
    //////////////////////////
    result.prev.type = CutEarData::PrevHalfEdge;
    result.prev.primary = result.front.he;
    result.prev.secondary = getPrevHalfEdge(result.front.he);
    result.prev.valid = false;
    result.prev.vi[0] = result.front.vi[0];
    result.prev.vi[1] = result.front.vi[1];
    result.prev.vi[2] = mesh_.getOriginatingVertexIndex(result.prev.secondary);
    p3 = (mesh_vertex_data_[result.prev.vi[2].get()]).getVector3fMap();
    result.prev.tri = getTriangleData(result.front, p3);

    // Check if triangle is valid
    if (result.prev.tri.valid)
      if (result.prev.tri.a < 1.22173 && result.prev.tri.b < 1.22173 && result.prev.tri.c < 1.22173)
        result.prev.valid = true;

    return result;
  }

  AfrontMeshing::PredictVertexResults AfrontMeshing::predictVertex(const AdvancingFrontData &afront) const
  {
    // Local Variables
    PredictVertexResults result;
    const FrontData &front = afront.front;

    result.afront = afront;

    // Get the allowed grow distance
    result.gdr = getGrowDistance(front.mp, front.length);

    // This is required because 2 * step size < front.length results in nan
    if (!std::isnan(result.gdr.l))
    {
      // Get predicted vertex
      result.pv = getPredictedVertex(front.mp, front.d, result.gdr.l);

      // Check and see if there any point in the grow direction of the front.
      // If not then it is at the boundary of the point cloud.
      Eigen::Vector3f closest = (mls_cloud_->at(result.pv.closest)).getVector3fMap();
      result.boundary = nearBoundary(front, closest);
      result.valid = !result.boundary;

      // Get triangle Data
      result.tri = getTriangleData(front, result.pv.point.getVector3fMap());
    }
    else
    {
      #ifdef AFRONTDEBUG
      std::printf("\x1B[32m  Unable to predict a valid vertex!\x1B[0m\n");
      #endif
      result.boundary = false;
      result.valid = false;
    }

    return result;
  }

  AfrontMeshing::CloseProximityResults AfrontMeshing::isCloseProximity(const PredictVertexResults &pvr) const
  {
    CloseProximityResults results;
    std::vector<int> K;
    std::vector<float> K_dist;

    mesh_tree_->radiusSearch(pvr.pv.point, 3.0 * pvr.gdr.estimated, K, K_dist);

    results.fences.reserve(K.size());
    results.verticies.reserve(K.size());
    results.found = false;

    // First check for closest proximity violation
    for(auto i = 0; i < K.size(); ++i)
    {
      MeshTraits::VertexData &data = mesh_vertex_data_.at(K[i]);
      VertexIndex vi = mesh_.getVertexIndex(data);

      // Don't include the front vertices
      if (vi == pvr.afront.front.vi[0] || vi == pvr.afront.front.vi[1])
        continue;

      if (mesh_.isBoundary(vi))
      {
        Eigen::Vector3f chkpt = mesh_vertex_data_[vi.get()].getVector3fMap();
        bool chkpt_valid = isPointValid(pvr.afront.front, chkpt);

        OHEAVC       circ     = mesh_.getOutgoingHalfEdgeAroundVertexCirculator(vi);
        const OHEAVC circ_end = circ;
        do
        {
          HalfEdgeIndex he = circ.getTargetIndex();
          VertexIndex evi = mesh_.getTerminatingVertexIndex(he);
          if (!mesh_.isBoundary(he))
          {
            he = mesh_.getOppositeHalfEdgeIndex(he);
            if (!mesh_.isBoundary(he))
              continue;
          }

          // Don't include the previouse or next half edge
          if (he == pvr.afront.next.secondary || he == pvr.afront.prev.secondary)
            continue;

          Eigen::Vector3f endpt = mesh_vertex_data_[evi.get()].getVector3fMap();
          bool endpt_valid = isPointValid(pvr.afront.front, endpt);
          if (chkpt_valid || endpt_valid) // If either vertex is valid add as a valid fence check.
          {
            if (std::find(results.fences.begin(), results.fences.end(), he) == results.fences.end())
            {
              results.fences.push_back(he);

              #ifdef AFRONTDEBUG
              fence_counter_ += 1;
              std::string fence_name = "fence" + std::to_string(fence_counter_);
              viewer_->addLine<pcl::PointXYZ, pcl::PointXYZ>(utils::convertEigenToPCL(chkpt), utils::convertEigenToPCL(endpt), 0, 255, 255, fence_name, 2) ; // orange
              viewer_->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 8, fence_name, 2);
              #endif
            }
          }

        } while (++circ != circ_end);

        if (!chkpt_valid)
          continue;

        double dist = utils::distPoint2Point(pvr.tri.p[2], chkpt);
        results.verticies.push_back(vi);
        if (dist < 0.5 * pvr.gdr.estimated)
        {
          if (!results.found)
          {
            results.found = true;
            results.dist = dist;
            results.closest = vi;
          }
          else
          {
            if (dist < results.dist)
            {
              results.dist = dist;
              results.closest = vi;
            }
          }
        }
      }
    }

//    int lqueue = queue_.size();
//    int lboundary = boundary_.size();
//    int fronts = lqueue + lboundary;

//    results.fences.reserve(fronts);
//    results.verticies.reserve(fronts);
//    results.found = false;

//    for(auto i = 0; i < fronts; ++i)
//    {
//      HalfEdgeIndex he;
//      if (i < lqueue)
//        he = queue_[i];
//      else
//        he = boundary_[i-lqueue];

//      VertexIndex vi[2];
//      vi[0] = mesh_.getOriginatingVertexIndex(he);
//      vi[1] = mesh_.getTerminatingVertexIndex(he);

//      Eigen::Vector3f chkpt = mesh_vertex_data_[vi[0].get()].getVector3fMap();
//      Eigen::Vector3f endpt = mesh_vertex_data_[vi[1].get()].getVector3fMap();

//      bool chkpt_valid = isPointValid(pvr.afront.front, chkpt, false) && (utils::distPoint2Point(pvr.tri.p[2], chkpt) < 3.0 * pvr.gdr.estimated);
//      bool endpt_valid = isPointValid(pvr.afront.front, endpt, false) && (utils::distPoint2Point(pvr.tri.p[2], endpt) < 3.0 * pvr.gdr.estimated);

//      if (chkpt_valid || endpt_valid) // If either vertex is valid add as a valid fence check.
//      {
//        results.fences.push_back(he);

//        #ifdef AFRONTDEBUG
//        fence_counter_ += 1;
//        std::string fence_name = "fence" + std::to_string(fence_counter_);
//        viewer_->addLine<pcl::PointXYZ, pcl::PointXYZ>(utils::convertEigenToPCL(chkpt), utils::convertEigenToPCL(endpt), 0, 255, 255, fence_name, 2) ; // orange
//        viewer_->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 8, fence_name, 2);
//        #endif
//      }

//      if (!chkpt_valid)
//        continue;

//      double dist = utils::distPoint2Point(pvr.tri.p[2], chkpt);
//      results.verticies.push_back(vi[0]);
//      if (dist < 0.5 * pvr.gdr.estimated)
//      {
//        if (!results.found)
//        {
//          results.found = true;
//          results.dist = dist;
//          results.closest = vi[0];
//        }
//        else
//        {
//          if (dist < results.dist)
//          {
//            results.dist = dist;
//            results.closest = vi[0];
//          }
//        }
//      }
//    }

    // If either the prev or next half edge vertice is with in tolerance default to it
    // over closest distant point.
    double prev_dist = utils::distPoint2Point(pvr.tri.p[2], pvr.afront.prev.tri.p[2]);
    double next_dist = utils::distPoint2Point(pvr.tri.p[2], pvr.afront.next.tri.p[2]);
    if (pvr.afront.prev.tri.valid && pvr.afront.next.tri.valid)
    {
      if (prev_dist < next_dist)
      {
        if (prev_dist < 0.5 * pvr.gdr.estimated)
        {
          results.found = true;
          results.dist = prev_dist;
          results.closest = pvr.afront.prev.vi[2];
        }
      }
      else
      {
        if (next_dist < 0.5 * pvr.gdr.estimated)
        {
          results.found = true;
          results.dist = next_dist;
          results.closest = pvr.afront.next.vi[2];
        }
      }
    }
    else if (pvr.afront.prev.tri.valid && prev_dist < 0.5 * pvr.gdr.estimated)
    {
      results.found = true;
      results.dist = prev_dist;
      results.closest = pvr.afront.prev.vi[2];
    }
    else if (pvr.afront.next.tri.valid && next_dist < 0.5 * pvr.gdr.estimated)
    {
      results.found = true;
      results.dist = next_dist;
      results.closest = pvr.afront.next.vi[2];
    }

    // If Nothing found check and make sure new vertex is not close to fence
    if (!results.found)
    {
      for(auto i = 0; i < results.fences.size(); ++i)
      {
        HalfEdgeIndex he = results.fences[i];

        VertexIndex vi[2];
        vi[0] = mesh_.getOriginatingVertexIndex(he);
        vi[1] = mesh_.getTerminatingVertexIndex(he);

        // It is not suffecient ot just compare half edge indexs because non manifold mesh is allowed.
        // Must check vertex index
        // if (he == pvr.afront.prev.secondary || he == pvr.afront.next.secondary)
        if (vi[0] == pvr.afront.front.vi[0] || vi[1] == pvr.afront.front.vi[0] || vi[0] == pvr.afront.front.vi[1] || vi[1] == pvr.afront.front.vi[1])
          continue;

        Eigen::Vector3f p1 = mesh_vertex_data_[vi[0].get()].getVector3fMap();
        Eigen::Vector3f p2 = mesh_vertex_data_[vi[1].get()].getVector3fMap();

        utils::DistPoint2LineResults dist = utils::distPoint2Line(p1, p2, pvr.tri.p[2]);
        if (dist.d < 0.5 * pvr.gdr.estimated)
        {

          bool check_p1 = isPointValid(pvr.afront.front, p1);
          bool check_p2 = isPointValid(pvr.afront.front, p2);
          int index;
          if (check_p1 && check_p2)
          {
            if (utils::distPoint2Point(p1, pvr.tri.p[2]) < utils::distPoint2Point(p2, pvr.tri.p[2]))
              index = 0;
            else
              index = 1;
          }
          else if (check_p1)
          {
            index = 0;
          }
          else
          {
            index = 1;
          }

          if (results.found && dist.d < results.dist)
          {
            results.found = true;
            results.dist = dist.d;
            results.closest = vi[index];
          }
          else
          {
            results.found = true;
            results.dist = dist.d;
            results.closest = vi[index];
          }
        }
      }
    }

    #ifdef AFRONTDEBUG
    Eigen::Vector3f p;
    p = pvr.tri.p[2];
    viewer_->addSphere(utils::convertEigenToPCL(p), 0.5 * pvr.gdr.estimated, 0, 255, 128, "ProxRadius", 1);
    viewer_->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 0.2, "ProxRadius", 1);
    #endif

    return results;
  }

  bool AfrontMeshing::checkPrevNextHalfEdge(const AdvancingFrontData &afront, TriangleData &tri, VertexIndex &vi) const
  {
    if (mesh_.isValid(vi) && vi == afront.prev.vi[2])
      if (afront.next.tri.valid && tri.c >= afront.next.tri.c)
      {
        vi = afront.next.vi[2];
        tri = afront.next.tri;
        return true;
      }

    if (mesh_.isValid(vi) && vi == afront.next.vi[2])
      if (afront.prev.tri.valid && tri.b >= afront.prev.tri.b)
      {
        vi = afront.prev.vi[2];
        tri = afront.prev.tri;
        return true;
      }

    if ((afront.next.tri.valid && tri.c >= afront.next.tri.c) && (afront.prev.tri.valid && tri.b >= afront.prev.tri.b))
    {
      if (afront.next.tri.c < afront.prev.tri.b)
      {
        vi = afront.next.vi[2];
        tri = afront.next.tri;
        return true;
      }
      else
      {
        vi = afront.prev.vi[2];
        tri = afront.prev.tri;
        return true;
      }
    }
    else if (afront.next.tri.valid && tri.c >= afront.next.tri.c)
    {
      vi = afront.next.vi[2];
      tri = afront.next.tri;
      return true;
    }
    else if (afront.prev.tri.valid && tri.b >= afront.prev.tri.b)
    {
      vi = afront.prev.vi[2];
      tri = afront.prev.tri;
      return true;
    }

    return false;
  }

  AfrontMeshing::FenceViolationResults AfrontMeshing::isFenceViolated(const VertexIndex &vi, const Eigen::Vector3f &p, const std::vector<HalfEdgeIndex> &fences, const VertexIndex &closest, const PredictVertexResults &pvr) const
  {
    // Now need to check for fence violation
    FenceViolationResults results;
    results.found = false;
    Eigen::Vector3f sp = mesh_vertex_data_[vi.get()].getVector3fMap();

    for(auto i = 0; i < fences.size(); ++i)
    {
      // Need to ignore fence if either of its verticies are the same as the vi
      if (vi == mesh_.getOriginatingVertexIndex(fences[i]) || vi == mesh_.getTerminatingVertexIndex(fences[i]))
        continue;

      // Need to ignore fence check if clocprset is associated to the fence half edge
      if (mesh_.isValid(closest))
        if (closest == mesh_.getOriginatingVertexIndex(fences[i]) || closest == mesh_.getTerminatingVertexIndex(fences[i]))
          continue;

      // Check for fence intersection
      bool fence_violated = false;
      Eigen::Vector3f he_p1, he_p2;
      he_p1 = (mesh_vertex_data_[mesh_.getOriginatingVertexIndex(fences[i]).get()]).getVector3fMap();
      he_p2 = (mesh_vertex_data_[mesh_.getTerminatingVertexIndex(fences[i]).get()]).getVector3fMap();

      MeshTraits::FaceData fd = mesh_.getFaceDataCloud()[mesh_.getOppositeFaceIndex(fences[i]).get()];

      Eigen::Vector3f u = he_p2 - he_p1;
      Eigen::Vector3f v = fd.getNormalVector3fMap();
      v = v.normalized() * rho_; // Should the height of the fence be rho?

      utils::IntersectionLine2PlaneResults lpr;
      lpr = utils::intersectionLine2Plane(sp, p, he_p1, u, v);

      if (!lpr.parallel) // May need to add additional check if parallel
        if (lpr.mw <= 1 && lpr.mw >= 0)      // This checks if line segement intersects the plane
          if (lpr.mu <= 1 && lpr.mu >= 0)    // This checks if intersection point is within the x range of the plane
            if (lpr.mv <= 1 && lpr.mv >= -1) // This checks if intersection point is within the y range of the plane
              fence_violated = true;

      if (fence_violated)
      {
        double dist = (lpr.mw * lpr.w).dot(pvr.afront.front.d);
        if (!results.found)
        {
          results.he = fences[i];
          results.index = i;
          results.lpr = lpr;
          results.dist = dist;
          results.found = true;
        }
        else
        {
          if (dist < results.dist)
          {
            results.he = fences[i];
            results.index = i;
            results.lpr = lpr;
            results.dist = dist;
          }
        }
      }
    }

    return results;
  }

  AfrontMeshing::TriangleToCloseResults AfrontMeshing::isTriangleToClose(const PredictVertexResults &pvr) const
  {
    TriangleToCloseResults results;
    results.pvr = pvr;

    // Check if new vertex is in close proximity to exising mesh verticies
    CloseProximityResults cpr = isCloseProximity(pvr);

    results.found = cpr.found;
    results.tri = pvr.tri;
    if (results.found)
    {
      results.closest = cpr.closest;
      results.tri = getTriangleData(pvr.afront.front, mesh_vertex_data_[results.closest.get()].getVector3fMap());
    }

    // Check if the proposed triangle interfers with the prev or next half edge
    if (checkPrevNextHalfEdge(pvr.afront, results.tri, results.closest))
      results.found = true;

    // Check if any fences are violated.
    FenceViolationResults fvr;
    for (int i = 0; i < cpr.fences.size(); ++i)
    {
      if (results.found && results.closest == pvr.afront.prev.vi[2])
      {
        fvr = isFenceViolated(pvr.afront.prev.vi[1], pvr.afront.prev.tri.p[2], cpr.fences, results.closest, pvr);
      }
      else if (results.found && results.closest == pvr.afront.next.vi[2])
      {
        fvr = isFenceViolated(pvr.afront.prev.vi[0], pvr.afront.next.tri.p[2], cpr.fences, results.closest, pvr);
      }
      else
      {
        FenceViolationResults fvr1, fvr2, min_fvr;
        fvr1 = isFenceViolated(pvr.afront.front.vi[0], results.tri.p[2], cpr.fences, results.closest, pvr);
        fvr2 = isFenceViolated(pvr.afront.front.vi[1], results.tri.p[2], cpr.fences, results.closest, pvr);
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
        viewer_->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 255, 140, 0, fence_name, 1);
        #endif

        VertexIndex fvi[2];
        Eigen::Vector3f fp[2];
        fvi[0] = mesh_.getOriginatingVertexIndex(fvr.he);
        fvi[1] = mesh_.getTerminatingVertexIndex(fvr.he);
        fp[0] = (mesh_vertex_data_[fvi[0].get()]).getVector3fMap();
        fp[1] = (mesh_vertex_data_[fvi[1].get()]).getVector3fMap();

        bool vp1 = isPointValid(pvr.afront.front, fp[0]);
        bool vp2 = isPointValid(pvr.afront.front, fp[1]);
        assert(vp1 || vp2);
        int index;
        if (vp1 && vp2)
        {
          double dist1 = utils::distPoint2Line(pvr.afront.front.p[0], pvr.afront.front.p[1], fp[0]).d;
          double dist2 = utils::distPoint2Line(pvr.afront.front.p[0], pvr.afront.front.p[1], fp[1]).d;
          if (dist1 < dist2)
            index = 0;
          else
            index = 1;
        }
        else if (vp1)
          index = 0;
        else
          index = 1;

        results.closest = fvi[index];
        results.tri = getTriangleData(pvr.afront.front, fp[index]);

        checkPrevNextHalfEdge(pvr.afront, results.tri, results.closest);
      }
      else
      {
        #ifdef AFRONTDEBUG
        Eigen::Vector3f p = pvr.tri.p[2];
        if (results.found)
          p = results.tri.p[2];

        viewer_->addSphere(utils::convertEigenToPCL(p), 0.1 * pvr.gdr.l, 255, 255, 0, "Closest", 1);
        #endif
        return results;
      }
    }

    // Need to print a warning message it should never get here.
    return results;
  }

  utils::DistLine2LineResults AfrontMeshing::distLineToHalfEdge(const Eigen::Vector3f &p1, const Eigen::Vector3f &p2, const HalfEdgeIndex &half_edge) const
  {
    return utils::distLine2Line(p1, p2,
                                 (mesh_vertex_data_[mesh_.getOriginatingVertexIndex(half_edge).get()]).getVector3fMap(),
                                 (mesh_vertex_data_[mesh_.getTerminatingVertexIndex(half_edge).get()]).getVector3fMap());
  }

  bool AfrontMeshing::isPointValid(const FrontData &front, const Eigen::Vector3f p) const
  {
    Eigen::Vector3f v = p - front.mp;
    double dot = v.dot(front.d);

    if ((dot > 0.0))
      return true;

    return false;
  }

  bool AfrontMeshing::nearBoundary(const FrontData &front, const Eigen::Vector3f p) const
  {
    Eigen::Vector3f v1 = (front.p[1] - front.mp).normalized();
    Eigen::Vector3f v2 = p - front.mp;
    double dot1 = v2.dot(front.d);
    double dot2 = v2.dot(v1);

    if ((dot1 > 0.0) && (std::abs(dot2) <= front.length/2.0))
      return false;

    return true;
  }

  void AfrontMeshing::grow(const PredictVertexResults &pvr)
  {
    // Add new face
    MeshTraits::FaceData new_fd = createFaceData(pvr.tri.p[0], pvr.tri.p[1], pvr.tri.p[2]);
    mesh_.addFace(pvr.afront.front.vi[0], pvr.afront.front.vi[1], mesh_.addVertex(pvr.pv.point), new_fd);

    // Add new half edges to the queue
    addToQueue(getNextHalfEdge(pvr.afront.prev.secondary));
    addToQueue(getPrevHalfEdge(pvr.afront.next.secondary));
  }

  void AfrontMeshing::merge(const TriangleToCloseResults &ttcr)
  {
    assert(ttcr.closest != mesh_.getTerminatingVertexIndex(ttcr.pvr.afront.prev.secondary));
    assert(ttcr.closest != mesh_.getOriginatingVertexIndex(ttcr.pvr.afront.next.secondary));
    // Need to make sure at lease one vertex of the half edge is in the grow direction
    if (ttcr.closest == mesh_.getOriginatingVertexIndex(ttcr.pvr.afront.prev.secondary))
    {
      #ifdef AFRONTDEBUG
      std::printf("\x1B[32m\tAborting Merge, Forced Ear Cut Opperation with Previous Half Edge!\x1B[0m\n");
      #endif
      cutEar(ttcr.pvr.afront.prev);
      return;
    }
    else if (ttcr.closest == mesh_.getTerminatingVertexIndex(ttcr.pvr.afront.next.secondary))
    {
      #ifdef AFRONTDEBUG
      std::printf("\x1B[32m\tAborting Merge, Forced Ear Cut Opperation with Next Half Edge!\x1B[0m\n");
      #endif
      cutEar(ttcr.pvr.afront.next);
      return;
    }

    MeshTraits::FaceData new_fd = createFaceData(ttcr.pvr.tri.p[0], ttcr.pvr.tri.p[1], (mesh_vertex_data_[ttcr.closest.get()]).getVector3fMap());
    mesh_.addFace(ttcr.pvr.afront.front.vi[0], ttcr.pvr.afront.front.vi[1], ttcr.closest, new_fd);

    // Add new half edges to the queue
    addToQueue(getNextHalfEdge(ttcr.pvr.afront.prev.secondary));
    addToQueue(getPrevHalfEdge(ttcr.pvr.afront.next.secondary));
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

  AfrontMeshing::GrowDistanceResults AfrontMeshing::getGrowDistance(const Eigen::Vector3f &mp, const double &edge_length) const
  {
    GrowDistanceResults gdr;
    gdr.estimated = getMaxStep(mp);
    gdr.l = std::sqrt(pow(gdr.estimated, 2.0) - pow(edge_length / 2.0, 2.0));

    return gdr;
  }

  double AfrontMeshing::getMaxStep(const Eigen::Vector3f &p) const
  {
    pcl::PointNormal pn;
    std::vector<int> k;
    std::vector<float> k_dist;
    pn.x = p(0);
    pn.y = p(1);
    pn.z = p(2);

    // What is shown in the afront paper. Need to figure out how to transverse the kdtree because
    // this could result in not enought point to meet the criteria.
    double len = std::numeric_limits<double>::max();
    double radius = 0;
    int j = 0;
    int pcnt = 0;
    bool finished = false;
    while (pcnt < (mls_cloud_->points.size() - 1))
    {
      int cnt = mls_cloud_tree_->radiusSearch(pn, (j + 1) * r_, k, k_dist);
      for(int i = pcnt; i < cnt; ++i)
      {
        float curvature = getCurvature(k[i]);
        double ideal = 2.0 * std::sin(rho_ / 2.0) / curvature;

        radius = sqrt(k_dist[i]);

        double step_required = (1.0 - reduction_) * radius + reduction_ * ideal;
        len = std::min(len, step_required);

        if (radius >= (len/ (1.0 - reduction_)))
        {
          finished = true;
          break;
        }
      }

      if (finished)
        break;

      pcnt = cnt - 1;
      ++j;
    }

    if (!finished)
      std::printf("\x1B[32m  Warning Max Step Not Found!\x1B[0m\n");

    return len;
  }

  MLSSampling::SamplePointResults AfrontMeshing::getPredictedVertex(const Eigen::Vector3f &mp, const Eigen::Vector3f &d, const double &l) const
  {
    Eigen::Vector3f p = mp + l * d;

    // Project new point onto the mls surface
    return mls_.samplePoint(pcl::PointXYZ(p(0), p(1), p(2)));
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
    assert(std::find(queue_.begin(), queue_.end(), half_edge) == queue_.end());
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
    if (event.getKeySym() == "i" && event.keyDown())
    {
      pause_ = true;
      std::printf("\x1B[36mInterupted!\x1B[0m\n");
      return;
    }

    if (event.getKeySym() == "n" && event.keyDown())
    {
      if (!isFinished())
      {
        stepMesh();

        viewer_->removePolygonMesh();
        viewer_->addPolygonMesh(getMesh());
      }
      return;
    }

    if (event.getKeySym() == "t" && event.keyDown())
    {
      pause_ = false;
      while (!isFinished() && !pause_)
      {
        stepMesh();

        viewer_->spinOnce(1, true);
        viewer_->removePolygonMesh();
        viewer_->addPolygonMesh(getMesh());
      }
      return;
    }

    if (event.getKeySym() == "m" && event.keyDown())
    {
      if (!isFinished())
      {
        for (auto i = 0; i < 1000; ++i)
        {
          stepMesh();
          viewer_->removePolygonMesh();
          viewer_->addPolygonMesh(getMesh());
        }
      }
      return;
    }
  }
#endif
}

