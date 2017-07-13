#include <pcl/geometry/mesh_conversion.h>
#include <pcl/io/pcd_io.h>
#include <pcl/surface/organized_fast_mesh.h>

#include <eigen3/Eigen/LU>

#include <meshing/afront_meshing.h>
#include <chrono>


namespace afront_meshing
{
  AfrontMeshing::AfrontMeshing() : mesh_vertex_data_(mesh_.getVertexDataCloud())
  {
    #ifdef AFRONTDEBUG
    counter_ = 0;
    fence_counter_ = 0;
    #endif

    setRho(0.5);
    setReduction(0.8);
    finished_ = false;
    threads_ = 1;

    mls_order_ = 2;
    int nr_coeff = (mls_order_ + 1) * (mls_order_ + 2) / 2;
    required_neighbors_ = 5 * nr_coeff;

    boundary_angle_threshold_ = M_PI_2;
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

    #ifdef AFRONTDEBUG
    viewer_ = pcl::visualization::PCLVisualizer::Ptr(new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer_->initCameraParameters();
    viewer_->setBackgroundColor (0, 0, 0);
    viewer_->addPolygonMesh(getMesh());

    int v1 = 1;
    viewer_->createViewPort(0.0, 0.5, 0.5, 1.0, v1);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(input_cloud_, 0, 255, 0);
    viewer_->addPointCloud<pcl::PointXYZ>(input_cloud_, single_color, "sample cloud", v1);

    //Show just mesh
    int v2 = 2;
    viewer_->createViewPort(0.5, 0.5, 1.0, 1.0, v2);

    //Show Final mesh results over the mls point cloud
    int v3 = 3;
    viewer_->createViewPort(0.0, 0.0, 0.5, 0.5, v3);
    pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointNormal> handler_k(mls_cloud_, "curvature");
    viewer_->addPointCloud<pcl::PointNormal>(mls_cloud_, handler_k, "mls_cloud", v3);

    //Show mls information
    int v4 = 4;
    viewer_->createViewPort(0.5, 0.0, 1.0, 0.5, v4);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointNormal> single_color2(mls_cloud_, 0, 255, 0);
    viewer_->addPointCloud<pcl::PointNormal>(mls_cloud_, single_color2, "mls_cloud2", v4);

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
    mls_.setPolynomialOrder(mls_order_);
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

  pcl::PointCloud<pcl::Normal>::ConstPtr AfrontMeshing::getMeshVertexNormals() const
  {
    pcl::PointCloud<pcl::Normal>::Ptr pn(new pcl::PointCloud<pcl::Normal>());;
    pcl::copyPointCloud(mesh_vertex_data_, *pn);

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
    viewer_->removePointCloud("Mesh_Vertex_Cloud_Normals", 4);
    viewer_->removeShape("MLSSurface", 4);
    viewer_->removeShape("MLSClosest", 4);
    viewer_->removeShape("MLSRadius", 4);
    viewer_->removeShape("MLSXAxis", 4);
    viewer_->removeShape("MLSYAxis", 4);
    viewer_->removeShape("MLSZAxis", 4);
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

    viewer_->addPointCloudNormals<pcl::PointXYZINormal>(mesh_vertex_data_copy_, 1, 0.005, "Mesh_Vertex_Cloud_Normals", 4);
    #endif

    if (mesh_.getOppositeFaceIndex(afront.next.secondary) != mesh_.getOppositeFaceIndex(afront.prev.secondary) && afront.next.vi[2] == afront.prev.vi[2]) // This indicates a closed area
    {
      #ifdef AFRONTDEBUG
      std::printf("\x1B[34m  Closing Area\x1B[0m\n");
      #endif
      MeshTraits::FaceData new_fd = createFaceData(afront.next.tri);
      // TODO: Need to check face normal to vertex normal. If not consistant to merge triangles together.
      //       This function has not been written yet. But it should look at the surrounding triangle and
      //       determine how to merge them.

      mesh_.addFace(afront.prev.vi[0], afront.prev.vi[1], afront.prev.vi[2], new_fd);
      removeFromQueue(afront.prev.secondary, afront.next.secondary);
      removeFromBoundary(afront.prev.secondary, afront.next.secondary);
    }
    else
    {
      // If we can not cut ear then try and grow.
      PredictVertexResults pvr = predictVertex(afront);

      #ifdef AFRONTDEBUG
      if (pvr.status == PredictVertexResults::Valid)
      {
        pcl::PolygonMesh mls_surface = getPolynomialSurface(pvr, r_/50);
        viewer_->addPolygonMesh(mls_surface, "MLSSurface", 4);
      }

      Eigen::Vector3f cpt = (mls_cloud_->at(pvr.pv.closest)).getVector3fMap();
      viewer_->addSphere(utils::convertEigenToPCL(cpt), 0.1 * r_, 255, 0, 0, "MLSClosest", 4);
      viewer_->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 0.2, "MLSClosest", 4);

      Eigen::Vector3f projected_pt = pvr.pv.point.getVector3fMap();
      viewer_->addSphere(utils::convertEigenToPCL(cpt), r_, 0, 255, 128, "MLSRadius", 4);
      viewer_->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 0.2, "MLSRadius", 4);

      pcl::PointXYZ mls_mean = utils::convertEigenToPCL(pvr.pv.mls.mean);
      Eigen::Vector3d mls_xaxis = pvr.pv.mls.mean + 0.1 * r_ * pvr.pv.mls.u_axis;
      Eigen::Vector3d mls_yaxis = pvr.pv.mls.mean + 0.1 * r_ * pvr.pv.mls.v_axis;
      Eigen::Vector3d mls_zaxis = pvr.pv.mls.mean + 0.1 * r_ * pvr.pv.mls.plane_normal;
      viewer_->addLine(mls_mean, utils::convertEigenToPCL(mls_xaxis), 255, 0, 0, "MLSXAxis", 4);
      viewer_->addLine(mls_mean, utils::convertEigenToPCL(mls_yaxis), 0, 255, 0, "MLSYAxis", 4);
      viewer_->addLine(mls_mean, utils::convertEigenToPCL(mls_zaxis), 0, 0, 255, "MLSZAxis", 4);

      viewer_->addSphere(utils::convertEigenToPCL(projected_pt), 0.02 * r_, 255, 128, 0, "MLSProjection", 4);
      viewer_->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 0.2, "MLSProjection", 4);

      p3 = utils::convertEigenToPCL(pvr.tri.p[2]);
      viewer_->addLine<pcl::PointXYZ, pcl::PointXYZ>(p1, p3, 0, 255, 0, "RightSide", 1);       // Green
      viewer_->addLine<pcl::PointXYZ, pcl::PointXYZ>(p2, p3, 0, 255, 0, "LeftSide", 1);        // Green
      viewer_->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 8, "RightSide", 1);
      viewer_->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 8, "LeftSide", 1);
      #endif

      if (pvr.status == PredictVertexResults::InvalidMLSResults)
      {
        #ifdef AFRONTDEBUG
        std::printf("\x1B[31m  Invalid MLS Results for nearest point!\x1B[0m\n");
        #endif
        boundary_.push_back(afront.front.he);
      }
      else if (pvr.status == PredictVertexResults::InvalidProjection)
      {
        #ifdef AFRONTDEBUG
        std::printf("\x1B[31m  Invalid Projection!\x1B[0m\n");
        #endif
        boundary_.push_back(afront.front.he);
      }
      else if (pvr.status == PredictVertexResults::AtBoundary)
      {
        #ifdef AFRONTDEBUG
        std::printf("\x1B[36m  At Point Cloud Boundary!\x1B[0m\n");
        #endif
        boundary_.push_back(afront.front.he);
      }
      else if (pvr.status == PredictVertexResults::InvalidStepSize)
      {
        #ifdef AFRONTDEBUG
        std::printf("\x1B[31m  Calculated Invalid Step Size!\x1B[0m\n");
        #endif
        boundary_.push_back(afront.front.he);
      }
      else
      {
        TriangleToCloseResults ttcr = isTriangleToClose(pvr);
        if (!ttcr.found)
        {
          if (pvr.status == PredictVertexResults::Valid)
          {
            #ifdef AFRONTDEBUG
            std::printf("\x1B[32m  Performed Grow Opperation\x1B[0m\n");
            #endif
            grow(pvr);
          }
          else if (pvr.status == PredictVertexResults::InvalidVertexNormal)
          {
            #ifdef AFRONTDEBUG
            std::printf("\x1B[31m  Projection point has inconsistant vertex normal!\x1B[0m\n");
            #endif
            boundary_.push_back(afront.front.he);
          }
          else if (pvr.status == PredictVertexResults::InvalidTriangleNormal)
          {
            #ifdef AFRONTDEBUG
            std::printf("\x1B[31m  Projection point triangle normal is inconsistant with vertex normals!\x1B[0m\n");
            #endif
            boundary_.push_back(afront.front.he);
          }
        }
        else
        {
          if (!ttcr.tri.vertex_normals_valid)
          {
            #ifdef AFRONTDEBUG
            std::printf("\x1B[31m  Closest point has inconsistant vertex normal!\x1B[0m\n");
            #endif
            boundary_.push_back(afront.front.he);
          }
          else if (!ttcr.tri.triangle_normal_valid)
          {
            #ifdef AFRONTDEBUG
            std::printf("\x1B[31m  Closest point triangle normal is inconsistant with vertex normals!\x1B[0m\n");
            #endif
            boundary_.push_back(afront.front.he);
          }
          else
          {
            #ifdef AFRONTDEBUG
            std::printf("\x1B[33m  Performed Topology Event Opperation\x1B[0m\n");
            #endif
            merge(ttcr);
          }
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
    mesh_tree_->setSortedResults(true);
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
    sp1.point.intensity = getMaxStep(p1);

    // search for the nearest neighbor
    std::vector<int> K;
    std::vector<float> K_dist;
    mls_cloud_tree_->nearestKSearch(utils::convertPointXYZINormalToPointNormal(sp1.point), 2, K, K_dist);

    // use l1 and nearest neighbor to extend edge
    pcl::PointNormal dp;
    MLSSampling::SamplePointResults sp2, sp3;
    Eigen::Vector3f p2, p3, v1, v2, mp, norm, proj;

    dp = mls_cloud_->points[K[1]];
    v1 = dp.getVector3fMap() - p1;
    v1 = v1.normalized();
    norm = dp.getNormalVector3fMap();

    proj = p1 + sp1.point.intensity * v1;
    sp2 = mls_.samplePoint(pcl::PointXYZ(proj(0), proj(1), proj(2)));
    p2 = sp2.point.getVector3fMap();
    sp2.point.intensity = getMaxStep(p2);

    mp = utils::getMidPoint(p1, p2);
    double d = utils::distPoint2Point(p1, p2);
    max_edge_length_ = utils::distPoint2Point(p1, p2);

    v2 = norm.cross(v1).normalized();

    double max_step = std::min(sp1.point.intensity, sp2.point.intensity);
    double l = std::sqrt(pow(max_step, 2.0) - pow(d / 2.0, 2.0)); // Calculate the height of the triangle
    proj = mp + l * v2;
    sp3 = mls_.samplePoint(pcl::PointXYZ(proj(0), proj(1), proj(2)));
    p3 = sp3.point.getVector3fMap();
    sp3.point.intensity = getMaxStep(p3);

    d = utils::distPoint2Point(p1, p3);
    if (d > max_edge_length_)
      max_edge_length_ = d;

    d = utils::distPoint2Point(p2, p3);
    if (d > max_edge_length_)
      max_edge_length_ = d;

    // Align normals
    utils::alignNormal(sp2.point, sp1.point);
    utils::alignNormal(sp3.point, sp1.point);

    MeshTraits::FaceData fd;
    Eigen::Vector3f center = (p1 + p2 + p3)/ 3;
    Eigen::Vector3f normal = ((p2 - p1).cross(p3 - p1)).normalized();
    utils::alignNormal(normal, sp1.point.getNormalVector3fMap());

    fd.x = center(0);
    fd.y = center(1);
    fd.z = center(2);
    fd.normal_x = normal(0);
    fd.normal_y = normal(1);
    fd.normal_z = normal(2);

    VertexIndices vi;
    vi.push_back(mesh_.addVertex(sp1.point));
    vi.push_back(mesh_.addVertex(sp2.point));
    vi.push_back(mesh_.addVertex(sp3.point));
    FaceIndex fi = mesh_.addFace(vi[0], vi[1], vi[2], fd);
    addToQueue(fi);
  }

  void AfrontMeshing::cutEar(const CutEarData &ccer)
  {
//    assert(ccer.tri.point_valid);
    if (ccer.tri.B > max_edge_length_)
      max_edge_length_ = ccer.tri.B;

    if (ccer.tri.C > max_edge_length_)
      max_edge_length_ = ccer.tri.C;

    MeshTraits::FaceData new_fd = createFaceData(ccer.tri);
    FaceIndex fi = mesh_.addFace(ccer.vi[0], ccer.vi[1], ccer.vi[2], new_fd);

    if (addToQueue(fi))
    {
      removeFromQueue(ccer.secondary);
      removeFromBoundary(ccer.secondary);
    }
    else
    {
      addToBoundary(ccer.primary);
    }
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

    pcl::PointXYZINormal p1, p2;
    p1 = mesh_vertex_data_[result.front.vi[0].get()];
    p2 = mesh_vertex_data_[result.front.vi[1].get()];

    result.front.p[0] = p1.getVector3fMap();
    result.front.p[1] = p2.getVector3fMap();
    result.front.n[0] = p1.getNormalVector3fMap();
    result.front.n[1] = p2.getNormalVector3fMap();

    // Calculate the half edge length
    result.front.length = utils::distPoint2Point(result.front.p[0], result.front.p[1]);

    // Get half edge midpoint
    result.front.mp = utils::getMidPoint(result.front.p[0], result.front.p[1]);

    // Calculate the grow direction vector
    result.front.d = getGrowDirection(result.front.p[0], result.front.mp, fd);

    // Get the maximum grow distance
    result.front.max_step = std::min(p1.intensity, p2.intensity);

    // Check Next Half Edge
    result.next = getNextHalfEdge(result.front);

    // Check Prev Half Edge
    result.prev = getPrevHalfEdge(result.front);

    return result;
  }

  AfrontMeshing::CutEarData AfrontMeshing::getNextHalfEdge(const FrontData &front) const
  {
    CutEarData next;
    next.type = CutEarData::NextHalfEdge;
    next.primary = front.he;
    next.secondary = mesh_.getNextHalfEdgeIndex(front.he);
    next.vi[0] = front.vi[0];
    next.vi[1] = front.vi[1];
    next.vi[2] = mesh_.getTerminatingVertexIndex(next.secondary);
    next.tri = getTriangleData(front, mesh_vertex_data_[next.vi[2].get()]);

    OHEAVC circ_next = mesh_.getOutgoingHalfEdgeAroundVertexCirculator(front.vi[1]);
    const OHEAVC circ_next_end = circ_next;
    do
    {
      HalfEdgeIndex he = circ_next.getTargetIndex();
      if (!mesh_.isValid(he))
        continue;

      VertexIndex evi = mesh_.getTerminatingVertexIndex(he);
      if (!mesh_.isBoundary(he))
      {
        he = mesh_.getOppositeHalfEdgeIndex(he);
        if (!mesh_.isBoundary(he))
          continue;
      }

      if (he == front.he)
        continue;

      TriangleData tri = getTriangleData(front, mesh_vertex_data_[evi.get()]);
      if ((tri.point_valid && !next.tri.point_valid) || (tri.point_valid && next.tri.point_valid && tri.c < next.tri.c))
      {
        next.secondary = he;
        next.vi[0] = front.vi[0];
        next.vi[1] = front.vi[1];
        next.vi[2] = evi;
        next.tri = tri;
      }
    } while (++circ_next != circ_next_end);

    return next;
  }

  AfrontMeshing::CutEarData AfrontMeshing::getPrevHalfEdge(const FrontData &front) const
  {
    CutEarData prev;
    prev.type = CutEarData::PrevHalfEdge;
    prev.primary = front.he;
    prev.secondary = mesh_.getPrevHalfEdgeIndex(front.he);
    prev.vi[0] = front.vi[0];
    prev.vi[1] = front.vi[1];
    prev.vi[2] = mesh_.getOriginatingVertexIndex(prev.secondary);
    prev.tri = getTriangleData(front, mesh_vertex_data_[prev.vi[2].get()]);

    OHEAVC circ_prev = mesh_.getOutgoingHalfEdgeAroundVertexCirculator(front.vi[0]);
    const OHEAVC circ_prev_end = circ_prev;
    do
    {
      HalfEdgeIndex he = circ_prev.getTargetIndex();
      if (!mesh_.isValid(he))
        continue;

      VertexIndex evi = mesh_.getTerminatingVertexIndex(he);
      if (!mesh_.isBoundary(he))
      {
        he = mesh_.getOppositeHalfEdgeIndex(he);
        if (!mesh_.isBoundary(he))
          continue;
      }

      if (he == front.he)
        continue;

      TriangleData tri = getTriangleData(front, mesh_vertex_data_[evi.get()]);

      if ((tri.point_valid && !prev.tri.point_valid) || (tri.point_valid && prev.tri.point_valid && tri.b < prev.tri.b))
      {
        prev.secondary = he;
        prev.vi[0] = front.vi[0];
        prev.vi[1] = front.vi[1];
        prev.vi[2] = evi;
        prev.tri = tri;
      }
    } while (++circ_prev != circ_prev_end);

    return prev;
  }

  AfrontMeshing::PredictVertexResults AfrontMeshing::predictVertex(const AdvancingFrontData &afront) const
  {
    // Local Variables
    PredictVertexResults result;
    const FrontData &front = afront.front;

    result.status = PredictVertexResults::Valid;
    result.afront = afront;

    // Get new point for growing a triangle to be projected onto the mls surface
    double l = std::sqrt(pow(front.max_step, 2.0) - pow(front.length / 2.0, 2.0)); // Calculate the height of the triangle

    // This is required because 2 * step size < front.length results in nan
    if (!std::isnan(l))
    {
      // Get predicted vertex
      Eigen::Vector3f p = front.mp + l * front.d;
      result.pv = mls_.samplePoint(pcl::PointXYZ(p(0), p(1), p(2)));
      utils::alignNormal(result.pv.point, mesh_vertex_data_[afront.front.vi[0].get()]);
      if (result.pv.mls.num_neighbors < required_neighbors_) // Maybe we should use the 5 * DOF that PCL uses
      {
        result.status = PredictVertexResults::InvalidMLSResults;
        return result;
      }

      // Check if the projected point is to far from the original point.
      double dist = utils::distPoint2Point(result.pv.orig.getVector3fMap(), result.pv.point.getVector3fMap());
      if (dist > front.max_step)
      {
        result.status = PredictVertexResults::InvalidProjection;
        return result;
      }

      // Check and see if there any point in the grow direction of the front.
      // If not then it is at the boundary of the point cloud.
      if (nearBoundary(front, result.pv.closest))
      {
        result.status = PredictVertexResults::AtBoundary;
        return result;
      }

      // Get triangle Data
      result.tri = getTriangleData(front, result.pv.point);
      if (!result.tri.point_valid)
        result.status = PredictVertexResults::InvalidProjection;
      else if(!result.tri.vertex_normals_valid)
        result.status = PredictVertexResults::InvalidVertexNormal;
      else if (!result.tri.triangle_normal_valid)
        result.status = PredictVertexResults::InvalidTriangleNormal;

      return result;
    }
    else
    {
      result.status = PredictVertexResults::InvalidStepSize;
      return result;
    }
  }

  AfrontMeshing::CloseProximityResults AfrontMeshing::isCloseProximity(const PredictVertexResults &pvr) const
  {
    CloseProximityResults results;
    std::vector<int> K;
    std::vector<float> K_dist;

    mesh_tree_->radiusSearch(pvr.pv.point, 1.0 * pvr.afront.front.max_step, K, K_dist);

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
        pcl::PointXYZINormal chkpn = mesh_vertex_data_[vi.get()];
        Eigen::Vector3f chkpt = chkpn.getVector3fMap();
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

          // Don't include any edge attached to the half edge.
          if (evi == pvr.afront.front.vi[0] || evi == pvr.afront.front.vi[1])
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
        if (dist < 0.5 * pvr.afront.front.max_step)
        {
          TriangleData tri = getTriangleData(pvr.afront.front, chkpn);
          if (!results.found)
          {
            results.found = true;
            results.dist = dist;
            results.closest = vi;
            results.tri = tri;
          }
          else
          {
            if ((results.tri.valid && tri.valid && dist < results.dist) ||
                (!results.tri.valid && !tri.valid && dist < results.dist) ||
                (!results.tri.valid && tri.valid))
            {
              results.dist = dist;
              results.closest = vi;
              results.tri = tri;
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
    // TODO: Should we be checking if triangle normals are valid?  
    double ar_tol = 0.85;
    double prev_dist = utils::distPoint2Line(pvr.afront.prev.tri.p[0], pvr.afront.prev.tri.p[2], pvr.tri.p[2]).d;
    double next_dist = utils::distPoint2Line(pvr.afront.next.tri.p[1], pvr.afront.next.tri.p[2], pvr.tri.p[2]).d;
    if (pvr.afront.prev.tri.point_valid && pvr.afront.next.tri.point_valid)
    {
      if (pvr.afront.prev.tri.aspect_ratio >= ar_tol || pvr.afront.next.tri.aspect_ratio >= ar_tol)
      {
        if (pvr.afront.prev.tri.aspect_ratio > pvr.afront.next.tri.aspect_ratio)
        {
          results.found = true;
          results.dist = prev_dist;
          results.closest = pvr.afront.prev.vi[2];
          results.tri = pvr.afront.prev.tri;
        }
        else
        {
          results.found = true;
          results.dist = next_dist;
          results.closest = pvr.afront.next.vi[2];
          results.tri = pvr.afront.next.tri;
        }
      }
      else
      {
        if (prev_dist < next_dist)
        {
          if (prev_dist < 0.5 * pvr.afront.front.max_step)
          {
            results.found = true;
            results.dist = prev_dist;
            results.closest = pvr.afront.prev.vi[2];
            results.tri = pvr.afront.prev.tri;
          }
        }
        else
        {
          if (next_dist < 0.5 * pvr.afront.front.max_step)
          {
            results.found = true;
            results.dist = next_dist;
            results.closest = pvr.afront.next.vi[2];
            results.tri = pvr.afront.next.tri;
          }
        }
      }
    }
    else if ((pvr.afront.prev.tri.point_valid && prev_dist < 0.5 * pvr.afront.front.max_step) || (pvr.afront.prev.tri.point_valid && pvr.afront.prev.tri.aspect_ratio >= ar_tol))
    {
      results.found = true;
      results.dist = prev_dist;
      results.closest = pvr.afront.prev.vi[2];
      results.tri = pvr.afront.prev.tri;
    }
    else if ((pvr.afront.next.tri.point_valid && next_dist < 0.5 * pvr.afront.front.max_step) || (pvr.afront.next.tri.point_valid && pvr.afront.next.tri.aspect_ratio >= ar_tol))
    {
      results.found = true;
      results.dist = next_dist;
      results.closest = pvr.afront.next.vi[2];
      results.tri = pvr.afront.next.tri;
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

        // It is not suffecient to just compare half edge indexs because non manifold mesh is allowed.
        // Must check vertex index
        if (vi[0] == pvr.afront.front.vi[0] || vi[1] == pvr.afront.front.vi[0] || vi[0] == pvr.afront.front.vi[1] || vi[1] == pvr.afront.front.vi[1])
          continue;

        Eigen::Vector3f p1 = mesh_vertex_data_[vi[0].get()].getVector3fMap();
        Eigen::Vector3f p2 = mesh_vertex_data_[vi[1].get()].getVector3fMap();

        utils::DistPoint2LineResults dist = utils::distPoint2Line(p1, p2, pvr.tri.p[2]);
        if (dist.d < 0.5 * pvr.afront.front.max_step)
        {

          bool check_p1 = isPointValid(pvr.afront.front, p1);
          bool check_p2 = isPointValid(pvr.afront.front, p2);
          int index;
          if (check_p1 && check_p2)
          {
            // TODO: Should check if triangle is valid for these points. If both are then check distance
            //       otherwise use the one that creates a valid triangle.
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
            results.tri = getTriangleData(pvr.afront.front, mesh_vertex_data_[results.closest.get()]);
          }
          else
          {
            results.found = true;
            results.dist = dist.d;
            results.closest = vi[index];
            results.tri = getTriangleData(pvr.afront.front, mesh_vertex_data_[results.closest.get()]);
          }
        }
      }
    }

    #ifdef AFRONTDEBUG
    Eigen::Vector3f p;
    p = pvr.tri.p[2];
    viewer_->addSphere(utils::convertEigenToPCL(p), 0.5 * pvr.afront.front.max_step, 0, 255, 128, "ProxRadius", 1);
    viewer_->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 0.2, "ProxRadius", 1);
    #endif

    return results;
  }

  bool AfrontMeshing::checkPrevNextHalfEdge(const AdvancingFrontData &afront, TriangleData &tri, VertexIndex &closest) const
  {
    if (mesh_.isValid(closest) && closest == afront.prev.vi[2])
      if (afront.next.tri.point_valid && tri.c >= afront.next.tri.c)
      {
        closest = afront.next.vi[2];
        tri = afront.next.tri;
        return true;
      }

    if (mesh_.isValid(closest) && closest == afront.next.vi[2])
      if (afront.prev.tri.point_valid && tri.b >= afront.prev.tri.b)
      {
        closest = afront.prev.vi[2];
        tri = afront.prev.tri;
        return true;
      }

    if ((afront.next.tri.point_valid && tri.c >= afront.next.tri.c) && (afront.prev.tri.point_valid && tri.b >= afront.prev.tri.b))
    {
      if (afront.next.tri.c < afront.prev.tri.b)
      {
        closest = afront.next.vi[2];
        tri = afront.next.tri;
        return true;
      }
      else
      {
        closest = afront.prev.vi[2];
        tri = afront.prev.tri;
        return true;
      }
    }
    else if (afront.next.tri.point_valid && tri.c >= afront.next.tri.c)
    {
      closest = afront.next.vi[2];
      tri = afront.next.tri;
      return true;
    }
    else if (afront.prev.tri.point_valid && tri.b >= afront.prev.tri.b)
    {
      closest = afront.prev.vi[2];
      tri = afront.prev.tri;
      return true;
    }

    utils::IntersectionLine2PlaneResults lpr;
    if (afront.next.tri.point_valid && isFenceViolated(afront.front.p[0], tri.p[2], afront.next.secondary, 2.0 * afront.next.tri.B * hausdorff_error_, lpr))
    {
      closest = afront.next.vi[2];
      tri = afront.next.tri;
      return true;
    }

    if (afront.prev.tri.point_valid && isFenceViolated(afront.front.p[1], tri.p[2], afront.prev.secondary, 2.0 * afront.prev.tri.C * hausdorff_error_, lpr))
    {
      closest = afront.prev.vi[2];
      tri = afront.prev.tri;
      return true;
    }

    return false;
  }

  bool AfrontMeshing::isFenceViolated(const Eigen::Vector3f &sp, const Eigen::Vector3f &ep, const HalfEdgeIndex &fence, const double fence_height, utils::IntersectionLine2PlaneResults &lpr) const
  {
    // Check for fence intersection
    Eigen::Vector3f he_p1, he_p2;
    he_p1 = (mesh_vertex_data_[mesh_.getOriginatingVertexIndex(fence).get()]).getVector3fMap();
    he_p2 = (mesh_vertex_data_[mesh_.getTerminatingVertexIndex(fence).get()]).getVector3fMap();

    MeshTraits::FaceData fd = mesh_.getFaceDataCloud()[mesh_.getOppositeFaceIndex(fence).get()];

    Eigen::Vector3f u = he_p2 - he_p1;
    Eigen::Vector3f n = fd.getNormalVector3fMap();
    Eigen::Vector3f v = n * fence_height;

    // Project the line on to the triangle plane. Note this is different from the paper
    Eigen::Vector3f sp_proj = sp - (sp - he_p1).dot(n) * n;
    Eigen::Vector3f ep_proj = ep - (ep - he_p1).dot(n) * n;
    lpr = utils::intersectionLine2Plane(sp_proj, ep_proj, he_p1, u, v);

//    lpr = utils::intersectionLine2Plane(sp, ep, he_p1, u, v);

    if (!lpr.parallel) // May need to add additional check if parallel
      if (lpr.mw <= 1 && lpr.mw >= 0)      // This checks if line segement intersects the plane
        if (lpr.mu <= 1 && lpr.mu >= 0)    // This checks if intersection point is within the x range of the plane
          if (lpr.mv <= 1 && lpr.mv >= -1) // This checks if intersection point is within the y range of the plane
            return true;

    return false;
  }

  AfrontMeshing::FenceViolationResults AfrontMeshing::isFencesViolated(const VertexIndex &vi, const Eigen::Vector3f &p, const std::vector<HalfEdgeIndex> &fences, const VertexIndex &closest, const PredictVertexResults &pvr) const
  {
    // Now need to check for fence violation
    FenceViolationResults results;
    results.found = false;
    Eigen::Vector3f sp = mesh_vertex_data_[vi.get()].getVector3fMap();

    for(auto i = 0; i < fences.size(); ++i)
    {
      // The list of fences should not include any that are associated to the requesting vi
      assert(vi != mesh_.getOriginatingVertexIndex(fences[i]));
      assert(vi != mesh_.getTerminatingVertexIndex(fences[i]));

      // Need to ignore fence check if closest point is associated to the fence half edge
      if (mesh_.isValid(closest))
        if (closest == mesh_.getOriginatingVertexIndex(fences[i]) || closest == mesh_.getTerminatingVertexIndex(fences[i]))
          continue;

      utils::IntersectionLine2PlaneResults lpr;
      double fence_height = 2.0 * pvr.afront.front.max_step * hausdorff_error_;
      bool fence_violated = isFenceViolated(sp, p, fences[i], fence_height, lpr);

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
      results.tri = cpr.tri;
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
        fvr = isFencesViolated(pvr.afront.prev.vi[1], pvr.afront.prev.tri.p[2], cpr.fences, results.closest, pvr);
      }
      else if (results.found && results.closest == pvr.afront.next.vi[2])
      {
        fvr = isFencesViolated(pvr.afront.prev.vi[0], pvr.afront.next.tri.p[2], cpr.fences, results.closest, pvr);
      }
      else
      {
        FenceViolationResults fvr1, fvr2, min_fvr;
        fvr1 = isFencesViolated(pvr.afront.front.vi[0], results.tri.p[2], cpr.fences, results.closest, pvr);
        fvr2 = isFencesViolated(pvr.afront.front.vi[1], results.tri.p[2], cpr.fences, results.closest, pvr);
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
          // TODO: Should check if triangle is valid for these points. If both are then check distance
          //       otherwise use the one that creates a valid triangle.
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
        results.tri = getTriangleData(pvr.afront.front, mesh_vertex_data_[results.closest.get()]);

        checkPrevNextHalfEdge(pvr.afront, results.tri, results.closest);
      }
      else
      {
        #ifdef AFRONTDEBUG
        Eigen::Vector3f p = pvr.tri.p[2];
        if (results.found)
          p = results.tri.p[2];

        viewer_->addSphere(utils::convertEigenToPCL(p), 0.1 * pvr.afront.front.max_step, 255, 255, 0, "Closest", 1);
        #endif
        return results;
      }
    }

    // Need to print a warning message it should never get here.
    return results;
  }

  bool AfrontMeshing::isPointValid(const FrontData &front, const Eigen::Vector3f p) const
  {
    Eigen::Vector3f v = p - front.mp;
    double dot = v.dot(front.d);

    if ((dot > 0.0))
      return true;

    return false;
  }

  bool AfrontMeshing::isBoundaryPoint(const int index) const
  {
    pcl::PointNormal closest = mls_cloud_->at(index);

    Eigen::Vector4f u;
    Eigen::Vector4f v;
    std::vector<int> K;
    std::vector<float> K_dist;

    u << mls_.getMLSResult(index).u_axis.cast<float>().array(), 0.0;
    v << mls_.getMLSResult(index).v_axis.cast<float>().array(), 0.0;

    // Need to modify mls library to store indicies instead of just the number of neighbors
    mls_cloud_tree_->radiusSearch(closest, r_, K, K_dist);

    if (K.size () < 3)
      return (false);

    if (!pcl_isfinite (closest.x) || !pcl_isfinite (closest.y) || !pcl_isfinite (closest.z))
      return (false);

    // Compute the angles between each neighboring point and the query point itself
    std::vector<float> angles (K.size ());
    float max_dif = FLT_MIN, dif;
    int cp = 0;

    for (size_t i = 0; i < K.size (); ++i)
    {
      if (!pcl_isfinite (mls_cloud_->points[K[i]].x) ||
          !pcl_isfinite (mls_cloud_->points[K[i]].y) ||
          !pcl_isfinite (mls_cloud_->points[K[i]].z))
        continue;

      Eigen::Vector4f delta = mls_cloud_->points[K[i]].getVector4fMap () - closest.getVector4fMap ();
      if (delta == Eigen::Vector4f::Zero())
        continue;

      angles[cp++] = atan2f (v.dot (delta), u.dot (delta)); // the angles are fine between -PI and PI too
    }
    if (cp == 0)
      return (false);

    angles.resize (cp);
    std::sort (angles.begin (), angles.end ());

    // Compute the maximal angle difference between two consecutive angles
    for (size_t i = 0; i < angles.size () - 1; ++i)
    {
      dif = angles[i + 1] - angles[i];
      if (max_dif < dif)
        max_dif = dif;
    }

    // Get the angle difference between the last and the first
    dif = 2 * static_cast<float> (M_PI) - angles[angles.size () - 1] + angles[0];
    if (max_dif < dif)
      max_dif = dif;

    // Check results
    if (max_dif > boundary_angle_threshold_)
      return (true);
    else
      return (false);
  }

  bool AfrontMeshing::nearBoundary(const FrontData &front, const int index) const
  {
    pcl::PointNormal closest = mls_cloud_->at(index);

    Eigen::Vector3f v1 = (front.p[1] - front.mp).normalized();
    Eigen::Vector3f v2 = closest.getVector3fMap() - front.mp;
    double dot1 = v2.dot(front.d);
    double dot2 = v2.dot(v1);

    if ((dot1 > 0.0) && (std::abs(dot2) <= front.length/2.0))
      return false;

    // Try the pcl boundary search only if the simple check says it is a boundary
    return isBoundaryPoint(index);

    return true;
  }

  void AfrontMeshing::grow(const PredictVertexResults &pvr)
  {
    // Add new face
    MeshTraits::FaceData new_fd = createFaceData(pvr.tri);
    pcl::PointXYZINormal p = pvr.pv.point;
    p.intensity = getMaxStep(pvr.tri.p[2]);
    FaceIndex fi = mesh_.addFace(pvr.afront.front.vi[0], pvr.afront.front.vi[1], mesh_.addVertex(p), new_fd);

    if (pvr.tri.B > max_edge_length_)
      max_edge_length_ = pvr.tri.B;

    if (pvr.tri.C > max_edge_length_)
      max_edge_length_ = pvr.tri.C;

    // Add new half edges to the queue
    if (!addToQueue(fi))
      addToBoundary(pvr.afront.front.he);
  }

  void AfrontMeshing::merge(const TriangleToCloseResults &ttcr)
  {
    assert(ttcr.closest != ttcr.pvr.afront.front.vi[0]);
    assert(ttcr.closest != ttcr.pvr.afront.front.vi[1]);
    // Need to make sure at lease one vertex of the half edge is in the grow direction
    if (ttcr.closest == ttcr.pvr.afront.prev.vi[2])
    {
      #ifdef AFRONTDEBUG
      std::printf("\x1B[32m\tAborting Merge, Forced Ear Cut Opperation with Previous Half Edge!\x1B[0m\n");
      #endif
      cutEar(ttcr.pvr.afront.prev);
      return;
    }
    else if (ttcr.closest == ttcr.pvr.afront.next.vi[2])
    {
      #ifdef AFRONTDEBUG
      std::printf("\x1B[32m\tAborting Merge, Forced Ear Cut Opperation with Next Half Edge!\x1B[0m\n");
      #endif
      cutEar(ttcr.pvr.afront.next);
      return;
    }

    if (ttcr.tri.B > max_edge_length_)
      max_edge_length_ = ttcr.tri.B;

    if (ttcr.tri.C > max_edge_length_)
      max_edge_length_ = ttcr.tri.C;

    MeshTraits::FaceData new_fd = createFaceData(ttcr.tri);
    FaceIndex fi = mesh_.addFace(ttcr.pvr.afront.front.vi[0], ttcr.pvr.afront.front.vi[1], ttcr.closest, new_fd);

    if (!addToQueue(fi))
      addToBoundary(ttcr.pvr.afront.front.he);

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

  double AfrontMeshing::getMaxStep(const Eigen::Vector3f &p) const
  {
    pcl::PointNormal pn;
    std::vector<int> k;
    std::vector<float> k_dist;
    pn.x = p(0);
    pn.y = p(1);
    pn.z = p(2);

    // What is shown in the afront paper. Need to figure out how to transverse the kdtree.
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
        int neighbors = mls_.getMLSResult(i).num_neighbors;
        if (neighbors < required_neighbors_)
          continue;

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

      if (cnt != 0)
        pcnt = cnt - 1;

      ++j;
    }

    if (!finished)
      std::printf("\x1B[32m  Warning Max Step Not Found! Length: %f\x1B[0m\n", len);

    return len;
  }

  AfrontMeshing::TriangleData AfrontMeshing::getTriangleData(const FrontData &front, const pcl::PointXYZINormal p) const
  {
    TriangleData result;
    Eigen::Vector3f v1, v2, v3, cross;
    double dot, sina, cosa, top, bottom, area;
    Eigen::Vector3f p3 = p.getArray3fMap();
    Eigen::Vector3f p3_normal = p.getNormalVector3fMap();

    result.valid = false;
    result.point_valid = true;
    result.vertex_normals_valid = true;
    result.triangle_normal_valid = true;

    v1 = front.p[1] - front.p[0];
    v2 = p3 - front.p[1];
    v3 = p3 - front.p[0];

    dot = v2.dot(front.d);
    if (dot <= 0.0)
      result.point_valid = false;

    result.A = v1.norm();
    result.B = v2.norm();
    result.C = v3.norm();

    // Calculate the first angle of triangle
    dot = v1.dot(v3);
    cross = v1.cross(v3);
    bottom = result.A * result.C;
    top = cross.norm();
    sina = top/bottom;
    cosa = dot/bottom;
    result.b = atan2(sina, cosa);

    area = 0.5 * top;
    result.aspect_ratio = (4 * area * std::sqrt(3)) / (result.A * result.A + result.B * result.B + result.C * result.C);
    result.normal = cross.normalized();
    utils::alignNormal(result.normal, front.n[0]);
    assert(!std::isnan(result.normal[0]));

    // Lets check triangle and vertex normals
    if (!utils::checkNormal(front.n[0], p3_normal, vertex_normal_tol_) || !utils::checkNormal(front.n[1], p3_normal, vertex_normal_tol_))
    {
      result.vertex_normals_valid = false;
      result.triangle_normal_valid = false;
    }
    else
    {
      Eigen::Vector3f avg_normal = (front.n[0] + front.n[1] + p3_normal) / 3.0;
      if (!utils::checkNormal(avg_normal, result.normal, triangle_normal_tol_))
        result.triangle_normal_valid = false;
    }

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

    result.valid = result.point_valid && result.vertex_normals_valid && result.triangle_normal_valid;
    return result;
  }

  AfrontMeshing::MeshTraits::FaceData AfrontMeshing::createFaceData(const TriangleData &tri) const
  {
    MeshTraits::FaceData center_pt;
    Eigen::Vector3f cp = (tri.p[0] + tri.p[1] + tri.p[2]) / 3.0;

    center_pt.x = cp(0);
    center_pt.y = cp(1);
    center_pt.z = cp(2);
    center_pt.normal_x = tri.normal(0);
    center_pt.normal_y = tri.normal(1);
    center_pt.normal_z = tri.normal(2);

    return center_pt;
  }

  void AfrontMeshing::addToQueueHelper(const HalfEdgeIndex &half_edge)
  {
    assert(std::find(queue_.begin(), queue_.end(), half_edge) == queue_.end());
    queue_.push_back(half_edge);
  }

  bool AfrontMeshing::addToQueue(const FaceIndex &face)
  {
    // This occures if the face is non-manifold.
    // It appears that non-manifold vertices are allowed but not faces.
    if (mesh_.isValid(face))
    {
      OHEAFC circ = mesh_.getOuterHalfEdgeAroundFaceCirculator(face);
      const OHEAFC circ_end = circ;
      do
      {
        HalfEdgeIndex he = circ.getTargetIndex();
        if (mesh_.isBoundary(he))
          addToQueueHelper(he);

      } while (++circ != circ_end);
    }
    else
    {
      std::printf("\x1B[32m\tUnable to perform merge, invalid face index!\x1B[0m\n");
      return false;
    }
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

  pcl::PolygonMesh AfrontMeshing::getPolynomialSurface(const PredictVertexResults &pvr, const double step) const
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr poly (new pcl::PointCloud<pcl::PointXYZ> ());

    int wh = 2 * r_/step + 1;
    poly->width = wh;
    poly->height = wh;
    poly->points.resize(wh * wh);
    int npoints = 0;
    for (auto i = 0; i < wh; i++)
    {
      double u = i * step - r_;
      for (auto j = 0; j < wh; j++)
      {
        double v = j * step - r_;
        double w = mls_.getPolynomialValue(u, v, pvr.pv.mls);
        poly->points[npoints].x = static_cast<float> (pvr.pv.mls.mean[0] + pvr.pv.mls.u_axis[0] * u + pvr.pv.mls.v_axis[0] * v + pvr.pv.mls.plane_normal[0] * w);
        poly->points[npoints].y = static_cast<float> (pvr.pv.mls.mean[1] + pvr.pv.mls.u_axis[1] * u + pvr.pv.mls.v_axis[1] * v + pvr.pv.mls.plane_normal[1] * w);
        poly->points[npoints].z = static_cast<float> (pvr.pv.mls.mean[2] + pvr.pv.mls.u_axis[2] * u + pvr.pv.mls.v_axis[2] * v + pvr.pv.mls.plane_normal[2] * w);
        npoints++;
      }
    }

    pcl::PolygonMesh output;
    pcl::OrganizedFastMesh<pcl::PointXYZ> ofm;
    ofm.setInputCloud(poly);
    ofm.setMaxEdgeLength(2 * step);
    ofm.setTriangulationType(pcl::OrganizedFastMesh<pcl::PointXYZ>::QUAD_MESH);
    ofm.reconstruct(output);
    return output;
  }

#ifdef AFRONTDEBUG
  void AfrontMeshing::keyboardEventOccurred(const pcl::visualization::KeyboardEvent &event, void*)
  {
    if (event.keyUp())
    {
      std::string k = event.getKeySym();
      if (isdigit(k[0]))
      {
        int num = std::stoi(k);
        int length = std::pow(10, num);
        if (!isFinished())
        {
          for (auto i = 0; i < length; ++i)
          {
            try
            {
              stepMesh();
            }
            catch (const std::exception& e)
            {
              std::printf("\x1B[31m\tFailed to step mesh!\x1B[0m\n");
              break;
            }

            if (isFinished())
              break;
          }
        }
        viewer_->removePolygonMesh();
        viewer_->addPolygonMesh(getMesh());
        return;
      }
    }
  }
#endif
}

