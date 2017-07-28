#ifndef AFRONT_MESHING_H
#define AFRONT_MESHING_H

//#define AFRONTDEBUG
#undef AFRONTDEBUG

#include <pcl/geometry/polygon_mesh.h>

#ifdef AFRONTDEBUG
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/impl/point_cloud_geometry_handlers.hpp>
#endif

// These are required for using custom point type
#include <pcl/octree/octree_search.h>
#include <pcl/octree/impl/octree_search.hpp>
#include <pcl/octree/octree_pointcloud.h>
#include <pcl/octree/impl/octree_pointcloud.hpp>
#include <pcl/kdtree/flann.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>
#include <pcl/search/flann_search.h>
#include <pcl/search/impl/search.hpp>
#include <pcl/search/impl/kdtree.hpp>
#include <pcl/search/impl/flann_search.hpp>
// End for custom point type

#include <deque>
#include <meshing/mls_sampling.h>
#include <meshing/afront_utils.h>


namespace afront_meshing
{
  const double AFRONT_DEFAULT_REDUCTION = 0.8;
  const double AFRONT_DEFAULT_RHO = 0.9;
  const int    AFRONT_DEFAULT_THREADS = 1;
  const int    AFRONT_DEFAULT_POLYNOMIAL_ORDER = 2;
  const double AFRONT_DEFAULT_BOUNDARY_ANGLE_THRESHOLD = M_PI_2;

  const double AFRONT_ASPECT_RATIO_TOLERANCE = 0.85;
  const double AFRONT_CLOSE_PROXIMITY_FACTOR = 0.5;
  const double AFRONT_FENCE_HEIGHT_FACTOR = 2.0;

  class AfrontMeshing
  {
    struct MeshTraits
    {
      typedef AfrontVertexPointType   VertexData;
      typedef int               HalfEdgeData;
      typedef int               EdgeData;
      typedef pcl::PointNormal  FaceData;

      typedef boost::false_type IsManifold;
    };

    typedef pcl::geometry::PolygonMesh<MeshTraits> Mesh;

    typedef Mesh::VertexIndex   VertexIndex;
    typedef Mesh::HalfEdgeIndex HalfEdgeIndex;
    typedef Mesh::FaceIndex     FaceIndex;

    typedef Mesh::VertexIndices   VertexIndices;
    typedef Mesh::HalfEdgeIndices HalfEdgeIndices;
    typedef Mesh::FaceIndices     FaceIndices;

    typedef Mesh::VertexAroundVertexCirculator           VAVC;
    typedef Mesh::OutgoingHalfEdgeAroundVertexCirculator OHEAVC;
    typedef Mesh::IncomingHalfEdgeAroundVertexCirculator IHEAVC;
    typedef Mesh::FaceAroundVertexCirculator             FAVC;
    typedef Mesh::VertexAroundFaceCirculator             VAFC;
    typedef Mesh::InnerHalfEdgeAroundFaceCirculator      IHEAFC;
    typedef Mesh::OuterHalfEdgeAroundFaceCirculator      OHEAFC;

    struct TriangleData
    {
      double A;                   /**< @brief The length for the first half edge (p1->p2) */
      double B;                   /**< @brief The length for the second half edge (p2->p3) */
      double C;                   /**< @brief The length for the remaining side of the triangle (p3->p1) */
      double a;                   /**< @brief The angle BC */
      double b;                   /**< @brief The angle AC */
      double c;                   /**< @brief The anble AB */
      double aspect_ratio;        /**< @brief The quality of the triangle (1.0 is the best) */
      Eigen::Vector3f normal;     /**< @brief The normal of the triangle */
      Eigen::Vector3f p[3];       /**< @brief Stores the point information for the triangle */
      bool point_valid;           /**< @brief Indicates if the triangle is valid. Both edges must point in the same direction as the grow direction. */
      bool vertex_normals_valid;  /**< @brief The vertex normals are not within some tolerance */
      bool triangle_normal_valid; /**< @brief The triangle normal is not within some tolerance to the vertex normals*/
      bool valid;                 /**< @brief If all condition are valid: point_valid, vertex_normals_valid and triangle_normal_valid */

      void print(const std::string description = "") const
      {
        if (description == "")
          std::printf("Triangle Data:\n");
        else
          std::printf("Triangle Data (%s):\n", description.c_str());

        std::printf("\t  A: %-10.4f  B: %-10.4f  C: %-10.4f\n", A, B, C);
        std::printf("\t  a: %-10.4f  b: %-10.4f  c: %-10.4f\n", a, b, c);
        std::printf("\t x1: %-10.4f y1: %-10.4f z1: %-10.4f\n", p[0][0], p[0][1], p[0][2]);
        std::printf("\t x2: %-10.4f y2: %-10.4f z2: %-10.4f\n", p[1][0], p[1][1], p[1][2]);
        std::printf("\t x3: %-10.4f y3: %-10.4f z3: %-10.4f\n", p[2][0], p[2][1], p[2][2]);
        std::printf("\t Aspect Ratio: %-10.4f\n", aspect_ratio);
        std::printf("\t Valid: %s\n", valid ? "true" : "false");
      }
    };

    struct FrontData
    {
      HalfEdgeIndex he;              /**< @brief The half edge index from which to grow the triangle */
      double length;                 /**< @brief The half edge length */
      double max_step;               /**< @brief The maximum grow distance */
      double max_step_search_radius; /**< @brief The approximate search radius for the finding max step for new point. */
      Eigen::Vector3f mp;            /**< @brief The half edge mid point */
      Eigen::Vector3f d;             /**< @brief The half edge grow direction */
      VertexIndex vi[2];             /**< @brief The half edge vertex indicies */
      Eigen::Vector3f p[2];          /**< @brief The half edge points (Origninating, Terminating) */
      Eigen::Vector3f n[2];          /**< @brief The half edge point normals (Origninating, Terminating) */
    };

    struct CutEarData
    {
      HalfEdgeIndex primary;   /**< @brief The advancing front half edge */
      HalfEdgeIndex secondary; /**< @brief The Secondary half edge triangle (Previouse or Next) */
      VertexIndex vi[3];       /**< @brief The vertex indicies of the potential triangle */
      TriangleData tri;        /**< @brief The Triangle information */
    };

    struct AdvancingFrontData
    {
      FrontData front;    /**< @brief The front data */
      CutEarData prev;    /**< @brief The results using the previous half edge */
      CutEarData next;    /**< @brief The results using the next half edge */
    };

    struct PredictVertexResults
    {
      enum PredictVertexTypes
      {
        Valid = 0,                        /**< @brief The project point is valid. */
        AtBoundary = 1,                   /**< @brief At the point cloud boundary. */
        InvalidStepSize = 2,              /**< @brief The step size is invalid for the given half edge (2 * step size < front.length). */
        InvalidVertexNormal = 3,          /**< @brief The projected points normal is not consistant with the other triangle normals. */
        InvalidTriangleNormal = 4,        /**< @brief The triangle normal created by the project point is not consistant with the vertex normals. */
        InvalidMLSResults = 5,            /**< @brief The nearest points mls results are invalid. */
        InvalidProjection = 6             /**< @brief The projected point is not in the grow direction */
      };

      AdvancingFrontData afront;          /**< @brief Advancing front data */
      TriangleData tri;                   /**< @brief The proposed triangle data */
      MLSSampling::SamplePointResults pv; /**< @brief The predicted point projected on the mls surface */
      Eigen::Vector2d k;                  /**< @brief The principal curvature using the polynomial */
      PredictVertexTypes status;          /**< @brief The predicted vertex is near the boundry of the point cloud. Don't Create Triangle */
    };

    struct CloseProximityResults
    {
      std::vector<VertexIndex> verticies; /**< @brief The valid mesh verticies. */
      std::vector<HalfEdgeIndex> fences;  /**< @brief The valid half edges. */
      VertexIndex closest;                /**< @brief The closest mesh vertex */
      double dist;                        /**< @brief This stores closest distance information. */
      bool found;                         /**< @brief If close proximity was found. */
      TriangleData tri;                   /**< @brief The triangle data created by the closest point. */
    };

    struct FenceViolationResults
    {
      HalfEdgeIndex he;                         /**< @brief The half edge index that was violated. */
      int index;                                /**< @brief The index in the array CloseProximityResults.fences. */
      utils::IntersectionLine2PlaneResults lpr; /**< @brief The line to plane intersection results for fence violations. */
      double dist;                              /**< @brief The distance from the intersection point and the advancing front. */
      bool found;                               /**< @brief If a mesh half edge was violated. */
    };

    struct TriangleToCloseResults
    {
      PredictVertexResults pvr;                 /**< @brief The predicted vertex information provided */
      VertexIndex closest;                      /**< @brief The closest vertex index */
      TriangleData tri;                         /**< @brief The Triangle information */
      bool found;                               /**< @brief True if triangle is close, otherwise false */
    };

  public:
    /** @brief AfrontMeshing Constructor */
    AfrontMeshing();

    /** @brief AfrontMeshing Destructor */
    ~AfrontMeshing() {}

    /** @brief This will process the input parameters and comput the guidance field */
    bool initialize();

    /** @brief This will mesh the point cloud passed to the setInputCloud funciton */
    void reconstruct();

    /** @brief Advance the mesh by adding one triangle */
    void stepReconstruction();

    /** @brief Indicates if it has finished meshing the surface */
    bool isFinished() {return finished_;}

    /** @brief Get the current polygon mesh */
    pcl::PolygonMesh getMesh() const;

    /** @brief Get the mesh vertex normals */
    pcl::PointCloud<pcl::Normal>::ConstPtr getMeshVertexNormals() const;

    #ifdef AFRONTDEBUG
    /**  @brief Get the internal viewer */
    pcl::visualization::PCLVisualizer::Ptr getViewer() {return viewer_;}
    #endif

    /** @brief This sets everything up for meshing */
    void setInputCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);

    /** @brief Set the primary variable used to control mesh triangulation size */
    void setRho(double val)
    {
      if (val >= M_PI_2 || val <= 0)
      {
        PCL_ERROR("AFront rho must be between 0 and PI/2. Using default value.\n");
        rho_ = AFRONT_DEFAULT_RHO;
      }
      else
      {
        rho_ = val;
      }

      hausdorff_error_ = (1.0 - sqrt((1.0 + 2.0 * cos(rho_)) / 3.0)) * (1.0 / (2.0 * sin(rho_ / 2)));
      updateTriangleTolerances();
    }

    /** @brief Get the primary variable used to control mesh triangulation size */
    double getRho() const {return rho_;}

    /** @brief Set how fast can the mesh grow and shrink. (val < 1.0) */
    void setReduction(double val)
    {
      if (val >= 1 || val <= 0)
      {
        PCL_ERROR("AFront reduction must be between 0 and 1. Using default value.\n");
        reduction_ = AFRONT_DEFAULT_REDUCTION;
      }
      else
      {
        reduction_ = val;
      }
      updateTriangleTolerances();
    }

    /** @brief Get the variable that controls how fast the mesh can grow and shrink. */
    double getReduction() const {return reduction_;}

    /** @brief Set the mls radius used for smoothing */
    void setSearchRadius(double val)
    {
      search_radius_ = val;
    }

    /** @brief Get the mls radius used for smoothing */
    double getSearchRadius() const {return search_radius_;}

    /** @brief Set the mls polynomial order */
    void setPolynomialOrder(const int order)
    {
      if (order < 2)
      {
        PCL_ERROR("AFront polynomial order must be greater than 1. Using default value.\n");
        polynomial_order_ = AFRONT_DEFAULT_POLYNOMIAL_ORDER;
      }
      else
      {
        polynomial_order_ = order;
      }

      int nr_coeff = (polynomial_order_ + 1) * (polynomial_order_ + 2) / 2;
      required_neighbors_ = 5 * nr_coeff;
    }

    /** @brief Get the mls polynomial order */
    int getPolynomialOrder() const {return polynomial_order_;}

    /** @brief Set the boundary angle threshold used to determine if a point is on the boundary of the point cloud. */
    void setBoundaryAngleThreshold(const double angle)
    {
      if (angle <= 0)
      {
        PCL_ERROR("AFront boundary angle threshold must be greater than 0. Using default value.\n");
        boundary_angle_threshold_ = AFRONT_DEFAULT_BOUNDARY_ANGLE_THRESHOLD;
      }
      else
      {
        boundary_angle_threshold_ = angle;
      }
    }

    /** @brief Get the boundary angle threshold used to determine if a point is on the boundary of the point cloud. */
    double getBoundaryAngleThreshold() const {return boundary_angle_threshold_;}


    /** @brief Set the number of threads to use */
    void setNumberOfThreads(const int threads)
    {
      if (threads <= 0)
      {
        PCL_ERROR("AFront number of threads must be greater than 0. Using default value.\n");
        threads_ = AFRONT_DEFAULT_THREADS;
      }
      else
      {
        threads_ = threads;
      }
    }

    /** @brief Get the number of threads to use */
    int getNumberOfThreads() {return threads_;}

    /** @brief Create the first triangle given a starting location. */
    void createFirstTriangle(const int &index);
    void createFirstTriangle(const double &x, const double &y, const double &z);

    /** @brief Get the predicted vertex for the provided front */
    PredictVertexResults predictVertex(const AdvancingFrontData &afront) const;

    /** @brief Check if features of the existing mesh are in close proximity to the proposed new triangle. */
    CloseProximityResults isCloseProximity(const PredictVertexResults &pvr) const;

    /**
     * @brief Check if a line intersects a mesh half edge fence.
     * @param sp Origin point of the line
     * @param ep Terminating point of the line
     * @param fence The half edge index
     * @param fence_height The height of the fence.
     * @param lpr The results of the line plane intersection
     * @return True if line interesects fence, otherwise false
     */
    bool isFenceViolated(const Eigen::Vector3f &sp, const Eigen::Vector3f &ep, const HalfEdgeIndex &fence, const double fence_height, utils::IntersectionLine2PlaneResults &lpr) const;

    /**
     * @brief Check if a line intersects a list of fences.
     * @param vi The vertex index representing the origin of the line.
     * @param p The terminating point of the line
     * @param fences The list of half edge indexes
     * @param closest The closest point in the mesh if one exists
     * @param pvr The predicted vertex results data
     * @return FenceViolationResults
     */
    FenceViolationResults isFencesViolated(const VertexIndex &vi, const Eigen::Vector3f &p, const std::vector<HalfEdgeIndex> &fences, const VertexIndex &closest, const PredictVertexResults &pvr) const;

    /**
     * @brief Check if the proposed triangle interferes with the previous or next half edge.
     *
     * If it does interfere the input variable tri will be modified to represent a tri created
     * by either the previous or next half edge.
     *
     * @param afront The advancing front data
     * @param tri The proposed triangle
     * @param closest The closest point in the mesh if one exists
     * @return True if it does interfere, otherwise false
     */
    bool checkPrevNextHalfEdge(const AdvancingFrontData &afront, TriangleData &tri, VertexIndex &closest) const;

    /** @brief Check if the proposed triangle is to close to the existing mesh. */
    TriangleToCloseResults isTriangleToClose(const PredictVertexResults &pvr) const;

    /** @brief Grow a triangle */
    void grow(const PredictVertexResults &pvr);

    /** @brief Merge triangle with the existing mesh */
    void merge(const TriangleToCloseResults &ttcr);

    /** @brief Perform an ear cut operation */
    void cutEar(const CutEarData &ccer);

    /** @brief Print all of the meshes vertices */
    void printVertices() const;

    /** @brief Print all of the meshes faces */
    void printFaces() const;

    /** @brief Print a given half edges vertices */
    void printEdge(const HalfEdgeIndex &half_edge) const;

    /** @brief Print a given face's information */
    void printFace(const FaceIndex &idx_face) const;

  private:
    /** @brief Used to get the next half edge */
    CutEarData getNextHalfEdge(const FrontData &front) const;

    /** @brief Used to get the previous half edge */
    CutEarData getPrevHalfEdge(const FrontData &front) const;

    /** @brief Add half edge to queue */
    bool addToQueue(const FaceIndex &face);
    void addToQueueHelper(const HalfEdgeIndex &half_edge);

    /** @brief Remove half edge from queue */
    void removeFromQueue(const HalfEdgeIndex &half_edge);
    void removeFromQueue(const HalfEdgeIndex &half_edge1, const HalfEdgeIndex &half_edge2);

    /** @brief Add half edge to boundary */
    void addToBoundary(const HalfEdgeIndex &half_edge);

    /** @brief Remove half edge from boundary */
    void removeFromBoundary(const HalfEdgeIndex &half_edge);
    void removeFromBoundary(const HalfEdgeIndex &half_edge1, const HalfEdgeIndex &half_edge2);

    /**
    * @brief This creates the MLS surface.
    * @return True if successful, otherwise false.
    */
    bool computeGuidanceField();

    /**
    * @brief This calculates useful data about the advancing front used throughout.
    * @param half_edge Advancing half edge index
    * @return Data about the advancing half edge
    */
    AdvancingFrontData getAdvancingFrontData(const HalfEdgeIndex &half_edge) const;

    /**
     * @brief Create face data to be stored with the face. Currently it stores the center
     * of the triangle and the normal of the face.
     */
    MeshTraits::FaceData createFaceData(const TriangleData &tri) const;

    /**
    * @brief Get the dirction to grow for a given half edge
    * @param p A vertex of the half edge
    * @param mp The mid point of the half edge
    * @param fd The face data associated to the opposing half edge
    * @return The grow direction vector
    */
    Eigen::Vector3f getGrowDirection(const Eigen::Vector3f &p, const Eigen::Vector3f &mp, const MeshTraits::FaceData &fd) const;

    /**
     * @brief Get the maximum step required for a given point
     * @param p The point for which to determine the max step
     * @param radius_found The radius at which the criteria was meet.
     * @return The max step required
     */
    double getMaxStep(const Eigen::Vector3f &p, float &radius_found) const;

    /**
    * @brief Calculate triangle information.
    * @param front The advancing front
    * @param p Third point of triangle
    * @return Returns information about the triangle: angles, edge lengths, etc.
    */
    TriangleData getTriangleData(const FrontData &front, const AfrontVertexPointType &p) const;

    /** @brief Update the allowed triangle tolerances. */
    void updateTriangleTolerances()
    {
      vertex_normal_tol_ = (1.0 + AFRONT_CLOSE_PROXIMITY_FACTOR) * (rho_ / reduction_);
      triangle_normal_tol_ = vertex_normal_tol_ / 2.0;
    }

    /** @brief Check if a point is in the grow direction of the front. */
    bool isPointValid(const FrontData &front, const Eigen::Vector3f p) const;

    /** @brief Check if the front is at or near the boundary of the point cloud. */
    bool nearBoundary(const FrontData &front, const int index) const;

    /** @brief This is a direct copy of the pcl isBoundaryPoint function */
    bool isBoundaryPoint(const int index) const;

    /** @brief Generate a plygon mesh that represent the mls polynomial surface. */
    pcl::PolygonMesh getPolynomialSurface(const PredictVertexResults &pvr, const double step) const;

    #ifdef AFRONTDEBUG
    /**
     * @brief keyboardEventOccurred
     * @param event
     * @return
     */
    void keyboardEventOccurred(const pcl::visualization::KeyboardEvent &event, void*);
    #endif

    // User defined data
    pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud_;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr input_cloud_tree_;

    double rho_;           /**< @brief The angle of the osculating circle where a triangle edge should optimally subtend */
    double reduction_;     /**< @brief The allowed percent reduction from triangle to triangle. */
    double search_radius_; /**< @brief The search radius used by mls */
    int polynomial_order_; /**< @brief The degree of the polynomial used by mls */
    int threads_;          /**< @brief The number of threads to be used by mls */

    // Algorithm Data
    double hausdorff_error_;
    double max_edge_length_;          /**< @brief This can be used to calculate the max error fo the reconstruction (max_edge_length_ * hausdorff_error_) */
    int required_neighbors_;          /**< @brief This the required number of neighbors for a given point found during the MLS. */
    double vertex_normal_tol_;        /**< @brief The angle tolerance for vertex normals for a given triangle */
    double triangle_normal_tol_;      /**< @brief The angle tolerance for the triangle normal relative to vertex normals */
    double boundary_angle_threshold_; /**< @brief The boundary angle threshold */

    // Guidance field data
    MLSSampling mls_;
    pcl::PointCloud<afront_meshing::AfrontGuidanceFieldPointType>::Ptr mls_cloud_;
    pcl::search::KdTree<afront_meshing::AfrontGuidanceFieldPointType>::Ptr mls_cloud_tree_;

    // Generated data
    Mesh mesh_; /**< The mesh object for inserting faces/vertices */
    pcl::PointCloud<MeshTraits::VertexData>::Ptr mesh_vertex_data_ptr_;
    pcl::octree::OctreePointCloudSearch<MeshTraits::VertexData>::Ptr mesh_octree_;
    pcl::IndicesPtr mesh_vertex_data_indices_;

    // Algorithm Status Data
    std::deque<HalfEdgeIndex> queue_;
    std::vector<HalfEdgeIndex> boundary_;
    bool finished_;
    bool initialized_;

    // Debug
    #ifdef AFRONTDEBUG
    std::uint64_t counter_;
    mutable std::uint64_t fence_counter_;
    pcl::visualization::PCLVisualizer::Ptr viewer_;
    #endif
  };
} // namespace afront_meshing

#endif // AFRONT_MESHING_H
