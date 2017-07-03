#ifndef AFRONT_MESHING_H
#define AFRONT_MESHING_H

#include <pcl/geometry/polygon_mesh.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <deque>

#include <meshing/mls_sampling.h>
#include <meshing/afront_utils.h>
#define AFRONTDEBUG
namespace afront_meshing
{
  class AfrontMeshing
  {
    struct MeshTraits
    {
      typedef pcl::PointNormal      VertexData;
      typedef int                   HalfEdgeData;
      typedef int                   EdgeData;
      typedef pcl::PointNormal      FaceData;

      typedef boost::false_type     IsManifold;
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

    struct GrowDistanceResults
    {
      double l;                  /**< @brief Allowed grow distance perpendicular to half edge */
      double estimated;          /**< @brief The calculated edge length */
      std::vector<int> sample;   /**< @brief Store a subset of search indicies used for boundary detection */
    };

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
      HalfEdgeIndex he;        /**< @brief The half edge index from which to grow the triangle */
      double length;           /**< @brief The half edge length */
      Eigen::Vector3f mp;      /**< @brief The half edge mid point */
      Eigen::Vector3f d;       /**< @brief The half edge grow direction */
      VertexIndex vi[2];       /**< @brief The half edge vertex indicies */
      Eigen::Vector3f p[2];    /**< @brief The half edge points (Origninating, Terminating) */
      Eigen::Vector3f n[2];    /**< @brief The half edge point normals (Origninating, Terminating) */
    };

    struct CutEarData
    {
      enum CutEarDataTypes
      {
        PrevHalfEdge = 0,     /**< @brief Can cut ear results with previous half edge. */
        NextHalfEdge = 1,     /**< @brief Can cut ear results with next half edge. */
      };

      CutEarDataTypes type;    /**< @brief Identifies whether generated using previous or next half edge */
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
      GrowDistanceResults gdr;            /**< @brief Allowed grow distance */
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
      VertexIndex closest;
      TriangleData tri;                         /**< @brief The Triangle information */
      bool found;
    };

  public:
    /** @brief AfrontMeshing Constructor */
    AfrontMeshing();

    /** @brief AfrontMeshing Destructor */
    ~AfrontMeshing() {}

    /** @brief This sets everything up for meshing */
    bool initMesher(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);

    /** @brief This will mesh the point cloud passed to the initMesher funciton */
    void generateMesh();

    /** @brief Advance the mesh by adding one triangle */
    void stepMesh();

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

    /** @brief Set the primary variable used to control mesh triangulation size */
    void setRho(double val)
    {
      rho_ = val;
      hausdorff_error_ = (1.0 - sqrt((1.0 + 2.0 * cos(rho_)) / 3.0)) * (1.0 / (2.0 * sin(rho_ / 2)));
    }

    /** @brief Get the primary variable used to control mesh triangulation size */
    double getRho() const {return rho_;}

    /** @brief Set how fast can the mesh grow and shrink. (val < 1.0) */
    void setReduction(double val) {(val >= 1 || val <= 0) ? reduction_ = 0.8 : reduction_ = val;}

    /** @brief Get the variable that controls how fast the mesh can grow and shrink. */
    double getReduction() const {return reduction_;}

    /** @brief Set the mls radius used for smoothing */
    void setRadius(double val){r_ = val;}

    /** @brief Get the mls radius used for smoothing */
    double getRadius() const {return r_;}

    /** @brief Set the number of threads to use */
    void setNumberOfThreads(const int threads) {threads_ = threads_;}

    /** @brief Get the number of threads to use */
    int getNumberOfThreads() {return threads_;}

    /** @brief Create the first triangle given a starting location. */
    void createFirstTriangle(const int &index);
    void createFirstTriangle(const double &x, const double &y, const double &z);

    /** @brief Get the predicted vertex for the provided front */
    PredictVertexResults predictVertex(const AdvancingFrontData &afront) const;

    CloseProximityResults isCloseProximity(const PredictVertexResults &pvr) const;

    FenceViolationResults isFenceViolated(const VertexIndex &vi, const Eigen::Vector3f &p, const std::vector<HalfEdgeIndex> &fences, const VertexIndex &closest, const PredictVertexResults &pvr) const;

    bool checkPrevNextHalfEdge(const AdvancingFrontData &afront, TriangleData &tri, VertexIndex &vi) const;

    /** @brief Check if the proposed triangle is to close to the existing mesh. */
    TriangleToCloseResults isTriangleToClose(const PredictVertexResults &pvr) const;

    /** @brief Grow a triangle */
    void grow(const PredictVertexResults &pvr);

    /** @brief Merge triangle with the existing mesh */
    void merge(const TriangleToCloseResults &ttcr);

    /** @brief Perform a topology event. This may modify the existing mesh to create quality triangles */
//    void topologyEvent(const TriangleToCloseResults &ttcr);

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
    * @brief Get the allowed grow distance
    * @param mp The mid point of the half edge
    * @param edge_length The half edge length
    * @return The allowed grow distance
    */
    GrowDistanceResults getGrowDistance(const Eigen::Vector3f &mp, const double &edge_length) const;

    /**
     * @brief Get the maximum step required for a given point
     * @param p The point for which to determine the max step
     * @return The max step required
     */
    double getMaxStep(const Eigen::Vector3f &p) const;

    /**
    * @brief Get the predicted vertex for the new triangle
    * @param mp The mid point of the half edge from which to grow the triangle
    * @param d The direction to grow the trianlge
    * @param l The allowed grow distance
    * @return The predicted vertex data.
    */
    MLSSampling::SamplePointResults getPredictedVertex(const Eigen::Vector3f &mp, const Eigen::Vector3f &d, const double &l) const;


    /** @brief Update the Kd Tree of the mesh vertices */
    void updateKdTree();

    /** @brief Get the curvature provided an index. */
    float getCurvature(const int &index) const;

    /** @brief Calculate the distance between a point and a half edge. */
    utils::DistLine2LineResults distLineToHalfEdge(const Eigen::Vector3f &p1, const Eigen::Vector3f &p2, const HalfEdgeIndex &half_edge) const;

    /**
    * @brief Calculate triangle information.
    * @param front The advancing front
    * @param p Third point of triangle
    * @return Returns information about the triangle: angles, edge lengths, etc.
    */
    TriangleData getTriangleData(const FrontData &front, const pcl::PointNormal p) const;

    /** @brief Check if a point is in the grow direction of the front. */
    bool isPointValid(const FrontData &front, const Eigen::Vector3f p) const;

    /** @brief Check if the front is at or near the boundary of the point cloud. */
    bool nearBoundary(const FrontData &front, const int index) const;

    /** @brief This is a direct copy of the pcl isBoundaryPoint function */
    bool isBoundaryPoint(const int index) const;

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

    double rho_;
    double reduction_;
    double r_;

    // Algorithm Data
    double hausdorff_error_;
    double max_edge_length_;          /**< @brief This can be used to calculate the max error fo the reconstruction (max_edge_length_ * hausdorff_error_) */
    int required_neighbors_;          /**< @brief This the required number of neighbors for a given point found during the MLS. */
    double boundary_angle_threshold_; /**< @brief The boundary angle threshold */

    // Guidance field data
    int mls_order_;
    MLSSampling mls_;
    pcl::PointCloud<pcl::PointNormal>::Ptr mls_cloud_;
    pcl::search::KdTree<pcl::PointNormal>::Ptr mls_cloud_tree_;
    int threads_;

    // Generated data
    Mesh mesh_; /**< The mesh object for inserting faces/vertices */
    pcl::PointCloud<MeshTraits::VertexData> &mesh_vertex_data_;
    pcl::PointCloud<MeshTraits::VertexData>::Ptr mesh_vertex_data_copy_;
    pcl::search::KdTree<MeshTraits::VertexData>::Ptr mesh_tree_;

    // Algorithm Status Data
    std::deque<HalfEdgeIndex> queue_;
    std::vector<HalfEdgeIndex> boundary_;
    bool finished_;

    // Debug
    #ifdef AFRONTDEBUG
    std::uint64_t counter_;
    mutable std::uint64_t fence_counter_;
    pcl::visualization::PCLVisualizer::Ptr viewer_;
    #endif
  };
} // namespace afront_meshing

#endif // AFRONT_MESHING_H
