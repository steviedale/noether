#ifndef AFRONT_MESHING_H
#define AFRONT_MESHING_H

#include <pcl/common/common.h>
#include <pcl/point_traits.h>
#include <pcl/point_types.h>
#include <pcl/PolygonMesh.h>

#include <pcl/surface/mls.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/principal_curvatures.h>

#include <pcl/geometry/polygon_mesh.h>
#include <pcl/PolygonMesh.h>

#include <pcl/visualization/pcl_visualizer.h>

#include <deque>

namespace afront_meshing
{
  //template <typename PointInT, typename PointOutT> void

  class MLSSampling : public pcl::MovingLeastSquaresOMP<pcl::PointXYZ, pcl::PointNormal>
  {
  public:
    struct SamplePointResults
    {
      pcl::PointNormal point;   /**< @brief The point projected on to the MLS surface */
      int              closest; /**< @brief The closest point index on the MLS surface to the project point */
      double           dist;    /**< @brief The distance squared between point and closest */
    };

    void process(pcl::PointCloud<pcl::PointNormal> &output);

    // expose protected function 'performUpsampling' from MLS
    MLSSampling::SamplePointResults samplePoint(const pcl::PointXYZ& pt) const;
    MLSSampling::SamplePointResults samplePoint(const pcl::PointNormal& pt) const;

  private:

    Eigen::Vector2f calculateCurvature(const float &u, const float &v, const MLSResult &mls_result) const;
    Eigen::Vector2f calculateCurvature(const int &index) const;

    pcl::PointCloud<pcl::PointXYZ> cloud_;

  };


  class AfrontMeshing
  {
    struct MeshTraits
    {
      typedef pcl::PointXYZ         VertexData;
      typedef int                   HalfEdgeData;
      typedef int                   EdgeData;
      typedef pcl::PointNormal      FaceData;

      typedef boost::false_type     IsManifold;
    };

    typedef pcl::geometry::PolygonMesh <MeshTraits > Mesh;

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
      double ideal;              /**< @brief The ideal edge length */
      double max_curv;           /**< @brief The maximum curvature found with search radius */
      std::vector<int> k;        /**< @brief The resultant indices of the neighboring points */
      std::vector<float> k_dist; /**< @brief The resultant squared distances to the neighboring points */
      bool valid;                /**< @brief True if successful, otherwise false */
    };

    struct TriangleData
    {
      double A;             /**< @brief The length for the first half edge (p1->p2) */
      double B;             /**< @brief The length for the second half edge (p2->p3) */
      double C;             /**< @brief The length for the remaining side of the triangle (p3->p1) */
      double a;             /**< @brief The angle BC */
      double b;             /**< @brief The angle AC */
      double c;             /**< @brief The anble AB */
      double aspect_ratio;  /**< @brief The quality of the triangle (1.0 is the best) */
      Eigen::Vector3f p[3]; /**< @brief Stores the point information for the triangle */
      bool valid;           /**< @brief Indicates if the triangle is valid. Both edges must point in the same direction as the grow direction. */

      void print(const std::string description = "") const
      {
        if (description == "")
          std::printf("Triangle Data:\n");
        else
          std::printf("Triangle Data (%s):\n", description.c_str());

        std::printf("\t A: %-10.4f B: %-10.4f C: %-10.4f\n", A, B, C);
        std::printf("\t a: %-10.4f b: %-10.4f c: %-10.4f\n", a, b, c);
        std::printf("\t Aspect Ratio: %-10.4f\n", aspect_ratio);
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
    };

    struct PredictVertexResults
    {
      FrontData front;                        /**< @brief Advancing front data */
      GrowDistanceResults gdr;                /**< @brief Allowed grow distance */
      TriangleData tri;                       /**< @brief The proposed triangle data */
      MLSSampling::SamplePointResults pv;     /**< @brief The predicted point projected on the mls surface */
      Eigen::Vector2d k;                      /**< @brief The principal curvature using the polynomial */
      bool at_boundary;                        /**< @brief The predicted vertex is near the boundry of the point cloud */
    };

    struct DistPointToHalfEdgeResults
    {
      HalfEdgeIndex he; /**< @brief The half edge index to check distance against. */
      double line;      /**< @brief The minimum distance to the line segment. */
      double start;     /**< @brief The distance to the line segment start point. */
      double end;       /**< @brief The distance to the line segment end point. */
    };

    struct CanCutEarResult
    {
      HalfEdgeIndex primary;   /**< @brief The advancing front half edge */
      HalfEdgeIndex secondary; /**< @brief The Secondary half edge triangle (Previouse or Next) */
      VertexIndices vi;        /**< @brief The vertex indicies of the potential triangle */
      TriangleData tri;        /**< @brief The Triangle information */
      bool valid;              /**< @brief Whether the triangle meets the criteria */
    };

    struct CanCutEarResults
    {
      CanCutEarResults() {valid = nullptr;}

      FrontData front;        /**< @brief Advancing front data */
      CanCutEarResult prev;   /**< @brief The results using the previous half edge */
      CanCutEarResult next;   /**< @brief The results using the next half edge */
      CanCutEarResult *valid; /**< @brief The valid ear cutting option available */
    };

    enum TriangleToCloseTypes
    {
      None = 0,             /**< @brief There is no violation */
      PrevHalfEdge = 1,     /**< @brief The new trianble interfereces with the previous half edge. */
      NextHalfEdge = 2,     /**< @brief The new trianble interfereces with the next half edge. */
      FenceViolation = 3,   /**< @brief The new triangle violates another half edges fence. */
      CloseProximity = 4,   /**< @brief The new triangle is in close proximity to another half edge. */
    };

    struct TriangleToCloseResults
    {
      TriangleToCloseTypes type;       /**< @brief The type of violation. */
      PredictVertexResults pvr;        /**< @brief The predicted vertex information provided */
      CanCutEarResults ccer;           /**< @brief The can cut ear results */
      DistPointToHalfEdgeResults dist; /**< @brief This stores closest distance information */
    };

  public:
    /** @brief AfrontMeshing */
    AfrontMeshing();
    ~AfrontMeshing() {}

    /** @brief This sets everything up for meshing */
    bool initMesher(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);

    /** @brief This will mesh the point cloud passed to the initMesher funciton */
    void generateMesh();

    /** @brief Attempted to advance the mesh by adding one triangle */
    void stepMesh();

    void setViewer(boost::shared_ptr<pcl::visualization::PCLVisualizer> &viewer) {viewer_ = viewer;}

    /** @brief This will force every point projected onto the mls surface to snap to existing data. */
    void enableSnap(const bool &enable) {snap_ = enable;}

    /** @brief Indicates if it has finished meshing the surface */
    bool isFinished() {return finished_;}

    /** @brief Get the current polygon mesh */
    pcl::PolygonMesh getMesh() const;

    /** @brief Get the normals */
    pcl::PointCloud<pcl::Normal>::ConstPtr getNormals() const;

    /** @brief Set the primary variable used to control mesh triangulation size */
    void setRho(double val){rho_ = val;}

    /** @brief Get the primary variable used to control mesh triangulation size */
    double getRho() const {return rho_;}

    /** @brief Set how fast can the mesh grow and shrink. (val > 1.0) */
    void setTriangleQuality(double val) {reduction_ = val;}

    /** @brief Get the variable that controls howw fast the mesh can grow and shrink. */
    double getTriangleQuality() const {return reduction_;}

    /** @brief Set the mls radius used for smoothing */
    void setRadius(double val){r_ = val;}

    /** @brief Get the mls radius used for smoothing */
    double getRadius() const {return r_;}

    /** @brief Create the first triangle given a starting location. */
    void createFirstTriangle(const int &index);
    void createFirstTriangle(const double &x, const double &y, const double &z);

    CanCutEarResults canCutEar(const FrontData &front) const;

    PredictVertexResults predictVertex(const FrontData &front) const;

    TriangleToCloseResults isTriangleToClose(const CanCutEarResults &ccer, const PredictVertexResults &pvr) const;

    void grow(const CanCutEarResults &ccer, const PredictVertexResults &pvr);

    void topologyEvent(const TriangleToCloseResults &ttcr);
    void merge(const TriangleToCloseResults &ttcr);

    /** @brief Perform an ear cut operation */
    void cutEar(const CanCutEarResult &data);

    /** @brief Print all of the meshes vertices */
    void printVertices() const;

    /** @brief Print all of the meshes faces */
    void printFaces() const;

    /** @brief Print a given half edges vertices */
    void printEdge(const HalfEdgeIndex &half_edge) const;

    /** @brief Print a given face's information */
    void printFace(const FaceIndex &idx_face) const;

    /** @brief Convert Eigen Vector3f to PCL PointXYZ */
    pcl::PointXYZ convertEigenToPCL(const Eigen::Vector3f &p) const;

  private:

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
     FrontData getAdvancingFrontData(const HalfEdgeIndex &half_edge) const;

     MeshTraits::FaceData createFaceData(const Eigen::Vector3f &p1, const Eigen::Vector3f &p2, const Eigen::Vector3f &p3) const;

     /**
      * @brief Get the mid point of a half edge given it's verticies
      * @param p1 Vertex of half edge
      * @param p2 Vectex of half edge
      * @return The mid point of the half edge
      */
     Eigen::Vector3f getMidPoint(const Eigen::Vector3f &p1, const  Eigen::Vector3f &p2) const;

     /**
      * @brief Get the length of a half edge given it's verticies
      * @param p1 Vertex of half edge
      * @param p2 Vectex of half edge
      * @return The lenght of the half edge
      */
     double getEdgeLength(const Eigen::Vector3f &p1, const Eigen::Vector3f &p2) const;

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
      * @param min_length The minium edge length attached to half edge
      * @param max_length The maximum edge length attached to half edge
      * @return The allowed grow distance
      */
     GrowDistanceResults getGrowDistance(const Eigen::Vector3f &mp, const double &edge_length, const double &min_length, const double &max_length) const;

     /**
      * @brief Get the predicted vertex for the new triangle
      * @param mp The mid point of the half edge from which to grow the triangle
      * @param d The direction to grow the trianlge
      * @param l The allowed grow distance
      * @return The predicted vertex data.
      */
     MLSSampling::SamplePointResults getPredictedVertex(const Eigen::Vector3f &mp, const Eigen::Vector3f &d, const double &l) const;

     /**
      * @brief Gets the Minimum and Maximum edge attached to half edge
      * @param half_edge The half edge from which to grow the triangle
      * @return [min, max] edge length
      */
     Eigen::Vector2d getMinMaxEdgeLength(const VertexIndex &v1, const VertexIndex &v2) const;

     /** @brief This will calculate the max curvature at each point the mls surface. */
     void createGuidanceField();

     /** @brief Update the Kd Tree of the mesh vertices */
     void updateKdTree();

     /** @brief Get the next half edge connected to the provided half edge. */
     HalfEdgeIndex getNextHalfEdge(const HalfEdgeIndex &half_edge) const {return mesh_.getNextHalfEdgeIndex(half_edge);}

     /** @brief Get the previous half edge connected to the provided half edge. */
     HalfEdgeIndex getPrevHalfEdge(const HalfEdgeIndex &half_edge) const {return mesh_.getPrevHalfEdgeIndex(half_edge);}

     /** @brief Get the curvature provided an index. */
     float getCurvature(const int &index) const;

     /** @brief Find the maximum curvature given a set of indicies. */
     float getMaxCurvature(const std::vector<int> &indices) const;

     /** @brief Calculate the distance between a point and a half edge. */
     DistPointToHalfEdgeResults distPointToHalfEdge(const Eigen::Vector3f p, const HalfEdgeIndex &half_edge) const;

     /**
      * @brief Calculate triangle information.
      * @param front The advancing front
      * @param p3 First point of triangle
      * @return Returns information about the triangle: angles, edge lengths, etc.
      */
     TriangleData getTriangleData(const FrontData &front, const Eigen::Vector3f p3) const;

     /**
      * @brief Check if a line segment intersects a half edge fence.
      * @param p1 Start point for line segment
      * @param p2 End point for line segment
      * @param half_edge Half edge for which to check for fence violation.
      * @return False if the line segment intersects the half edge fence, otherwise True
      */
     bool checkFence(const Eigen::Vector3f p1, const Eigen::Vector3f p2, const HalfEdgeIndex &half_edge) const;


     pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud_;
     pcl::search::KdTree<pcl::PointXYZ>::Ptr input_cloud_tree_;

     MLSSampling mls_;
     pcl::PointCloud<pcl::PointNormal>::Ptr mls_cloud_;
     pcl::search::KdTree<pcl::PointNormal>::Ptr mls_cloud_tree_;

     double rho_;

     double reduction_;

     double r_;

     Mesh mesh_; /**< The mesh object for inserting faces/vertices */
     pcl::PointCloud<MeshTraits::VertexData> &mesh_vertex_data_;
     pcl::PointCloud<MeshTraits::VertexData>::Ptr mesh_vertex_data_copy_;
     pcl::search::KdTree<MeshTraits::VertexData>::Ptr mesh_tree_;

     std::deque<HalfEdgeIndex> queue_;
     std::vector<HalfEdgeIndex> boundary_;
     bool snap_;
     bool finished_;

     std::uint64_t counter_;
     boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer_;
  };


}

#endif // AFRONT_MESHING_H
