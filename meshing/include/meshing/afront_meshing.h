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

namespace afront_meshing
{
  //template <typename PointInT, typename PointOutT> void

  class MLSSampling : public pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointNormal>
  {
  public:
    // expose protected function 'performUpsampling' from MLS
    pcl::PointNormal samplePoint(const pcl::PointXYZ& pt) const;
    pcl::PointNormal samplePoint(const pcl::PointNormal& pt) const;

  private:
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

    typedef pcl::geometry::PolygonMesh <pcl::geometry::DefaultMeshTraits< pcl::PointXYZ, int, int, pcl::PointNormal> > Mesh;

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

    struct PredictVertexResults
    {
      HalfEdgeIndex he;   /**< @brief The half edge index from which to grow the triangle */
      double l;           /**< @brief Allowed grow distance */
      VertexIndices vi;   /**< @brief Stores triangle indicies */
      pcl::PointXYZ p[3]; /**< @brief Stores the point information for the triangle */
      pcl::PointXYZ mp;   /**< @brief The half edge mid point */
      Eigen::Vector3d d;  /**< @brief The grow direction */
    };

    struct TriangleToCloseResults
    {
      PredictVertexResults pvr;        /**< @brief The predicted vertex information provided */
      VertexIndex closest;             /**< @brief The index of the closest point to the predicted vertex  */
      double dist;                     /**< @brief The squared distance to the closest vertex */
      bool valid;                      /**< @brief True if not to close otherwise false */
    };

  public:
     void setInputCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);
     bool computeGuidanceField();
     void startMeshing();

     pcl::PolygonMesh getMesh();

     /**
      * @brief setRho The primary variable used to control mesh triangulation size
      * @param val
      */
     void setRho(double val){rho_ = val;}
     double getRho() const {return rho_;}

     void setRadius(double val){r_ = val;}
     double getRadius() const {return r_;}

     VertexIndex createFirstTriangle(const double &x, const double &y, const double &z);

     PredictVertexResults predictVertex(const HalfEdgeIndex &half_edge) const;

     TriangleToCloseResults isTriangleToClose(const PredictVertexResults &pvr) const;

     VertexIndex grow(const PredictVertexResults &pvr);

     VertexIndex merge(const TriangleToCloseResults &tc);

     /**
      * @brief Perform an ear cut operation if possible
      * @param half_edge1 Front half edge index
      * @param half_edge2 Front half edge index
      * @return True if a ear cutting operation was possible
      */
     bool cutEar(const HalfEdgeIndex &half_edge1, const HalfEdgeIndex &half_edge2);

     // Some output functions
     void printVertices (const Mesh& mesh)
     {
       std::cout << "Vertices:\n   ";
       for (unsigned int i=0; i<mesh.sizeVertices (); ++i)
       {
         std::cout << mesh.getVertexDataCloud () [i] << " ";
       }
       std::cout << std::endl;
     }

     void printEdge (const Mesh& mesh, const HalfEdgeIndex& idx_he)
     {
       std::cout << "  "
                 << mesh.getVertexDataCloud () [mesh.getOriginatingVertexIndex (idx_he).get ()]
                 << " "
                 << mesh.getVertexDataCloud () [mesh.getTerminatingVertexIndex (idx_he).get ()]
                 << std::endl;
     }

     void printFace (const Mesh& mesh, const FaceIndex& idx_face)
     {
       // Circulate around all vertices in the face
       VAFC       circ     = mesh.getVertexAroundFaceCirculator (idx_face);
       const VAFC circ_end = circ;
       std::cout << "  ";
       do
       {
         std::cout << mesh.getVertexDataCloud () [circ.getTargetIndex ().get ()] << " ";
       } while (++circ != circ_end);
       std::cout << std::endl;
     }

     void printFaces (const Mesh& mesh)
     {
       std::cout << "Faces:\n";
       for (unsigned int i=0; i<mesh.sizeFaces (); ++i)
       {
         printFace (mesh, FaceIndex (i));
       }
     }

  private:
     MeshTraits::FaceData createFaceData(const pcl::PointXYZ p1, const pcl::PointXYZ p2, const pcl::PointXYZ p3);

     /**
      * @brief Get the mid point of a half edge given it's verticies
      * @param p1 Vertex of half edge
      * @param p2 Vectex of half edge
      * @return The mid point of the half edge
      */
     pcl::PointXYZ getMidPoint(const pcl::PointXYZ &p1, const pcl::PointXYZ &p2) const;

     /**
      * @brief Get the dirction to grow for a given half edge
      * @param p A vertex of the half edge
      * @param mp The mid point of the half edge
      * @param fd The face data associated to the opposing half edge
      * @return The grow direction vector
      */
     Eigen::Vector3d getGrowDirection(const pcl::PointXYZ &p, const pcl::PointXYZ &mp, const MeshTraits::FaceData &fd) const;

     /**
      * @brief Get the allowed grow distance
      * @param mp The mid point of the half edge
      * @return The allowed grow distance
      */
     double getGrowDistance(const pcl::PointXYZ &mp) const;

     /**
      * @brief Get the predicted vertex for the new triangle
      * @param mp The mid point of the half edge from which to grow the triangle
      * @param d The direction to grow the trianlge
      * @param l The allowed grow distance
      * @return The predicted vertex.
      */
     pcl::PointXYZ getPredictedVertex(const pcl::PointXYZ &mp, const Eigen::Vector3d &d, const double &l) const;

     pcl::search::KdTree<pcl::PointXYZ>::Ptr input_cloud_tree_;

     pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud_;
     pcl::PointCloud<pcl::PointNormal>::Ptr cloud_normals_;

     MLSSampling mls_;
     /**
      * @brief curvatures_ each point contains: principal_curvature[3], pc1, and pc2
      * principal_curvature contains the eigenvector for the minimum eigen value
      * pc1 = eigenvalues_ [2] * indices_size;
      * pc2 = eigenvalues_ [1] * indices_size;
      * curvature change calculation = eig_val_min/(sum(eig_vals))
      * pc1 and pc2 are not necessarily the same as k1 and k2 (curvature typically found in literature)
      */
     pcl::PointCloud<pcl::PrincipalCurvatures>::Ptr curvatures_;

     double getCurvature(const int index) const;

     double getAverageCurvature(const std::vector<int> &indices) const;

     double rho_;

     double r_;

     Mesh mesh_; /**< The mesh object for inserting faces/vertices */
     pcl::PointCloud<MeshTraits::VertexData>::Ptr mesh_vertex_data_;
     pcl::search::KdTree<MeshTraits::VertexData>::Ptr mesh_tree_;
  };


}

#endif // AFRONT_MESHING_H
