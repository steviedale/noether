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
    pcl::PointNormal samplePoint(const pcl::PointXYZ& pt);

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
      typedef pcl::Normal           FaceData;

      typedef boost::false_type     IsManifold;
    };

    typedef pcl::geometry::PolygonMesh <AfrontMeshing::MeshTraits> Mesh;

    typedef Mesh::VertexIndex   VertexIndex;
    typedef Mesh::HalfEdgeIndex HalfEdgeIndex;
    typedef Mesh::FaceIndex     FaceIndex;

    typedef Mesh::VertexIndices   VertexIndices;
    typedef Mesh::HalfEdgeIndices HalfEdgeIndices;
    typedef Mesh::FaceIndices     FaceIndices;

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
     double getRho(){return rho_;}

     void setRadius(double val){r_ = val;}
     double getRadius(){return r_;}

  private:

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

     double getCurvature(int index);

     double getAverageCurvature(std::vector<int>& indices);

     double rho_;

     double r_;


     Mesh mesh_; /**< The mesh object for inserting faces/vertices */

  };


}

#endif // AFRONT_MESHING_H
