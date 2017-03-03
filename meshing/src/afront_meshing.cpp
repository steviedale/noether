#include <pcl/geometry/mesh_conversion.h>
#include <meshing/afront_meshing.h>

//template class PCL_EXPORTS afront_meshing::MLSSampling<pcl::PointXYZ, pcl::PointNormal>;
namespace afront_meshing
{
  pcl::PointNormal MLSSampling::samplePoint(const pcl::PointXYZ& pt)
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    cloud->push_back(pt);

    setUpsamplingMethod(pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointNormal>::DISTINCT_CLOUD);
    setDistinctCloud(cloud);

    pcl::PointCloud<pcl::PointNormal> cloud_out;
    performUpsampling(cloud_out);

    return cloud_out.points[0];
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
    mls_.setComputeNormals (true);
    mls_.setInputCloud (input_cloud_);
    mls_.setPolynomialFit (true);
    mls_.setSearchMethod (input_cloud_tree_);
    mls_.setSearchRadius (r_);
    mls_.process (*cloud_normals_);

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
    pcl::PointXYZ pt1 = input_cloud_->points[start_pt];
    double k1 = getCurvature(start_pt);
    double l1 = rho_ / k1;

    std::cout << "k1 and L1 value " << k1 << " " << l1 <<  "\n";

    std::vector<int> K(1);
    std::vector<float> K_dist(1);

    // search for the nearest neighbor
    input_cloud_tree_->nearestKSearch(pt1, 1, K, K_dist);

    // use l1 and nearest neighbor to extend edge
    pcl::PointXYZ pt2 = input_cloud_->points[K[0]];
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
    pcl::PointNormal pt2_norm = mls_.samplePoint(pt2);
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
    mesh_.addVertex(pt1);
    mesh_.addVertex(pt2);
    mesh_.addVertex(pt3);

    // Add new face
    VertexIndices indices;
    VertexIndex id;
    id.set(0);

    indices.push_back(id);
    id.set(1);
    indices.push_back(id);
    id.set(2);
    indices.push_back(id);
    mesh_.addFace(indices);

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
    curv /= indices.size();
  }

}

