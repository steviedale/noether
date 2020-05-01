#ifndef AFRONT_POINT_TYPE_H
#define AFRONT_POINT_TYPE_H
#define PCL_NO_PRECOMPILE

#include <pcl/point_types.h>
#include <pcl/register_point_struct.h>
#include <pcl/impl/instantiate.hpp>
#include <ostream>

namespace afront_meshing
{
  struct EIGEN_ALIGN16 _AfrontGuidanceFieldPointType
  {
    PCL_ADD_POINT4D; // This adds the members x,y,z which can also be accessed using the point (which is float[4])
    PCL_ADD_NORMAL4D; // This adds the member normal[3] which can also be accessed using the point (which is float[4])
    union
    {
      struct
      {
        float curvature;
        float ideal_edge_length;
      };
      float data_c[4];
    };
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };

  struct AfrontGuidanceFieldPointType : public _AfrontGuidanceFieldPointType
  {
    inline AfrontGuidanceFieldPointType (const AfrontGuidanceFieldPointType &p)
    {
      x = p.x; y = p.y; z = p.z; data[3] = 1.0f;
      normal_x = p.normal_x; normal_y = p.normal_y; normal_z = p.normal_z; data_n[3] = 0.0f;
      curvature = p.curvature;
      ideal_edge_length = p.ideal_edge_length;
    }

    inline AfrontGuidanceFieldPointType ()
    {
      x = y = z = 0.0f;
      data[3] = 1.0f;
      normal_x = normal_y = normal_z = data_n[3] = 0.0f;
      curvature = 0.0f;
      ideal_edge_length = 0.0f;
    }

    friend std::ostream& operator << (std::ostream& os, const AfrontGuidanceFieldPointType& p)
    {
      os << p.x << "\t" << p.y << "\t" << p.z;
      return (os);
    }
  };

  struct EIGEN_ALIGN16 _AfrontVertexPointType
  {
    PCL_ADD_POINT4D; // This adds the members x,y,z which can also be accessed using the point (which is float[4])
    PCL_ADD_NORMAL4D; // This adds the member normal[3] which can also be accessed using the point (which is float[4])
    union
    {
      struct
      {
        float curvature;
        float max_step;
        float max_step_search_radius;
      };
      float data_c[4];
    };
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };

  struct AfrontVertexPointType : public _AfrontVertexPointType
  {
    inline AfrontVertexPointType (const AfrontVertexPointType &p)
    {
      x = p.x; y = p.y; z = p.z; data[3] = 1.0f;
      normal_x = p.normal_x; normal_y = p.normal_y; normal_z = p.normal_z; data_n[3] = 0.0f;
      curvature = p.curvature;
      max_step = p.max_step;
      max_step_search_radius = p.max_step_search_radius;
    }

    inline AfrontVertexPointType ()
    {
      x = y = z = 0.0f;
      data[3] = 1.0f;
      normal_x = normal_y = normal_z = data_n[3] = 0.0f;
      curvature = 0.0f;
      max_step = 0.0f;
      max_step_search_radius = 0.0f;
    }

    friend std::ostream& operator << (std::ostream& os, const AfrontVertexPointType& p)
    {
      os << p.x << "\t" << p.y << "\t" << p.z;
      return (os);
    }
  };
}
POINT_CLOUD_REGISTER_POINT_STRUCT (afront_meshing::AfrontVertexPointType,
                                   (float, x, x)
                                   (float, y, y)
                                   (float, z, z)
                                   (float, normal_x, normal_x)
                                   (float, normal_y, normal_y)
                                   (float, normal_z, normal_z)
                                   (float, curvature, curvature)
                                   (float, max_step, max_step)
                                   (float, max_step_search_radius, max_step_search_radius)
)

POINT_CLOUD_REGISTER_POINT_STRUCT (afront_meshing::AfrontGuidanceFieldPointType,
                                   (float, x, x)
                                   (float, y, y)
                                   (float, z, z)
                                   (float, normal_x, normal_x)
                                   (float, normal_y, normal_y)
                                   (float, normal_z, normal_z)
                                   (float, curvature, curvature)
                                   (float, ideal_edge_length, ideal_edge_length)
)

#endif // AFRONT_POINT_TYPE_H
