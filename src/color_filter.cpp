#include <iostream>
#include <thread>
#include <vector>

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/filter_indices.h> // for pcl::removeNaNFromPointCloud
#include <pcl/segmentation/region_growing_rgb.h>
#include "ros/ros.h"
#include "sensor_msgs/PointCloud2.h"
#include <pcl_conversions/pcl_conversions.h>

#include <sstream>
using namespace std::chrono_literals;

using Point = pcl::PointXYZRGB;
using PointCloud = pcl::PointCloud<Point>;
using PointCloudPtr = PointCloud::Ptr;

/*!
* @brief Helper method to convert point cloud messages to a point cloud object.
*/
inline PointCloudPtr ConvertMessageToPointCloud(sensor_msgs::PointCloud2 const &pointCloudData) noexcept
{
  // http://wiki.ros.org/pcl/Overview
  // pcl_conversions is the only supported/valid conversion of pointcloud data
  auto source = pcl::make_shared<PointCloud>();
  pcl::fromROSMsg(pointCloudData, *source);

  // ellipsis performs move operation
  return source;
}

inline sensor_msgs::PointCloud2 ConvertPointCloudToMessage(PointCloud const &pointCloud) noexcept
{
  sensor_msgs::PointCloud2 message;
  pcl::toROSMsg(pointCloud, message);

  return message;
}

void FilterColouredPointCloud(const sensor_msgs::PointCloud2 &msg)
{
  using namespace std::chrono_literals;

  pcl::search::Search<Point>::Ptr tree(new pcl::search::KdTree<Point>);
  PointCloudPtr cloud = ConvertMessageToPointCloud(msg);

  pcl::IndicesPtr indices(new std::vector<int>);
  pcl::removeNaNFromPointCloud(*cloud, *indices);

  pcl::RegionGrowingRGB<Point> regionGrowing;
  regionGrowing.setInputCloud(cloud);
  regionGrowing.setIndices(indices);
  regionGrowing.setSearchMethod(tree);
  regionGrowing.setDistanceThreshold(10);
  regionGrowing.setPointColorThreshold(6);
  regionGrowing.setRegionColorThreshold(5);
  regionGrowing.setMinClusterSize(600);

  std::vector<pcl::PointIndices> clusters;
  regionGrowing.extract(clusters);

  PointCloudPtr colored_cloud = regionGrowing.getColoredCloud();

  pcl::visualization::CloudViewer viewer("Cluster viewer");
  viewer.showCloud(colored_cloud);

  std::this_thread::sleep_for(5s);
}

/**
 * This tutorial demonstrates simple sending of messages over the ROS system.
 */
int main(int argc, char **argv)
{

  ros::init(argc, argv, "color_filter");

  ros::NodeHandle n;

  ros::Publisher chatter_pub = n.advertise<sensor_msgs::PointCloud2>("color_filtered_cloud", 1000);
  ros::Subscriber sub = n.subscribe("/zedm/zed_node/point_cloud/cloud_registered", 1000, &FilterColouredPointCloud);

  ros::Rate loop_rate(10);

  while (ros::ok())
  {
    ros::spinOnce();
    loop_rate.sleep();
  }

  return 0;
}
