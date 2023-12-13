#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"
#include "nav_msgs/msg/path.hpp"
#include "nav_msgs/msg/grid_cells.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "tf2/LinearMath/Quaternion.h"

#include <chrono>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>

class PathPlanner : public rclcpp::Node {
public:
    PathPlanner();

private:
    int range;
    int map_size_x;
    int map_size_y;
    int resolution;
    float float_resolution;
    sensor_msgs::msg::PointCloud2::SharedPtr lane_perception_msg;

    rclcpp::TimerBase::SharedPtr publisher_timer_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr publisher_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;

    void publisher_timer_callback();
    void lane_callback(const sensor_msgs::msg::PointCloud2::SharedPtr lane_perception_msg);
    static bool small_x_value(const std::vector<int>& a, const std::vector<int>& b);
    static bool long_length(const std::vector<int>& a, const std::vector<int>& b);
    void addPose(nav_msgs::msg::Path& path_msg, int x, int y, float theta);

    ///////////////////////////////LANE DEBUG///////////////////////////////
    rclcpp::Publisher<nav_msgs::msg::GridCells>::SharedPtr left_lane_publisher;
    rclcpp::Publisher<nav_msgs::msg::GridCells>::SharedPtr right_lane_publisher;
    void addCell(nav_msgs::msg::GridCells& grid_cells_msg, int x, int y, int z);
    ///////////////////////////////LANE DEBUG///////////////////////////////
};