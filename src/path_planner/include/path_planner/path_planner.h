#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"
#include "nav_msgs/msg/path.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
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

    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr publisher_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
    rclcpp::TimerBase::SharedPtr publisher_timer_;

    void publisher_timer_callback();
    void scan_callback(const sensor_msgs::msg::PointCloud2::SharedPtr lane_perception_msg);
    static bool small_x_value(const std::vector<int>& a, const std::vector<int>& b);
    static bool long_length(const std::vector<int>& a, const std::vector<int>& b);
    void addPose(nav_msgs::msg::Path& path_msg, int x, int y, float theta);

    ///////////////////////////////LANE DEBUG///////////////////////////////
    ///////////////////////////////LANE DEBUG///////////////////////////////
    ///////////////////////////////LANE DEBUG///////////////////////////////
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr debug_publisher_1;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr debug_publisher_2;
    ///////////////////////////////LANE DEBUG///////////////////////////////
    ///////////////////////////////LANE DEBUG///////////////////////////////
    ///////////////////////////////LANE DEBUG///////////////////////////////
};