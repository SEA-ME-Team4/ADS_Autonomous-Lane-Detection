#include "lane_perception.h"
#include <string>
#include <math.h>

LanePerception::LanePerception() : rclcpp::Node("lane_percetpion") {
    publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/perception/lane", 10);
    subscription_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
    "/scan", 10, std::bind(&LanePerception::scan_callback, this,  std::placeholders::_1));
    seq = 1;
}


void LanePerception::scan_callback(const sensor_msgs::msg::LaserScan::SharedPtr scan_msg) const {
    // Left Front Right
    // 1079  540    0
    float range;
    int perception_distance = 5;
    float degree;

    std::vector<float> x_list;
    std::vector<float> y_list;

    // From -40 To +40
    for (int index = 380; index < 700; index++) {
        range = scan_msg->ranges[index];
        if (round(range)>perception_distance) {
            continue;
        }
        degree = ((index-180)/4.0) * (3.14159/180);
        // +x = front
        // +y = left 
        float x = std::sin(degree) * range;
        float y = -std::cos(degree) * range;
        x_list.push_back(x);
        y_list.push_back(y);
    }

    // Generate Message
    sensor_msgs::msg::PointCloud2 lane_perception_msg;
    sensor_msgs::PointCloud2Modifier modifier(lane_perception_msg);

    modifier.setPointCloud2Fields(3,
    "x", 1, sensor_msgs::msg::PointField::FLOAT32,
    "y", 1, sensor_msgs::msg::PointField::FLOAT32,
    "z", 1, sensor_msgs::msg::PointField::FLOAT32);

    lane_perception_msg.header = std_msgs::msg::Header();
    lane_perception_msg.header.stamp = rclcpp::Clock().now();
    lane_perception_msg.header.frame_id = "ego_racecar/laser";

    lane_perception_msg.height = 1;
    lane_perception_msg.width = x_list.size();
    lane_perception_msg.is_dense = true;

    lane_perception_msg.is_bigendian = false;
    lane_perception_msg.point_step = 12;
    lane_perception_msg.row_step = lane_perception_msg.point_step * x_list.size();
    lane_perception_msg.data.resize(lane_perception_msg.row_step);

    sensor_msgs::PointCloud2Iterator<float> iter_x(lane_perception_msg, "x");
    sensor_msgs::PointCloud2Iterator<float> iter_y(lane_perception_msg, "y");
    sensor_msgs::PointCloud2Iterator<float> iter_z(lane_perception_msg, "z");

    for (size_t i = 0; i < x_list.size(); ++i) {
        *iter_x = x_list[i];
        *iter_y = y_list[i];
        *iter_z = 0;
        ++iter_x;
        ++iter_y;
        ++iter_z;
    }
    
    publisher_->publish(lane_perception_msg);
}