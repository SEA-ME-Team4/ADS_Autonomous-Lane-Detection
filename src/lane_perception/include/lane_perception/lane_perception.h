#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"

#include <string>
#include <cmath>

class LanePerception : public rclcpp::Node {
public:
    LanePerception();

private:
    sensor_msgs::msg::LaserScan::SharedPtr scan_msg;

    rclcpp::TimerBase::SharedPtr publisher_timer_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr subscription_;

    void publisher_timer_callback();
    void scan_callback(const sensor_msgs::msg::LaserScan::SharedPtr scan_msg);
};