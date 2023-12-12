#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/path.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"
#include "tf2/LinearMath/Quaternion.h"
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/convert.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <string>
#include <cmath>

class VehicleController : public rclcpp::Node {
public:
    VehicleController();

private:
    float pursuit_x;
    float pursuit_y;
    float test_pursuit_y;

    nav_msgs::msg::Path::SharedPtr path_msg;
    nav_msgs::msg::Odometry::SharedPtr odom_msg;

    rclcpp::TimerBase::SharedPtr publisher_timer_;
    rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr publisher_;
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr path_subscription_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_subscription_;

    void publisher_timer_callback();
    void path_callback(const nav_msgs::msg::Path::SharedPtr path_msg);
    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr odom_msg);
    float angle_difference(float x1, float y1, float x2, float y2);
};