#include "vehicle_controller.h"

VehicleController::VehicleController() : rclcpp::Node("vehicle_controller") {
    pursuit_x = 0;
    pursuit_y = 0;

    publisher_ = this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>("/drive", 10);
    path_subscription_ = this->create_subscription<nav_msgs::msg::Path>(
    "/planner/path", 10, std::bind(&VehicleController::path_callback, this,  std::placeholders::_1));
    odom_subscription_ = this->create_subscription<nav_msgs::msg::Odometry>(
    "/ego_racecar/odom", 10, std::bind(&VehicleController::odom_callback, this,  std::placeholders::_1));
    publisher_timer_ = this->create_wall_timer(
        std::chrono::milliseconds(10),
        std::bind(&VehicleController::publisher_timer_callback, this)
    );
}

void VehicleController::path_callback(const nav_msgs::msg::Path::SharedPtr path_msg) {
    this->path_msg = path_msg;

    // Save First Point of Path
    // Todo : Localization => Give Absolute Coordinate of Path
    // pursuit_x = path_msg->poses[1].pose.position.x + odom_msg->pose.pose.position.x;
    // pursuit_y = path_msg->poses[1].pose.position.y + odom_msg->pose.pose.position.y;

    // Local Path Plannings
    float test_sum_y = 0;
    for (int i = 0; i < static_cast<int>(path_msg->poses.size()); ++i) {
        test_sum_y += path_msg->poses[i].pose.position.y;
    }
    test_pursuit_y = test_sum_y/static_cast<int>(path_msg->poses.size());
}

void VehicleController::odom_callback(const nav_msgs::msg::Odometry::SharedPtr odom_msg) {
    this->odom_msg = odom_msg;
}

void VehicleController::publisher_timer_callback() {
    if (!path_msg || !odom_msg) {
        return;
    }

    // Current Status
    float vehicle_x = odom_msg->pose.pose.position.x;
    float vehicle_y = odom_msg->pose.pose.position.y;

    // tf2::Quaternion vehicle_quat = odom_msg->pose.pose.orientation;
    tf2::Quaternion vehicle_quat;
    tf2::fromMsg(odom_msg->pose.pose.orientation, vehicle_quat);
    double roll;
    double pitch;
    double vehicle_head_angle;
    tf2::Matrix3x3 matrix(vehicle_quat);
    matrix.getRPY(roll, pitch, vehicle_head_angle);

    // Angle Difference
    float desire_angle = this->angle_difference(vehicle_x, vehicle_y, pursuit_x, pursuit_y);
    float control_angle = (desire_angle - vehicle_head_angle)*2;
    float control_speed = 3;
    control_angle = (test_pursuit_y)/2;

    control_speed -= 0.5 * control_speed * (1-std::pow(control_angle,2));

    RCLCPP_INFO(this->get_logger(), "Iter");
    RCLCPP_INFO(this->get_logger(), "Vehicle XY");
    RCLCPP_INFO(this->get_logger(), std::to_string(vehicle_x));
    RCLCPP_INFO(this->get_logger(), std::to_string(vehicle_y));
    RCLCPP_INFO(this->get_logger(), "Pursuit XY");
    RCLCPP_INFO(this->get_logger(), std::to_string(pursuit_x));
    RCLCPP_INFO(this->get_logger(), std::to_string(pursuit_y));
    RCLCPP_INFO(this->get_logger(), "Angle D H C");
    RCLCPP_INFO(this->get_logger(), std::to_string(desire_angle));
    RCLCPP_INFO(this->get_logger(), std::to_string(vehicle_head_angle));
    // RCLCPP_INFO(this->get_logger(), std::to_string(desire_angle-vehicle_head_angle));
    RCLCPP_INFO(this->get_logger(), std::to_string(control_angle));
    // RCLCPP_INFO(this->get_logger(), "");

    // Generate Message
    ackermann_msgs::msg::AckermannDriveStamped drive_msg;

    drive_msg.header = std_msgs::msg::Header();
    drive_msg.header.stamp = rclcpp::Clock().now();
    drive_msg.header.frame_id = "ego_racecar/laser";
    drive_msg.drive.steering_angle = control_angle;
    drive_msg.drive.speed = 3;

    publisher_->publish(drive_msg);
}

float VehicleController::angle_difference(float x1, float y1, float x2, float y2) {
    return atan2((y2-y1),(x2-x1));
}