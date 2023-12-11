#include "path_planner.h"

PathPlanner::PathPlanner() : rclcpp::Node("path_planner") {
    // Map Setting
    range = 5;
    map_size_x = 10;
    map_size_y = 10;
    resolution = 100;
    float_resolution = static_cast<float>(resolution);
    
    // Calculate resolution
    map_size_x *= resolution;
    map_size_y *= resolution;
    range *= resolution;

    publisher_ = this->create_publisher<nav_msgs::msg::Path>("/planner/path", 10);
    subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
    "/perception/lane", 10, std::bind(&PathPlanner::scan_callback, this,  std::placeholders::_1));
    publisher_timer_ = this->create_wall_timer(
        std::chrono::milliseconds(10),
        std::bind(&PathPlanner::publisher_timer_callback, this)
    );

    ///////////////////////////////LANE DEBUG///////////////////////////////
    ///////////////////////////////LANE DEBUG///////////////////////////////
    ///////////////////////////////LANE DEBUG///////////////////////////////
    debug_publisher_1 = this->create_publisher<nav_msgs::msg::Path>("/planner/path_debug1", 10);
    debug_publisher_2 = this->create_publisher<nav_msgs::msg::Path>("/planner/path_debug2", 10);
    ///////////////////////////////LANE DEBUG///////////////////////////////
    ///////////////////////////////LANE DEBUG///////////////////////////////
    ///////////////////////////////LANE DEBUG///////////////////////////////
}

void PathPlanner::publisher_timer_callback() {
    if (!lane_perception_msg) {
        return;
    }
    // x, y Storage
    sensor_msgs::PointCloud2Iterator<float> iter_x(*lane_perception_msg, "x");
    sensor_msgs::PointCloud2Iterator<float> iter_y(*lane_perception_msg, "y");
    size_t length = lane_perception_msg->width;

    int current_x;
    int current_y;
    std::vector<int> current_xy;
    std::vector<std::vector<int>> xy_list(length, std::vector<int>(2, 0));
    // +x = front
    // +y = left
    for (size_t i = 0; i < length; ++i) {
        current_x = static_cast<int>((*iter_x) * resolution);
        current_y = static_cast<int>((*iter_y) * resolution);
        current_xy = {current_x, current_y};
        xy_list[i] = current_xy;
        ++iter_x;
        ++iter_y;
    }

    // Sort list based on x value (min)
    std::sort(xy_list.begin(), xy_list.end(), small_x_value);

    // Relation Graph by Distance of Points for Lane Divide
    std::vector<std::vector<int>> graph(length, {{}});
    int distance;
    for (size_t i = 0; i < length; ++i) {
        current_xy = xy_list[i];
        current_x = current_xy[0];
        current_y = current_xy[1];
        for (size_t j = 0; j < i; ++j) {
            distance = static_cast<int>(std::sqrt(std::pow((current_x - xy_list[j][0]), 2) + std::pow(current_y - xy_list[j][1], 2)));
            if (distance < resolution) {
                graph[i].push_back(j);
                graph[j].push_back(i);
            }
        }
    }

    // BFS for Making Lanes
    std::vector<bool> visited(length, false);
    std::vector<std::vector<int>> lanes = {};
    for (int i = 0; i < static_cast<int>(length); ++i) {
        if (!visited[i]) {
            visited[i] = true;
            std::vector<int> lane = {i};
            int lane_count = 0;
            while (lane_count < static_cast<int>(lane.size())) {
                int current_point = lane[lane_count];                
                std::vector<int> next_points = graph[current_point];
                for (int j = 0; j < static_cast<int>(next_points.size()); ++j) {
                    int next_point = next_points[j];
                    if (!visited[next_point]) {
                        lane.push_back(next_point);
                        visited[next_point] = true;
                    }
                }
                lane_count += 1;
            }
            lanes.push_back(lane);
        }
    }

    // Find Left and Right Lane
    std::vector<int> left_lane = {};
    std::vector<int> right_lane = {};
    std::sort(lanes.begin(), lanes.end(), long_length);
    for (int i = 0; i < static_cast<int>(lanes.size()); ++i) {
        // Get longest vector for each Lane
        // Now : Get Longest vector
        // is Better? : Get smallest x value vector considering Noise
        int lane_y = xy_list[lanes[i][0]][1];
        if ((lane_y > 0) && (left_lane.size()<lanes[i].size())) {
            left_lane = lanes[i];
        }
        else if ((lane_y < 0) && (right_lane.size()<lanes[i].size())) {
            right_lane = lanes[i];
        }
    }

    // // Occupancy Grid Map
    // std::vector<std::vector<uint8_t>> occupancy_grid_map(map_size_x, std::vector<uint8_t>(map_size_y, 0));
    // // Add Lane Information 
    // for (size_t i = 0; i < lanes.size(); ++i) {
    //     for (size_t j = 0; j < lanes[i].size(); ++j) {
            
    //     }

    //     // int x = 500 - 100*iter_y[i];
    //     // int y = round(500 - 100*iter_x[i]);
    //     // occupancy_grid_map[x][y] = 100;

    // }

    // std::array<int, 2> start_point = {500, 500};
    // std::array<uint8_t, 2> end_point = {0,0};
    // publisher_->publish(path_msg);


    ///////////////////////////////LANE DEBUG///////////////////////////////
    ///////////////////////////////LANE DEBUG///////////////////////////////
    ///////////////////////////////LANE DEBUG///////////////////////////////
    nav_msgs::msg::Path lane_debug1_msg;
    lane_debug1_msg.header = std_msgs::msg::Header();
    lane_debug1_msg.header.stamp = rclcpp::Clock().now();
    lane_debug1_msg.header.frame_id = "ego_racecar/laser";
    for (int i = 0; i < static_cast<int>(left_lane.size()); ++i) {
        this->addPose(lane_debug1_msg, xy_list[left_lane[i]][0], xy_list[left_lane[i]][1], 0.0);
    }
    debug_publisher_1->publish(lane_debug1_msg);

    nav_msgs::msg::Path lane_debug2_msg;
    lane_debug2_msg.header = std_msgs::msg::Header();
    lane_debug2_msg.header.stamp = rclcpp::Clock().now();
    lane_debug2_msg.header.frame_id = "ego_racecar/laser";
    for (int i = 0; i < static_cast<int>(right_lane.size()); ++i) {
        this->addPose(lane_debug2_msg, xy_list[right_lane[i]][0], xy_list[right_lane[i]][1], 0.0);
    }
    debug_publisher_2->publish(lane_debug2_msg);
    ///////////////////////////////LANE DEBUG///////////////////////////////
    ///////////////////////////////LANE DEBUG///////////////////////////////
    ///////////////////////////////LANE DEBUG///////////////////////////////
}

void PathPlanner::scan_callback(const sensor_msgs::msg::PointCloud2::SharedPtr lane_perception_msg){
    this->lane_perception_msg = lane_perception_msg;
}

bool PathPlanner::small_x_value(const std::vector<int>& a, const std::vector<int>& b) {
    return a[0] < b[0];
}

bool PathPlanner::long_length(const std::vector<int>& a, const std::vector<int>& b) {
    return a.size() > b.size();
}

void PathPlanner::addPose(nav_msgs::msg::Path& path_msg, int x, int y, float theta) {
    geometry_msgs::msg::PoseStamped pose_stamped;
    pose_stamped.pose.position.x = x / float_resolution;
    pose_stamped.pose.position.y = y / float_resolution;
    
    tf2::Quaternion quat;
    quat.setRPY(0, 0, theta);
    pose_stamped.pose.orientation.x = quat.x();
    pose_stamped.pose.orientation.y = quat.y();
    pose_stamped.pose.orientation.z = quat.z();
    pose_stamped.pose.orientation.w = quat.w();

    pose_stamped.header.frame_id = "ego_racecar/laser";
    pose_stamped.header.stamp = rclcpp::Clock().now();
    path_msg.poses.push_back(pose_stamped);
}