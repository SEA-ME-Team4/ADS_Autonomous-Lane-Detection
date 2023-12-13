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
    "/perception/lane", 10, std::bind(&PathPlanner::lane_callback, this,  std::placeholders::_1));
    publisher_timer_ = this->create_wall_timer(
        // Problem with OpenGL when timer set 100ms
        std::chrono::milliseconds(10),
        std::bind(&PathPlanner::publisher_timer_callback, this)
    );

    ///////////////////////////////LANE DEBUG///////////////////////////////
    left_lane_publisher = this->create_publisher<nav_msgs::msg::GridCells>("/planner/left_lane", 10);
    right_lane_publisher = this->create_publisher<nav_msgs::msg::GridCells>("/planner/right_lane", 10);
    ///////////////////////////////LANE DEBUG///////////////////////////////
}

void PathPlanner::lane_callback(const sensor_msgs::msg::PointCloud2::SharedPtr lane_perception_msg){
    this->lane_perception_msg = lane_perception_msg;
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
            // Manhattan Distance
            distance = std::abs(current_x - xy_list[j][0]) + std::abs(current_y - xy_list[j][1]);
            // Euclidean Distance
            // distance = static_cast<int>(std::sqrt(std::pow((current_x - xy_list[j][0]), 2) + std::pow(current_y - xy_list[j][1], 2)));
            if (distance < resolution) {
                graph[i].push_back(j);
                graph[j].push_back(i);
            }
        }
    }

    // BFS for Making Lanes
    std::vector<bool> visited(length, false);
    std::vector<std::vector<int>> lanes = {};
    std::vector<int> lane;
    int lane_count;
    int current_point;
    std::vector<int> next_points;
    int next_point;
    for (int i = 0; i < static_cast<int>(length); ++i) {
        if (!visited[i]) {
            visited[i] = true;
            lane = {i};
            lane_count = 0;
            while (lane_count < static_cast<int>(lane.size())) {
                current_point = lane[lane_count];                
                next_points = graph[current_point];
                for (int j = 0; j < static_cast<int>(next_points.size()); ++j) {
                    next_point = next_points[j];
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
    int lane_y;
    for (int i = 0; i < static_cast<int>(lanes.size()); ++i) {
        // Get longest vector for each Lane
        // Now : Get Longest vector
        // is Better? : Get smallest x value vector considering Noise
        lane_y = xy_list[lanes[i][0]][1];
        if ((lane_y > 0) && (left_lane.size()<lanes[i].size())) {
            left_lane = lanes[i];
        }
        else if ((lane_y < 0) && (right_lane.size()<lanes[i].size())) {
            right_lane = lanes[i];
        }
    }

    // Path Planning
    int mid_x;
    int mid_y;
    std::vector<std::vector<int>> mid_xy_list;
    // Find Middle Lane Between two (If Both Lane Detected)
    if (!right_lane.empty() && !left_lane.empty()) {
        float interval = (static_cast<float>(right_lane.size())/left_lane.size());
        float j = 0;
        int current_left;
        int current_right;
        for (int i = 0; i < static_cast<int>(left_lane.size()); i++) {
            current_left = left_lane[i];
            current_right = right_lane[int(j)];
            mid_x = (xy_list[current_right][0] + xy_list[current_left][0]) / 2;
            mid_y = (xy_list[current_right][1] + xy_list[current_left][1]) / 2;
            mid_xy_list.push_back({mid_x, mid_y});
            j += interval;
        }
    }
    // Exception Handling for One Lane Detect Situation
    else {
        std::vector<int> only_lane;
        std::vector<int> offset = {resolution};
        if (!right_lane.empty()) {
            only_lane = right_lane;
            offset.push_back(resolution);
        }
        else if (!left_lane.empty()) {
            only_lane = left_lane;
            offset.push_back(-resolution);
        }
        // Todo : If no Lane Information
        else {
            return;
        }
        int current_point;
        for (int i = 0; i < static_cast<int>(only_lane.size()); i++) {
            current_point = only_lane[i];
            mid_x = (xy_list[current_point][0] + offset[0]) / 2;
            mid_y = (xy_list[current_point][1] + offset[1]) / 2;
            mid_xy_list.push_back({mid_x, mid_y});
        }
    }

    // Publish Path Message
    nav_msgs::msg::Path path_msg;
    path_msg.header = std_msgs::msg::Header();
    path_msg.header.stamp = rclcpp::Clock().now();
    path_msg.header.frame_id = "ego_racecar/laser";
    this->addPose(path_msg, 0.0, 0.0, 0.0);
    for (int i = 0; i < static_cast<int>(mid_xy_list.size()); ++i) {
        this->addPose(path_msg, mid_xy_list[i][0], mid_xy_list[i][1], 0.0);
    }
    publisher_->publish(path_msg);

    ///////////////////////////////LANE DEBUG///////////////////////////////
    nav_msgs::msg::GridCells left_lane_msg;
    left_lane_msg.header = std_msgs::msg::Header();
    left_lane_msg.header.stamp = rclcpp::Clock().now();
    left_lane_msg.header.frame_id = "ego_racecar/laser";
    left_lane_msg.cell_width = 0.1;
    left_lane_msg.cell_height = 0.1;
    for (int i = 0; i < static_cast<int>(left_lane.size()); ++i) {
        this->addCell(left_lane_msg, xy_list[left_lane[i]][0], xy_list[left_lane[i]][1], 0.0);
    }
    left_lane_publisher->publish(left_lane_msg);

    nav_msgs::msg::GridCells right_lane_msg;
    right_lane_msg.header = std_msgs::msg::Header();
    right_lane_msg.header.stamp = rclcpp::Clock().now();
    right_lane_msg.header.frame_id = "ego_racecar/laser";
    right_lane_msg.cell_width = 0.1;
    right_lane_msg.cell_height = 0.1;
    for (int i = 0; i < static_cast<int>(right_lane.size()); ++i) {
        this->addCell(right_lane_msg, xy_list[right_lane[i]][0], xy_list[right_lane[i]][1], 0.0);
    }
    right_lane_publisher->publish(right_lane_msg);
    ///////////////////////////////LANE DEBUG///////////////////////////////
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

    // pose_stamped.header.frame_id = "ego_racecar/laser";
    // pose_stamped.header.stamp = rclcpp::Clock().now();
    path_msg.poses.push_back(pose_stamped);
}


///////////////////////////////LANE DEBUG///////////////////////////////
void PathPlanner::addCell(nav_msgs::msg::GridCells& grid_cells_msg, int x, int y, int z) {
    geometry_msgs::msg::Point point;
    point.x = x / float_resolution;
    point.y = y / float_resolution;
    point.z = z / float_resolution;
    grid_cells_msg.cells.push_back(point);
}
///////////////////////////////LANE DEBUG///////////////////////////////