#include "lane_perception.h"

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LanePerception>());
  rclcpp::shutdown();
  return 0;
}