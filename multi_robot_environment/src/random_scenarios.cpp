#include <ros/ros.h>
#include <iostream>
#include <random>
#include <fstream>
#include <cstdlib>
#include <Eigen/Core>
int main(int argc, char** argv)
{
    ros::init(argc, argv, ros::this_node::getName());
    ros::NodeHandle node;
    // Window parameters
    std::vector<double> window_size, window_center, window_rotate;
    double window_scale;
    bool show_data, show_grid;
    // Floor parameters
    std::string map_name;
    std::vector<double> floor_size;
    // Robot parameters
    std::vector<double> laser_pose;
    int number_of_robots;

    // Get param from ROS
    node.getParam("window_size", window_size);
    node.getParam("window_center", window_center);
    node.getParam("window_rotate", window_rotate);
    node.getParam("window_scale", window_scale);
    node.getParam("show_data", show_data);
    node.getParam("show_grid", show_grid);
    node.getParam("map_name", map_name);
    node.getParam("floor_size", floor_size);
    node.getParam("laser_pose", laser_pose);
    node.getParam("number_of_robots", number_of_robots);
    // Create a world file
    std::ofstream world_file("/home/leanhchien/deep_rl_ws/src/multi_robot_deep_rl_navigation/multi_robot_environment/stage/" + map_name + ".world");
    // Write header settings
    std::string header_settings = "include \"robot.inc\"\n"
                        "include \"sick.inc\"\n"
                        "define floor model\n"
                        "(\n"
                        "color \"gray30\"\n"
                        "boundary 1\n"
                        "gui_nose 0\n"
                        "gui_grid 0\n"
                        "gui_outline 0\n"
                        "gripper_return 0\n"
                        "fiducial_return 0\n"
                        "laser_return 1\n"
                        ")\n"
                        "resolution 0.025\n"
                        "interval_sim 100\n";
    // Write window settings
    std::string window_settings;
    window_settings.append("window\n");
    window_settings.append("(\n");
    window_settings.append("size [" + std::to_string(window_size[0]) + " " + std::to_string(window_size[1]) + "]\n");
    window_settings.append("center [" + std::to_string(window_center[0]) + " " + std::to_string(window_center[1]) + "]\n");
    window_settings.append("rotate [" + std::to_string(window_rotate[0]) + " " + std::to_string(window_rotate[1]) + "]\n");
    window_settings.append("scale " + std::to_string(window_scale) + "\n");
    window_settings.append("show_data " + std::to_string(show_data) + "\n");
    window_settings.append("show_grid " + std::to_string(show_grid) + "\n");
    window_settings.append(")\n");
    // Write floor settings
    std::string floor_settings;
    floor_settings.append("floor\n");
    floor_settings.append("(\n");
    floor_settings.append("name \"" + map_name + "\"\n");
    floor_settings.append("bitmap \"../maps/" + map_name + ".png\"" + "\n");
    floor_settings.append("size [" + std::to_string(floor_size[0]) + " " + std::to_string(floor_size[1]) + " " + std::to_string(floor_size[2]) + "]\n");
    floor_settings.append(")\n");
    // Write robot settings
    std::string robot_settings;
    robot_settings.append("define robot robot_base\n");
    robot_settings.append("(\n");
    robot_settings.append("sick_laser(pose [" + std::to_string(laser_pose[0]) + " " + std::to_string(laser_pose[1]) + " " + std::to_string(laser_pose[2]) + " " + std::to_string(laser_pose[3]) + "])\n");
    robot_settings.append(")\n");
    
    // Seed the random number generator.
    srand(time(NULL));
    // Random number generator
    std::random_device rd;
    std::mt19937 random_generator(rd());
    // Create a uniform real distribution
    std::uniform_real_distribution<double> x_distribution(double(-floor_size[0]/2 + 1), double(floor_size[0]/2) - 1);
    std::uniform_real_distribution<double> y_distribution(double(-floor_size[1]/2 + 1), double(floor_size[1]/2) - 1);
    std::uniform_real_distribution<double> theta_distribution(-180.0, 180.0);
    std::uniform_real_distribution<double> color_distribution(0.0, 1.0);
    std::vector<Eigen::Vector3d> robot_pose_vector;
    for (int i = 0; i < number_of_robots; i++)
    {
        robot_pose_vector.push_back(Eigen::Vector3d(x_distribution(random_generator), y_distribution(random_generator), theta_distribution(random_generator)));
        robot_settings.append("robot\n");
        robot_settings.append("(\n");
        robot_settings.append("name " + std::to_string(i) + "\n");
        robot_settings.append("pose [" + std::to_string(robot_pose_vector.back().x()) + " " + std::to_string(robot_pose_vector.back().y()) + " " + std::to_string(0.0) + " " + std::to_string(robot_pose_vector.back().z()) + "]\n");
        robot_settings.append("color_rgba [" + std::to_string(color_distribution(random_generator)) + " " + std::to_string(color_distribution(random_generator)) + " " + std::to_string(color_distribution(random_generator)) + " " + std::to_string(1.0) + "]\n");
        robot_settings.append(")\n");
    }
    // Write file
    world_file << header_settings;
    world_file << window_settings;
    world_file << floor_settings;
    world_file << robot_settings;
    world_file.close();

    // Write launch file for robot tf
    std::ofstream tf_file("/home/leanhchien/deep_rl_ws/src/multi_robot_deep_rl_navigation/multi_robot_environment/launch/robot_tf.launch");
    tf_file << "<launch>\n";
    // "<node pkg="tf" type="static_transform_publisher" name="base_link_to_laser" args="0.4 0 0 3.14 0 0 base_link laser 20"/>"
    for (int i = 0; i < number_of_robots; i++)
    {
        std::string tf_settings;
        
        tf_settings.append("\t<node pkg=\"tf\" type=\"static_transform_publisher\" ");
        tf_settings.append("name=\"map_to_odom_robot_" + std::to_string(i) + "\" ");
        tf_settings.append("args=\"" + std::to_string(robot_pose_vector[i].x()) + " " + std::to_string(robot_pose_vector[i].y()) + " "
                            + std::to_string(0.0) + " " + std::to_string(robot_pose_vector[i].z()) + " 0.0 0.0 map robot_" 
                            + std::to_string(i) + "/odom 20\"/>\n");
        tf_file << tf_settings ;
    }
    
    tf_file << "</launch>";

    tf_file.close();
    ros::spinOnce();
    return 0;
}