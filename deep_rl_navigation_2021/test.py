import cv2

image = cv2.imread("/home/leanhchien/deep_rl_ws/src/multi_robot_deep_rl_navigation/multi_robot_environment/maps/plus_map.png", cv2.IMREAD_GRAYSCALE)

cv2.imwrite("/home/leanhchien/deep_rl_ws/src/multi_robot_deep_rl_navigation/multi_robot_environment/maps/plus_map.png", cv2.resize(image, (800, 800)))