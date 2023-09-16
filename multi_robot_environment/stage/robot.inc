
define robot_base position
(
    drive "diff" # Differential steering model.
    gui_nose 1 # Draw a nose on the robot so we can see which way it points

    gui_nose 1                  	# Draw a nose on the robot so we can see which way it points
    obstacle_return 1           	# Can hit things.
    ranger_return 0.5            	# reflects sonar beams

    # alternative odometric localization with simple error model
    localization "odom"         		# Change to "gps" to have impossibly perfect, global odometry
    odom_error [ 0.05 0.05 0.05 0.05 ]   	# Odometry error or slip in X, Y and Theta
                                        # (Uniform random distribution)   

    # Actual size
    size [0.9 0.6 0.4]
    # Center of rotation is offset from its center of area
    origin [0.0 0.0 0.0 0.0]
    # Estimate mass in Kg
    mass 100
)

