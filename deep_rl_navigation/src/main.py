#!/usr/bin/env python

import rospy
from deep_rl_navigation.neuron_network import DeepReinforcementLearningNavigator

rospy.init_node("deep_rl_node", anonymous=False)

controller = DeepReinforcementLearningNavigator(rospy.get_name())
rospy.spin()
