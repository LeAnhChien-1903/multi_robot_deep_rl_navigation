#!/usr/bin/env python

import rospy
from deep_rl_navigation.agent import Agent

rospy.init_node("deep_rl_node", anonymous=False)

controller = Agent(rospy.get_name())
rospy.spin()
