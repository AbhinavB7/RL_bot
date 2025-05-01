from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty
from functools import partial
import numpy as np
import math
from gazebo_msgs.srv import DeleteEntity, SpawnEntity, SetModelState, SetEntityState
import os
from ament_index_python.packages import get_package_share_directory

class RobotController(Node):

    def __init__(self):
        super().__init__('robot_controller')
        self.get_logger().info("The robot controller node has just been created")

        self.action_pub = self.create_publisher(Twist, '/demo/cmd_vel', 10)
        self.pose_sub = self.create_subscription(Odometry, '/demo/odom', self.pose_callback, 1)
        self.laser_sub = self.create_subscription(LaserScan, '/demo/scan', self.laser_callback, 1)
        self.client_state = self.create_client(SetEntityState, "/demo/set_entity_state")      
        
        self._pkg_dir = os.path.join(
            get_package_share_directory("rl_bot"), "models",
        "turtlebot3_waffle", "model.sdf")

        self._agent_location = np.array([np.float32(1),np.float32(16)])
        self._laser_reads = np.array([np.float32(10)] * 61)
    
    def send_velocity_command(self, velocity):
        msg = Twist()
        msg.linear.x = float(velocity[0])
        msg.angular.z = float(velocity[1])
        self.action_pub.publish(msg)

    def pose_callback(self, msg: Odometry):
        self._agent_location = np.array([np.float32(np.clip(msg.pose.pose.position.x,-12,12)), np.float32(np.clip(msg.pose.pose.position.y,-35,21))])
        self._agent_orientation = 2* math.atan2(msg.pose.pose.orientation.z, msg.pose.pose.orientation.w)
        self._done_pose = True

    def laser_callback(self, msg: LaserScan):
        self._laser_reads = np.array(msg.ranges)

        self._laser_reads[self._laser_reads == np.inf] = np.float32(10)
        self._done_laser = True

    def call_set_robot_state_service(self, robot_pose=[1, 16, -0.707, 0.707]):
        while not self.client_state.wait_for_service(1.0):
            self.get_logger().warn("Waiting for service.1..")

        request = SetEntityState.Request()
        request.state.name = self.robot_name
        request.state.pose.position.x = float(robot_pose[0])
        request.state.pose.position.y = float(robot_pose[1])
        request.state.pose.orientation.z = float(robot_pose[2])
        request.state.pose.orientation.w = float(robot_pose[3])
        request.state.twist.linear.x = float(0)
        request.state.twist.linear.y = float(0)
        request.state.twist.linear.z = float(0)
        request.state.twist.angular.x = float(0)
        request.state.twist.angular.y = float(0)
        request.state.twist.angular.z = float(0)

        future = self.client_state.call_async(request)
        future.add_done_callback(partial(self.callback_set_robot_state))

    def callback_set_robot_state(self, future):
        try:
            response= future.result()
            self._done_set_rob_state = True
        except Exception as e:
            self.get_logger().error("Service call failed: %r" % (e,))

    def call_set_target_state_service(self, position=[1, 10]):
        while not self.client_state.wait_for_service(1.0):
            self.get_logger().warn("Waiting for service.2..")

        request = SetEntityState.Request()
        request.state.name = "Target"
        request.state.pose.position.x = float(position[0])
        request.state.pose.position.y = float(position[1])

        future = self.client_state.call_async(request)
        future.add_done_callback(partial(self.callback_set_target_state))

    def callback_set_target_state(self, future):
        try:
            response= future.result()
        except Exception as e:
            self.get_logger().error("Service call failed: %r" % (e,))
