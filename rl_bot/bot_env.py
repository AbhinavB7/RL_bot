import rclpy
from gymnasium import Env
from gymnasium.spaces import Dict, Box
import numpy as np
from rl_bot.robot_controller import RobotController
import math
#from rcl_interfaces.srv import GetParameters

class BotEnv(RobotController, Env):
    
    def __init__(self):
        
        # Initialize the Robot Controller Node
        super().__init__()
        self.get_logger().info("All the publishers/subscribers have been started")

        # ✅ WAIT FOR /demo/set_entity_state service before continuing
        self.get_logger().info("Waiting for /demo/set_entity_state service to become available...")
        if not self.client_state.wait_for_service(timeout_sec=10.0):
            self.get_logger().error("/demo/set_entity_state service not available after 10 seconds.")
        else:
            self.get_logger().info("/demo/set_entity_state service is now available.")

        # ENVIRONMENT PARAMETERS
        self.robot_name = 'turtlebot'
        self._target_location = np.array([1, 10], dtype=np.float32) # Default is [1, 10]
        self._initial_agent_location = np.array([1, 16, -90], dtype=np.float32) # Default is [1, 16, -90]

        self._randomize_env_level = 3
        self._normalize_obs = True
        self._normalize_act = True
        self._visualize_target = True
        self._reward_method = 1
        self._max_linear_velocity = 1
        self._min_linear_velocity = 0
        self._angular_velocity = 1

        self._minimum_dist_from_target = 0.42

        self._minimum_dist_from_obstacles = 0.26
        
        self._attraction_threshold = 3
        self._attraction_factor = 1
        self._repulsion_threshold = 1
        self._repulsion_factor = 0.1
        self._distance_penalty_factor = 1

        self._num_steps = 0

        self._num_episodes = 0

        # Debug prints on console
        self.get_logger().info("INITIAL TARGET LOCATION: " + str(self._target_location))
        self.get_logger().info("INITIAL AGENT LOCATION: " + str(self._initial_agent_location))
        self.get_logger().info("MAX LINEAR VEL: " + str(self._max_linear_velocity))
        self.get_logger().info("MIN LINEAR VEL: " + str(self._min_linear_velocity))
        self.get_logger().info("ANGULAR VEL: " + str(self._angular_velocity))
        self.get_logger().info("MIN TARGET DIST: " + str(self._minimum_dist_from_target))
        self.get_logger().info("MIN OBSTACLE DIST: " + str(self._minimum_dist_from_obstacles))

        # Warning for training
        if self._visualize_target == True:
            self.get_logger().info("WARNING! TARGET VISUALIZATION IS ACTIVATED, SET IT FALSE FOR TRAINING")

        if self._normalize_act == True:
            self.action_space = Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)

        else:
            self.action_space = Box(low=np.array([self._min_linear_velocity, -self._angular_velocity]), high=np.array([self._max_linear_velocity, self._angular_velocity]), dtype=np.float32)
        
        if self._normalize_obs == True:
            self.observation_space = Dict(
                {
                    "agent": Box(low=np.array([0, 0]), high=np.array([6, 1]), dtype=np.float32),
                    "laser": Box(low=0, high=1, shape=(61,), dtype=np.float32),
                }
            )

        else:
            self.observation_space = Dict(
                {
                    "agent": Box(low=np.array([0, -math.pi]), high=np.array([60, math.pi]), dtype=np.float32),
                    "laser": Box(low=0, high=np.inf, shape=(61,), dtype=np.float32),
                }
            )
        

        self._which_waypoint = 0

        self._successes = 0
        self._failures = 0
        self._completed_paths = 0

    def step(self, action):

        self._num_steps += 1

        if self._normalize_act == True:
            action = self.denormalize_action(action)

        self.send_velocity_command(action)

        self.spin()

        self.transform_coordinates()

        observation = self._get_obs()

        info = self._get_info()

        reward = self.compute_rewards(info)

        if (self._randomize_env_level <= 6.5):
            done = (info["distance"] < self._minimum_dist_from_target) or (any(info["laser"] < self._minimum_dist_from_obstacles))

        return observation, reward, done, False, info

    def render(self):

        pass

    def reset(self, seed=None, options=None):

        self._num_episodes += 1

        pose2d = self.randomize_robot_location()
        
        self._done_set_rob_state = False
        self.call_set_robot_state_service(pose2d)
        while self._done_set_rob_state == False:
            rclpy.spin_once(self)
        
        if (self._randomize_env_level >= 2):
            self.randomize_target_location()

        if self._visualize_target == True:
            self.call_set_target_state_service(self._target_location)

        self.spin()
        self.transform_coordinates()

        observation = self._get_obs()
        info = self._get_info()

        self._num_steps = 0

        return observation, info

    def _get_obs(self):
        # Returns the current state of the system
        obs = {"agent": self._polar_coordinates, "laser": self._laser_reads}
        # Normalize observations
        if self._normalize_obs == True:
            obs = self.normalize_observation(obs)
        #self.get_logger().info("Agent Location: " + str(self._agent_location))
        return obs

    def _get_info(self):
        # returns the distance from agent to target and laser reads
        return {
            "distance": math.dist(self._agent_location, self._target_location),
            "laser": self._laser_reads,
            "angle": self._theta
        }

    def spin(self):
        # This function spins the node until it gets new sensor data (executes both laser and odom callbacks)
        self._done_pose = False
        self._done_laser = False
        while (self._done_pose == False) or (self._done_laser == False):
            rclpy.spin_once(self)

    def transform_coordinates(self):

        # Radius
        self._radius = math.dist(self._agent_location, self._target_location)

        self._robot_target_x = math.cos(-self._agent_orientation)* \
            (self._target_location[0]-self._agent_location[0]) - \
                math.sin(-self._agent_orientation)*(self._target_location[1]-self._agent_location[1])
        
        self._robot_target_y = math.sin(-self._agent_orientation)* \
            (self._target_location[0]-self._agent_location[0]) + \
                math.cos(-self._agent_orientation)*(self._target_location[1]-self._agent_location[1])

        self._theta = math.atan2(self._robot_target_y, self._robot_target_x)

        self._polar_coordinates = np.array([self._radius,self._theta], dtype=np.float32)

    def randomize_target_location(self):

        if (self._randomize_env_level == 2) or (self._randomize_env_level == 3):
            self._target_location = np.array([1, 10], dtype=np.float32) # Base position [1,10]
            self._target_location[0] += np.float32(np.random.rand(1)*6-3) # Random contr. on target x ranges in [-3,+3]
            self._target_location[1] += np.float32(np.random.rand(1)*4-1) # Random contr. on target y ranges in [-1,+3]
            
    def randomize_robot_location(self):

        if (self._randomize_env_level == 0) or (self._randomize_env_level == 2):
            position_x = float(self._initial_agent_location[0])
            position_y = float(self._initial_agent_location[1])
            angle = float(math.radians(self._initial_agent_location[2]))
            orientation_z = float(math.sin(angle/2))
            orientation_w = float(math.cos(angle/2))

        if (self._randomize_env_level == 1) or (self._randomize_env_level == 3):
            # This method randomizes robot's initial position in a simple way
            position_x = float(1) + float(np.random.rand(1)*2-1) # Random contribution [-1,1]
            position_y = float(16) + float(np.random.rand(1) - 0.5) # Random contribution [-0.5,0.5]
            angle = float(math.radians(-90) + math.radians(np.random.rand(1)*60-30))
            orientation_z = float(math.sin(angle/2))
            orientation_w = float(math.cos(angle/2))

        return [position_x, position_y, orientation_z, orientation_w]
    
    def compute_rewards(self, info):

        if (info["distance"] < self._minimum_dist_from_target):
            reward = 1
            self.get_logger().info("TARGET REACHED")
        elif (any(info["laser"] < self._minimum_dist_from_obstacles)):
            reward = -1
            self.get_logger().info("HIT AN OBSTACLE")
        else:
            reward = 0
            
        return reward
    
    def normalize_observation(self, observation):

        observation["agent"][0] = observation["agent"][0]/10
        # Angle from target can range from -pi to pi
        observation["agent"][1] = (observation["agent"][1] + math.pi)/(2*math.pi)
        # Laser reads range from 0 to 10
        observation["laser"] = observation["laser"]/10


        
        return observation

    def denormalize_action(self, norm_act):

        action_linear = ((self._max_linear_velocity*(norm_act[0]+1)) + (self._min_linear_velocity*(1-norm_act[0])))/2
        # Angular velicity is symmetric
        action_angular = ((self._angular_velocity*(norm_act[1]+1)) + (-self._angular_velocity*(1-norm_act[1])))/2

        return np.array([action_linear, action_angular], dtype=np.float32)

    def compute_statistics(self, info):
        if (info["distance"] < self._minimum_dist_from_target):
                # If the agent reached the target it gets a positive reward
                self._successes += 1
                if (self._which_waypoint == len(self.waypoints_locations[0])-1):
                    self._completed_paths += 1
                #self.get_logger().info("Agent: X = " + str(self._agent_location[0]) + " - Y = " + str(self._agent_location[1]))
        elif (any(info["laser"] < self._minimum_dist_from_obstacles)):
                # If the agent hits an abstacle it gets a negative reward
                self._failures += 1
        else:
            pass

    def close(self):

        self.destroy_node()