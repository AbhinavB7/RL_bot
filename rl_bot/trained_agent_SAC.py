#!usr/bin/env python3

import rclpy
from rclpy.node import Node
from gymnasium.envs.registration import register
from rl_bot.bot_env import BotEnv
import gymnasium as gym
from stable_baselines3 import SAC  # CHANGED
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
import os
import numpy as np

class TrainedAgent(Node):

    def __init__(self):
        super().__init__("trained_bot", allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)

def main(args=None):
    rclpy.init()
    node = TrainedAgent()
    node.get_logger().info("Trained agent node has been created")

    home_dir = os.path.expanduser('~')
    pkg_dir = '690_ws/src/rl_bot'
    trained_model_path = os.path.join(home_dir, pkg_dir, 'rl_models', 'SAC_test_100k.zip')  

    register(
        id="botEnv-v0",
        entry_point="rl_bot.bot_env:BotEnv",
        max_episode_steps=300,  
    )

    env = gym.make('botEnv-v0')
    env = Monitor(env)

    check_env(env)

    custom_obj = {'action_space': env.action_space, 'observation_space': env.observation_space}

    model = SAC.load(trained_model_path, env=env, custom_objects=custom_obj)  

    Mean_ep_rew, Num_steps = evaluate_policy(
        model,
        env=env,
        n_eval_episodes=5,
        return_episode_rewards=True,
        deterministic=True 
    )

    node.get_logger().info("Mean Reward: " + str(np.mean(Mean_ep_rew)) + " - Std Reward: " + str(np.std(Mean_ep_rew)))
    node.get_logger().info("Max Reward: " + str(np.max(Mean_ep_rew)) + " - Min Reward: " + str(np.min(Mean_ep_rew)))
    node.get_logger().info("Mean episode length: " + str(np.mean(Num_steps)))

    env.close()

    node.get_logger().info("The script is completed, now the node is destroyed")
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
