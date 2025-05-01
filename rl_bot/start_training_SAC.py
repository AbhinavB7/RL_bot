#!usr/bin/env python3

import rclpy
from rclpy.node import Node
from gymnasium.envs.registration import register
from rl_bot.bot_env import BotEnv
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import os
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

class TrainingNode(Node):

    def __init__(self):
        super().__init__("rl_bot_training", allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)

        self._training_mode = "training"

def main(args=None):

    rclpy.init()
    node = TrainingNode()
    node.get_logger().info("Training node has been created")

    home_dir = os.path.expanduser('~')
    pkg_dir = '690_ws/src/rl_bot'
    trained_models_dir = os.path.join(home_dir, pkg_dir, 'rl_models')
    log_dir = os.path.join(home_dir, pkg_dir, 'logs')
    
    # If the directories do not exist we create them
    if not os.path.exists(trained_models_dir):
        os.makedirs(trained_models_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    register(
        id="BotEnv-v0",
        entry_point="rl_bot.bot_env:BotEnv",
        max_episode_steps=300,
    )

    node.get_logger().info("The environment has been registered")

    env = gym.make('BotEnv-v0')
    env = Monitor(env)

    check_env(env)
    node.get_logger().info("Environment check finished")

    # Now we create two callbacks which will be executed during training
    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=900, verbose=1)
    eval_callback = EvalCallback(env, callback_on_new_best=stop_callback, eval_freq=100000, best_model_save_path=trained_models_dir, n_eval_episodes=40)
    
    if node._training_mode == "random_agent":
        # N° Episodes
        episodes = 10
        ## Execute a random agent
        node.get_logger().info("Starting the RANDOM AGENT now")
        for ep in range(episodes):
            obs = env.reset()
            done = False
            while not done:
                obs, reward, done, truncated, info = env.step(env.action_space.sample())
                node.get_logger().info("Agent state: [" + str(info["distance"]) + ", " + str(info["angle"]) + "]")
                node.get_logger().info("Reward at step " + ": " + str(reward))
    
    elif node._training_mode == "training":
        model = SAC(
            "MultiInputPolicy",
            env,
            verbose=1,
            tensorboard_log=log_dir,
            gamma=0.9880614935504514,
            ent_coef=0.00009689939917928778,
            learning_rate=0.00011770118633714448,
            buffer_size=1000000,  
            batch_size=256,      
            train_freq=1,         
            gradient_steps=1     
        )

        try:
            model.learn(total_timesteps=int(400000), reset_num_timesteps=False, callback=eval_callback, tb_log_name="SAC_test")
        except KeyboardInterrupt:
            model.save(f"{trained_models_dir}/SAC_test")
        # Save the trained model
        model.save(f"{trained_models_dir}/SAC_test")

    # Shutting down the node
    node.get_logger().info("The training is finished, now the node is destroyed")
    node.destroy_node()
    rclpy.shutdown()

def optimize_SAC(trial):
    return {
        'n_steps': trial.suggest_int('n_steps', 2048, 8192), # Default: 2048
        'gamma': trial.suggest_loguniform('gamma', 0.8, 0.9999), # Default: 0.99
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-6, 1e-3), # Default: 3e-4
        'clip_range': trial.suggest_uniform('clip_range', 0.1, 0.4), # Default: 0.02
        'gae_lambda': trial.suggest_uniform('gae_lambda', 0.8, 0.99), # Default: 0.95
        'ent_coef': trial.suggest_loguniform('ent_coef', 0.00000001, 0.1), # Default: 0.0
        'vf_coef': trial.suggest_uniform('vf_coef', 0, 1), # Default: 0.5
    }

def optimize_SAC_refinement(trial):
    return {
        'gamma': trial.suggest_float('gamma', 0.96, 0.9999, log=True),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
        'ent_coef': trial.suggest_float('ent_coef', 1e-8, 1e-5, log=True),
        'buffer_size': trial.suggest_categorical('buffer_size', [500000, 1000000]),
        'batch_size': trial.suggest_categorical('batch_size', [256, 512]),
        'tau': trial.suggest_float('tau', 0.005, 0.02),
        'train_freq': trial.suggest_categorical('train_freq', [1, 4, 8]),
        'gradient_steps': trial.suggest_categorical('gradient_steps', [1, 4, 8])
    }


def optimize_agent(trial):
    try:
        env_opt = gym.make('BotEnv-v0')
        # Setup dirs
        HOME_DIR = os.path.expanduser('~')
        PKG_DIR = '690_ws/src/rl_bot'
        LOG_DIR = os.path.join(HOME_DIR, PKG_DIR, 'logs')
        SAVE_PATH = os.path.join(HOME_DIR, PKG_DIR, 'tuning', 'trial_{}'.format(trial.number))
        
        model_params = optimize_SAC_refinement(trial)

        model = SAC("MultiInputPolicy", env_opt, tensorboard_log=LOG_DIR, verbose=0, **model_params)
        model.learn(total_timesteps=150000)

        mean_reward, _ = evaluate_policy(model, env_opt, n_eval_episodes=20)

        env_opt.close()
        del env_opt

        model.save(SAVE_PATH)

        return mean_reward

    except Exception as e:
        return -10000

if __name__ == "__main__":
    main()