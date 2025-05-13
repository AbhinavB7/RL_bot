# RL Bot Project

This project contains a Reinforcement Learning (RL) training and inference pipeline for robot simulation in ROS 2 + Gazebo.

---

## Getting Started

Install the dependencies with pip

```bash
pip install -r requirements.txt
```

## Build the Workspace

Inside the container or local environment:

```bash
colcon build
source install/setup.bash
```

---

## Launch the Gazebo World

```bash
ros2 launch rl_bot gazebo_world.launch.py
```

---

## Start Training

Modify the executable in `start_training_launch.py` depending on the algorithm:

- For SAC: `executable='start_training_SAC'`
- For PPO (default): `executable='start_training'`

Then launch training:

```bash
ros2 launch rl_bot start_training_launch.py
```

---

## Start Inference

Modify the executable in `trained_agent.launch.py` depending on the algorithm:

- For SAC: `executable='trained_agent_SAC'`
- For PPO (default): `executable='trained_agent'`

Then launch inference:

```bash
ros2 launch rl_bot trained_agent.launch.py
```

---

## Pre-trained Models

Pre-trained models are available in the `rl_models` folder or can be downloaded from:

 [Download Models from Google Drive](https://drive.google.com/drive/folders/1WyVQ4GdqMRqLuDKZg-7Hs_6e12jkZgqs?usp=sharing)