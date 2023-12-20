# Reinforcement Learning Simulation and Bot Control

## Overview

This repository contains code for a reinforcement learning simulation and a bot controller. The simulation environment is designed to allow a bot to learn a heuristic that helps it navigate towards a goal.

## Repository Structure

- **src/simulation**: This directory contains the code for the reinforcement learning simulation environment. The simulation environment includes the world, the goal, and the bot that learns to navigate towards the goal.

- **src/base_chalo**: The bot controller code resides in this directory. It includes the implementation of the reinforcement learning algorithm used to train the bot.

## Getting Started

Follow these steps to set up and run the reinforcement learning simulation:

   ```bash
   git clone https://gitlab.com/project-manas/ai/igvc_2024/navigation/rl-towards-goal.git
   mkdir -p rl_ws/src
   mv rl-towards-goal/* rl_ws/src/
   cd rl_ws
   catkin build
   ```
Source the workspace

To launch the simulation
  ```bash
  roslaunch simulation sim.launch # gui:=false, its better to not leave gazebo open
  ```

Train your bot
  ```bash
  rosrun base_chalo driver <name-of-file-without-extension>
  ```

Test your weights
  ```bash
  rosrun base_chalo custom_goal <name-of-file-without-extension>
  ```

Make it follow randomly generated path
  ```bash
  rosrun base_chalo follow_a_path <name-of-file-without-extension>
  ```
Some trained weights are stored under the folder "five"
- five/run1.pth 
- five/run2.pth # input_length = 4, output_length = 5 (five angular velocities)


