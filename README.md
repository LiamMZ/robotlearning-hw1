# robotlearning-hw1
# Robotic Learning Framework: SAC and GAIL Implementation

## Overview

This project explores advanced reinforcement learning techniques, specifically the Soft Actor-Critic (SAC) and Generative Adversarial Imitation Learning (GAIL), to train a robotic agent to perform complex tasks. Leveraging the strengths of both algorithms, this framework aims to achieve efficient learning through exploration (SAC) and expert imitation (GAIL).

## SAC Implementation

### Core Components

- **Gaussian Policy and Deterministic Policy ([model.py](#))**: Implements stochastic and deterministic policies for action selection, facilitating exploration and exploitation.
- **Q-Network (Critic) ([model.py](#))**: Estimates the action-value function, employing two Q-networks to mitigate positive bias in policy improvement.
- **SAC Algorithm ([sac.py](#))**: Encapsulates the SAC algorithm's functionality, including network initialization, action selection, and parameter updates.
- **Replay Memory ([replay_memory.py](#))**: Stores and samples transition data, supporting experience replay for effective learning.
- **Utility Functions ([utils.py](#))**: Provides mathematical and update utilities for the SAC algorithm.

### Scripts

- **Training and Testing ([train.py](#), [test.py](#))**: Facilitate the training and evaluation processes, demonstrating the agent's performance in the environment.

## GAIL Implementation

### Normalization

- **RunningStat and ZFilter ([zfilter.py](#35))**: Perform input normalization and clipping, essential for stabilizing training and improving convergence.

### Utilities

- **Utility Functions ([utils.py](#34))**: Include action sampling, entropy calculation, log probability density computation, reward generation based on discriminator feedback, and model checkpointing.

### Model, Training, and Testing

- **Model ([model.py](#))**: Defines the architecture for the generator (policy network) and discriminator, central to the GAIL algorithm.
- **Training ([train.py](#), [train_model.py](#))**: Scripts for the GAIL training loop, detailing environment setup, model initialization, and learning process.
- **Testing ([test.py](#))**: Evaluates the trained model, assessing its ability to imitate expert behaviors in the environment.

## Getting Started

To run the training or testing scripts for either SAC or GAIL:
** WORK IN PROGRESS**
```bash
python train.py
python test.py

