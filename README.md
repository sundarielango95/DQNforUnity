# DQNforUnity

This project was created to develop a RL model or agent to play a game developed in Unity. The NetZMQ module is used to communicate with the Unity game. 

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Contact](#contact)

## Overview

In this project, I worked on developing an RL agent that can interact with a game developed in the Unity environment. The Unity game is a simple 3d tennis game where the player or the agent has 2 controls - a forehand (FH) or backhand (BH) shot. It is a 2-player game where the computer controls the other player. Although the player only has 2 controls, the action space for the RL agent is 3 - [FH, BH, no_action]. The no_action control is included because most of the time, the agent does not need to do any action - for e.g., when the player is running to the ball, there is no output necessary from the agent as the game is coded in such a way that this part is self-controlled. Currently, the DQN model is a simple feedforward MLP model, with flipflop neurons governing the recurrent layer. It takes in position data of the ball and the player as the input and gives output as 0, 1, or 2, which is translated on the Unity side into appropriate actions. 

## Features

- Recurrent neural network with Flipflop neurons as a DQN
- Can be used to control a Unity game with a simple action space of 3
- Uses position data as input
- Can be modified to accept image data [code for this will be uploaded soon]

## Contact

For more information, or details, please reach out: sundarielango95@gmail.com
